"""Publish stage.

The properties under test are safety properties. Publishing reaches outside the
machine and cannot be undone from here, so the things that must hold are:
nothing leaves without an explicit instruction, nothing is deleted without a
second explicit instruction, and no secret is ever handled by this code.

`method = "local"` drives the same rsync code path against a directory standing
in for the club's host, so the real command is exercised without a server.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rotary_archive.publish import (
    ALWAYS_EXCLUDE,
    PublishError,
    describe_target,
    is_stale,
    parse_itemized,
    preflight,
    publish,
    redact,
    rsync_command,
    sftp_batch,
    site_stats,
)

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="rsync is not installed"
)


@pytest.fixture
def built(project):
    """A minimal built site plus a local directory standing in for the host."""
    site = project.paths.site
    (site / "assets").mkdir(parents=True, exist_ok=True)
    (site / "media").mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text("<!doctype html><title>Archive</title>")
    (site / "embed.html").write_text("<!-- embed -->")
    (site / "assets" / "app.js").write_text("window.x = 1;")
    (site / "media" / "a-800.webp").write_bytes(b"webp-bytes")

    remote = project.paths.root / "fakehost"
    remote.mkdir(parents=True, exist_ok=True)
    return project, remote


def local_config(remote: Path, **overrides):
    return {
        "method": "local",
        "local_path": str(remote),
        "exclude": [],
        "delete": False,
        **overrides,
    }


# ------------------------------------------------------------------ target --


def test_describe_target_for_a_remote_host():
    assert describe_target(
        {"method": "rsync", "user": "ken", "host": "club.org",
         "remote_path": "/var/www/history"}
    ) == "ken@club.org:/var/www/history"


def test_describe_target_without_a_user():
    assert describe_target(
        {"method": "rsync", "host": "club.org", "remote_path": "/history"}
    ) == "club.org:/history"


def test_describe_target_says_so_when_unconfigured():
    assert "no host set" in describe_target({"method": "rsync"})


# --------------------------------------------------------------- preflight --


def test_preflight_passes_on_a_good_build(built):
    project, remote = built
    checks = preflight(project.paths, local_config(remote))
    assert checks.ok
    assert checks.files == 4
    assert not checks.problems


def test_preflight_refuses_when_nothing_is_built(project):
    checks = preflight(project.paths, {"method": "local", "local_path": "/tmp/x"})
    assert not checks.ok
    assert any("rotary build" in p for p in checks.problems)


def test_preflight_requires_a_host_for_remote_methods(built):
    project, _ = built
    checks = preflight(project.paths, {"method": "rsync", "exclude": []})
    assert not checks.ok
    assert any("host" in p for p in checks.problems)
    assert any("remote_path" in p for p in checks.problems)


def test_preflight_rejects_an_unknown_method(built):
    project, remote = built
    checks = preflight(project.paths, local_config(remote, method="carrier-pigeon"))
    assert not checks.ok
    assert any("Unknown publish method" in p for p in checks.problems)


def test_preflight_rejects_a_missing_identity_file(built):
    project, remote = built
    checks = preflight(
        project.paths,
        {"method": "rsync", "host": "h", "remote_path": "/p",
         "identity": "/nope/id_rsa", "exclude": []},
    )
    assert any("identity file not found" in p for p in checks.problems)


def test_preflight_warns_when_delete_is_enabled(built):
    project, remote = built
    checks = preflight(project.paths, local_config(remote, delete=True))
    assert checks.ok
    assert any("delete is enabled" in w for w in checks.warnings)


def test_stale_build_is_detected(built):
    """Publishing a site older than the archive would put yesterday's
    catalogue on the web while the database says otherwise."""
    project, remote = built
    index = project.paths.site / "index.html"

    import os, time
    old = time.time() - 3600
    os.utime(index, (old, old))
    project.paths.database.write_bytes(b"changed")

    assert is_stale(project.paths.site, project.paths.database)
    checks = preflight(project.paths, local_config(remote))
    assert any("changed since" in w for w in checks.warnings)


def test_fresh_build_is_not_stale(built):
    project, remote = built
    project.paths.database.write_bytes(b"x")
    import os, time
    now = time.time() + 10
    os.utime(project.paths.site / "index.html", (now, now))
    assert not is_stale(project.paths.site, project.paths.database)


# ----------------------------------------------------------------- command --


def test_source_path_has_a_trailing_slash(built):
    """Without it rsync creates a `site` subdirectory inside the target, so
    the archive lands at /history/site/ instead of /history/."""
    project, remote = built
    command = rsync_command(
        project.paths.site, local_config(remote), dry_run=True, delete=False
    )
    source = next(a for a in command if a.endswith("site/"))
    assert source.endswith("/site/")


def test_dry_run_flag_is_present_only_for_a_dry_run(built):
    project, remote = built
    dry = rsync_command(project.paths.site, local_config(remote), dry_run=True, delete=False)
    wet = rsync_command(project.paths.site, local_config(remote), dry_run=False, delete=False)
    assert "--dry-run" in dry
    assert "--dry-run" not in wet


def test_delete_flag_is_present_only_when_asked(built):
    project, remote = built
    without = rsync_command(project.paths.site, local_config(remote), dry_run=True, delete=False)
    with_it = rsync_command(project.paths.site, local_config(remote), dry_run=True, delete=True)
    assert "--delete" not in without
    assert "--delete" in with_it


def test_always_excluded_patterns_cannot_be_configured_away(built):
    """The archive database must never be uploaded to a public web server,
    whatever the config says."""
    project, remote = built
    command = rsync_command(
        project.paths.site, local_config(remote, exclude=["only-this"]),
        dry_run=True, delete=False,
    )
    for pattern in ALWAYS_EXCLUDE:
        assert pattern in command
    assert "only-this" in command


def test_ssh_transport_is_used_for_remote_methods(built):
    project, _ = built
    command = rsync_command(
        project.paths.site,
        {"method": "rsync", "host": "club.org", "remote_path": "/history",
         "user": "ken", "port": 2222, "exclude": []},
        dry_run=True, delete=False,
    )
    assert "-e" in command
    transport = command[command.index("-e") + 1]
    assert transport.startswith("ssh")
    assert "-p 2222" in transport
    assert command[-1] == "ken@club.org:/history/"


def test_no_ssh_transport_for_a_local_target(built):
    project, remote = built
    command = rsync_command(
        project.paths.site, local_config(remote), dry_run=True, delete=False
    )
    assert "-e" not in command


def test_redact_shortens_home_paths():
    """An identity path can disclose a home directory; nothing secret is on
    the command line, but the display is tightened anyway."""
    line = redact(["rsync", "-i", str(Path.home() / ".ssh" / "id_ed25519")])
    assert str(Path.home()) not in line
    assert "~/.ssh/id_ed25519" in line


def test_no_password_is_ever_accepted(built):
    """Authentication is the SSH client's job. A password in config.toml would
    be a password in the user's backups and git history."""
    project, remote = built
    command = rsync_command(
        project.paths.site,
        local_config(remote, password="hunter2", user="ken"),
        dry_run=True, delete=False,
    )
    assert not any("hunter2" in str(arg) for arg in command)


# ------------------------------------------------------------------ itemize --


def test_parse_itemized_splits_changes_from_deletions():
    output = "\n".join([
        ">f+++++++++ index.html",
        ">f.st...... assets/app.js",
        "cd+++++++++ media/",
        "*deleting   robots.txt",
    ])
    changed, deleted = parse_itemized(output)
    assert changed == ["index.html", "assets/app.js"]
    assert deleted == ["robots.txt"]


def test_parse_itemized_of_empty_output():
    assert parse_itemized("") == ([], [])


# ------------------------------------------------------------------ upload --


def test_dry_run_sends_nothing(built):
    project, remote = built
    result = publish(project.paths, local_config(remote), dry_run=True)

    assert result.executed is False
    assert result.changed, "a dry run should still report what would change"
    assert list(remote.iterdir()) == [], "the dry run delivered files"


def test_execute_uploads_every_file(built):
    project, remote = built
    result = publish(project.paths, local_config(remote), dry_run=False)

    assert result.executed is True
    assert (remote / "index.html").exists()
    assert (remote / "assets" / "app.js").exists()
    assert (remote / "media" / "a-800.webp").read_bytes() == b"webp-bytes"


def test_republishing_unchanged_content_transfers_nothing(built):
    project, remote = built
    publish(project.paths, local_config(remote), dry_run=False)

    again = publish(project.paths, local_config(remote), dry_run=True)
    assert again.changed == []


def test_only_changed_files_are_sent(built):
    project, remote = built
    publish(project.paths, local_config(remote), dry_run=False)

    (project.paths.site / "index.html").write_text("<!doctype html><title>New</title>")
    result = publish(project.paths, local_config(remote), dry_run=True)

    assert result.changed == ["index.html"]


def test_delete_off_leaves_unknown_remote_files_alone(built):
    """The target may hold files this tool did not put there. Removing a
    stranger's files on a live club site is not recoverable from here."""
    project, remote = built
    publish(project.paths, local_config(remote), dry_run=False)
    stranger = remote / "robots.txt"
    stranger.write_text("User-agent: *")

    publish(project.paths, local_config(remote), dry_run=False, delete=False)
    assert stranger.exists()


def test_delete_requires_execute_as_well(built):
    """--delete on its own is still a preview. Two explicit instructions are
    needed before anything is removed."""
    project, remote = built
    publish(project.paths, local_config(remote), dry_run=False)
    stranger = remote / "robots.txt"
    stranger.write_text("User-agent: *")

    result = publish(project.paths, local_config(remote), dry_run=True, delete=True)
    assert "robots.txt" in result.deleted
    assert stranger.exists(), "a dry run deleted a remote file"


def test_delete_when_executed_removes_unknown_files(built):
    project, remote = built
    publish(project.paths, local_config(remote), dry_run=False)
    stranger = remote / "robots.txt"
    stranger.write_text("User-agent: *")

    publish(project.paths, local_config(remote), dry_run=False, delete=True)
    assert not stranger.exists()


def test_explicit_delete_argument_overrides_config(built):
    project, remote = built
    publish(project.paths, local_config(remote), dry_run=False)
    stranger = remote / "robots.txt"
    stranger.write_text("x")

    # Config says delete, the call says no.
    publish(
        project.paths, local_config(remote, delete=True),
        dry_run=False, delete=False,
    )
    assert stranger.exists()


def test_unknown_method_raises(built):
    project, remote = built
    with pytest.raises(PublishError):
        publish(project.paths, local_config(remote, method="smoke-signal"))


# -------------------------------------------------------------------- sftp --


def test_sftp_dry_run_admits_it_cannot_preview(built):
    """sftp has no dry-run mode. Saying so is better than printing a fake
    preview that does not reflect what would happen."""
    project, _ = built
    result = publish(
        project.paths,
        {"method": "sftp", "host": "club.org", "remote_path": "/history",
         "exclude": []},
        dry_run=True,
    )
    assert result.executed is False
    assert any("cannot preview" in line for line in result.changed)


def test_sftp_batch_creates_the_remote_directory_first(built):
    project, _ = built
    command, script = sftp_batch(
        project.paths.site,
        {"method": "sftp", "host": "club.org", "remote_path": "/history",
         "exclude": []},
    )
    lines = script.splitlines()
    assert lines[0] == "-mkdir /history"
    assert lines[1] == "cd /history"
    assert lines[-1] == "bye"
    assert command[0] == "sftp"
    assert command[-1] == "club.org"


def test_sftp_batch_honours_exclusions(built):
    project, _ = built
    (project.paths.site / ".DS_Store").write_bytes(b"junk")
    _, script = sftp_batch(
        project.paths.site,
        {"method": "sftp", "host": "h", "remote_path": "/p", "exclude": []},
    )
    assert ".DS_Store" not in script


# ------------------------------------------------------------------- stats --


def test_site_stats_counts_files_and_bytes(built):
    project, _ = built
    files, size = site_stats(project.paths.site)
    assert files == 4
    assert size > 0
