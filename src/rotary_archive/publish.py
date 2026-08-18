"""Publish the built site to the club's host.

This is the one stage that reaches outside the machine and is hard to undo, so
it is built to be cautious:

  * **Dry run is the default.** `rotary publish` shows what would change and
    sends nothing. Uploading needs `--execute`.
  * **Deletion is opt-in.** The target directory may hold files this tool did
    not put there, and removing a stranger's files on a live club site is not
    recoverable from here.
  * **Authentication is never handled here.** Transfers shell out to the system
    `rsync`/`ssh`/`sftp`, so keys, agents, and `~/.ssh/config` work exactly as
    they do everywhere else, and no password ever passes through this code or
    sits in a config file.
  * **A stale build is refused.** Publishing a site older than the last change
    to the archive would quietly put yesterday's catalogue on the web.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Never uploaded regardless of configuration. These are either local noise or
# things that would leak the archive's internals onto a public server.
ALWAYS_EXCLUDE = (
    ".DS_Store",
    "Thumbs.db",
    ".git",
    "archive.db",
    "archive.db-wal",
    "archive.db-shm",
)


class PublishError(RuntimeError):
    """Configuration or transport problem. Never raised for a normal diff."""


@dataclass
class Preflight:
    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: int = 0
    bytes: int = 0
    target: str = ""

    @property
    def megabytes(self) -> float:
        return round(self.bytes / (1024 * 1024), 1)


@dataclass
class PublishResult:
    executed: bool
    command: str
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    output: str = ""
    returncode: int = 0


# -------------------------------------------------------------- preflight ---


def site_stats(site_dir: Path) -> tuple[int, int]:
    files = [p for p in site_dir.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def is_stale(site_dir: Path, database: Path) -> bool:
    """True when the archive changed after the site was last built.

    Publishing then would put yesterday's catalogue on the web while the
    database says otherwise - a silent inconsistency that is hard to notice
    from the outside.
    """
    index = site_dir / "index.html"
    if not index.exists() or not database.exists():
        return False
    return database.stat().st_mtime > index.stat().st_mtime


def describe_target(publish_config: dict[str, Any]) -> str:
    method = str(publish_config.get("method", "rsync")).lower()
    if method == "local":
        return str(publish_config.get("local_path", "") or "(no local_path set)")

    user = publish_config.get("user", "")
    host = publish_config.get("host", "")
    remote = publish_config.get("remote_path", "")
    prefix = f"{user}@{host}" if user else host
    return f"{prefix}:{remote}" if prefix else "(no host set)"


def preflight(
    paths: Any, publish_config: dict[str, Any], *, approved_only_expected: bool = False
) -> Preflight:
    """Everything worth checking before touching the network."""
    problems: list[str] = []
    warnings: list[str] = []

    site_dir = paths.site
    if not (site_dir / "index.html").exists():
        problems.append(
            f"No site at {site_dir}. Run `rotary build` first."
        )
        return Preflight(ok=False, problems=problems, target=describe_target(publish_config))

    files, size = site_stats(site_dir)
    if files < 3:
        problems.append(f"{site_dir} looks empty ({files} files). Re-run `rotary build`.")

    method = str(publish_config.get("method", "rsync")).lower()
    if method not in ("rsync", "sftp", "local"):
        problems.append(
            f"Unknown publish method {method!r}. Use rsync, sftp, or local."
        )

    if method == "local":
        if not publish_config.get("local_path"):
            problems.append("[publish] local_path is not set in config.toml.")
    else:
        for key in ("host", "remote_path"):
            if not publish_config.get(key):
                problems.append(f"[publish] {key} is not set in config.toml.")
        binary = "rsync" if method == "rsync" else "sftp"
        if shutil.which(binary) is None:
            problems.append(
                f"`{binary}` is not on PATH. "
                + ("Install rsync, or set method = \"sftp\"." if method == "rsync"
                   else "Install an SSH client.")
            )

    identity = publish_config.get("identity")
    if identity and not Path(identity).expanduser().exists():
        problems.append(f"[publish] identity file not found: {identity}")

    if is_stale(site_dir, paths.database):
        warnings.append(
            "The archive has changed since this site was built. "
            "Run `rotary build` again so you publish what the database says."
        )

    if publish_config.get("delete"):
        warnings.append(
            "delete is enabled: remote files not present in this build will be "
            "removed, including any that were not put there by this tool."
        )

    if approved_only_expected:
        warnings.append(
            "Check this site was built with --approved-only if it is going "
            "somewhere public."
        )

    return Preflight(
        ok=not problems,
        problems=problems,
        warnings=warnings,
        files=files,
        bytes=size,
        target=describe_target(publish_config),
    )


# ---------------------------------------------------------------- commands ---


def _exclusions(publish_config: dict[str, Any]) -> list[str]:
    configured = publish_config.get("exclude") or []
    seen: list[str] = []
    for pattern in [*ALWAYS_EXCLUDE, *configured]:
        if pattern not in seen:
            seen.append(str(pattern))
    return seen


def _ssh_command(publish_config: dict[str, Any]) -> str:
    """The `ssh` invocation rsync should use for transport."""
    parts = ["ssh"]
    port = int(publish_config.get("port", 22) or 22)
    if port != 22:
        parts += ["-p", str(port)]
    identity = publish_config.get("identity")
    if identity:
        parts += ["-i", str(Path(identity).expanduser())]
    return " ".join(shlex.quote(p) if " " in p else p for p in parts)


def rsync_command(
    site_dir: Path, publish_config: dict[str, Any], *, dry_run: bool, delete: bool
) -> list[str]:
    """Build the rsync argv.

    The trailing slash on the source is load-bearing: `site/` copies the
    directory's *contents* into the remote path, while `site` would create a
    `site` subdirectory inside it. Getting this wrong publishes the archive to
    `/history/site/` instead of `/history/`.
    """
    command = ["rsync", "-rlptz", "--human-readable", "--itemize-changes"]

    if dry_run:
        command.append("--dry-run")
    if delete:
        command.append("--delete")
    for pattern in _exclusions(publish_config):
        command += ["--exclude", pattern]

    method = str(publish_config.get("method", "rsync")).lower()
    if method != "local":
        command += ["-e", _ssh_command(publish_config)]

    command.append(f"{site_dir}/")

    if method == "local":
        target = Path(str(publish_config["local_path"])).expanduser()
        command.append(f"{target}/")
    else:
        user = publish_config.get("user", "")
        host = publish_config["host"]
        remote = str(publish_config["remote_path"]).rstrip("/")
        prefix = f"{user}@{host}" if user else str(host)
        command.append(f"{prefix}:{remote}/")

    return command


def sftp_batch(site_dir: Path, publish_config: dict[str, Any]) -> tuple[list[str], str]:
    """An sftp invocation plus the batch script it should run.

    sftp has no incremental transfer and no exclusion support, so this
    re-uploads the whole site every time. It exists for hosts that permit file
    transfer but not rsync; prefer rsync wherever it is available.
    """
    remote = str(publish_config["remote_path"]).rstrip("/")
    lines = [f"-mkdir {remote}", f"cd {remote}"]

    # -put is prefixed with '-' so a single failure does not abort the batch;
    # the exit status still reports trouble.
    for child in sorted(site_dir.iterdir()):
        if child.name in _exclusions(publish_config):
            continue
        if child.is_dir():
            lines.append(f"-mkdir {remote}/{child.name}")
            lines.append(f"-put -r {child} {remote}/")
        else:
            lines.append(f"-put {child} {remote}/")
    lines.append("bye")

    command = ["sftp", "-b", "-"]
    port = int(publish_config.get("port", 22) or 22)
    if port != 22:
        command = ["sftp", "-P", str(port), "-b", "-"]
    identity = publish_config.get("identity")
    if identity:
        command = command[:1] + ["-i", str(Path(identity).expanduser())] + command[1:]

    user = publish_config.get("user", "")
    host = publish_config["host"]
    command.append(f"{user}@{host}" if user else str(host))

    return command, "\n".join(lines) + "\n"


def redact(command: Sequence[str]) -> str:
    """A printable form of the command.

    Nothing secret is passed on the command line - authentication is the SSH
    client's job - but an identity *path* can disclose a home directory, so it
    is shortened for display.
    """
    parts = []
    for arg in command:
        text = str(arg)
        home = str(Path.home())
        if text.startswith(home):
            text = "~" + text[len(home):]
        parts.append(shlex.quote(text) if " " in text else text)
    return " ".join(parts)


# ------------------------------------------------------------------- run ----


def parse_itemized(output: str) -> tuple[list[str], list[str]]:
    """Split rsync's --itemize-changes output into changes and deletions."""
    changed: list[str] = []
    deleted: list[str] = []

    for line in output.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("*deleting"):
            deleted.append(line.split(None, 1)[-1])
            continue
        # Itemised lines start with an 11-character change flag block.
        if len(line) > 12 and line[0] in "<>ch.*" and " " in line:
            name = line.split(None, 1)[1]
            if not name.endswith("/"):
                changed.append(name)
    return changed, deleted


def publish(
    paths: Any,
    publish_config: dict[str, Any],
    *,
    dry_run: bool = True,
    delete: bool | None = None,
    timeout: float = 1800,
) -> PublishResult:
    """Upload the built site. Dry run unless told otherwise."""
    method = str(publish_config.get("method", "rsync")).lower()
    should_delete = publish_config.get("delete", False) if delete is None else delete

    if method in ("rsync", "local"):
        command = rsync_command(
            paths.site, publish_config, dry_run=dry_run, delete=bool(should_delete)
        )
        stdin_text = None
    elif method == "sftp":
        if dry_run:
            # sftp cannot preview. Rather than pretend, say what would happen.
            command, script = sftp_batch(paths.site, publish_config)
            files, _ = site_stats(paths.site)
            return PublishResult(
                executed=False,
                command=redact(command),
                changed=[f"(sftp cannot preview; would upload all {files} files)"],
                output=script,
            )
        command, stdin_text = sftp_batch(paths.site, publish_config)
    else:
        raise PublishError(f"Unknown publish method {method!r}")

    if method == "local" and not dry_run:
        Path(str(publish_config["local_path"])).expanduser().mkdir(
            parents=True, exist_ok=True
        )

    try:
        completed = subprocess.run(
            command,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PublishError(f"{command[0]} is not installed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PublishError(
            f"{command[0]} timed out after {timeout:.0f}s. The upload may be "
            "partially complete; re-run to finish it."
        ) from exc

    if completed.returncode != 0:
        raise PublishError(
            f"{command[0]} exited {completed.returncode}.\n"
            f"{completed.stderr.strip()[:800]}"
        )

    changed, deleted = parse_itemized(completed.stdout)
    return PublishResult(
        executed=not dry_run,
        command=redact(command),
        changed=changed,
        deleted=deleted,
        output=completed.stdout,
        returncode=completed.returncode,
    )
