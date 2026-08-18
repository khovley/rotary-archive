# Putting the archive on the club's website

The archive is a folder of ordinary files — HTML, CSS, one JavaScript file, and
images. No server code, no database, no PHP. Any host that can serve static
files can serve it.

There are two steps: get the files onto the host, then show them on a page.

---

## Before you publish

**Build with `--approved-only`.** Without it, `rotary build` includes everything
that has been analysed, whether or not you have reviewed it. That is right for
previewing and wrong for a public site.

```bash
rotary build --approved-only --serve
```

That opens the site locally so you can check it. Look for:

- Anything you would not want a member's family to read
- Names on photographs — the archive only names people whose names appear in
  writing on the item, but it is worth confirming
- Dates marked *about* — those were deduced, not printed, and the reasoning is
  shown beside them

**Set the club's details** in `config.toml` first:

```toml
[site]
club_name = "Rotary Club of Brookfield"
tagline   = "Seventy years of service, in clippings and photographs"
contact   = "history@brookfieldrotary.org"
```

`contact` appears on unidentified photographs, so members know how to tell you
who is in them. That is how a club archive actually gets identified.

---

## Step 1: get the files onto the host

### If the host offers SSH (most do)

```toml
[publish]
method      = "rsync"
host        = "brookfieldrotary.org"
user        = "your-ssh-username"
remote_path = "/home/your-ssh-username/public_html/history"
```

Then:

```bash
rotary publish              # preview: shows what would change, sends nothing
rotary publish --execute    # upload
```

**`rotary publish` is a dry run by default.** It prints exactly which files
would be uploaded and which, if any, would be removed. Nothing leaves your
machine until you add `--execute`, and even then it asks once more.

rsync transfers only what changed, so after the first upload subsequent
publishes take seconds.

### If the host has no rsync

```toml
method = "sftp"
```

Everything else is the same. sftp cannot preview and re-uploads the whole site
each time, so it is slower — but it works where rsync is not installed.

### If you sync some other way

```toml
method     = "local"
local_path = "/Users/you/Dropbox/club-site/history"
```

Copies into a local folder; whatever syncs that folder does the rest.

### Authentication

There is deliberately **no password setting** anywhere in the configuration. A
password in `config.toml` is a password in your backups and, if the project is
ever pushed anywhere, in your git history.

Transfers use the system `ssh`, so authentication works exactly as it does when
you run `ssh` yourself:

- An SSH key in `~/.ssh/` with an agent loaded — nothing to configure
- A specific key: set `identity = "~/.ssh/club_key"`
- A password: `ssh` will prompt you directly

If `ssh you@host` works in your terminal, `rotary publish` will work.

---

## Step 2: show it on a WordPress page

The built site includes `site/embed.html` with a ready-made snippet and
instructions. In short:

1. Create or edit the page where the archive should appear
2. Add an Elementor **HTML** widget
3. Paste everything between the `SNIPPET STARTS/ENDS HERE` markers
4. Change `src="/history/"` if you uploaded somewhere else

The snippet includes a small script that resizes the frame to fit its content.
Without it the archive either scrolls inside a short box or gets cut off.

### Why an iframe rather than a plugin

The archive is self-contained and has no WordPress dependencies. An iframe
keeps it that way: a theme change, a plugin conflict, or a WordPress upgrade
cannot break it, and the archive can be moved to a different host later without
touching the page it is embedded in.

Deep links work inside the frame — the archive uses hash routing
(`#/item/…`, `#/person/…`), which needs no server rewrite rules.

---

## Updating the archive later

The whole cycle, once more material is photographed:

```bash
cp ~/Pictures/new-batch/*.HEIC inbox/
rotary run                       # ingest, segment, rectify, then review
rotary analyze                   # read the new items
rotary review                    # approve them
rotary build --approved-only
rotary publish                   # preview
rotary publish --execute
```

Every stage skips work already done, so this only processes what is new.

---

## Things worth knowing

**Masters never leave your machine.** Only web-optimised WebP derivatives are
published. The full-resolution archival scans stay in `masters/` — back that
folder up separately, since it is the irreplaceable part.

**Nothing is deleted on the host unless you ask.** The target directory may
contain files this tool did not put there. `--delete` removes remote files not
in your build, and it still needs `--execute` alongside it.

**Publishing a stale build is caught.** If the archive changed after the site
was last built, `rotary publish` warns you before doing anything.

**The archive works offline.** `site/index.html` opens directly from disk with
no server. Useful for checking a build, and for handing the whole thing to
someone on a USB stick.

**No external requests.** No fonts, no analytics, no CDN. Nothing to break when
a service disappears in ten years, and nobody browsing the club's history is
tracked doing it.
