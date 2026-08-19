# How to photograph the collection

Read this before you shoot the bulk of the material. Ten minutes of setup here
saves hours of correcting crops later.

The single biggest factor in whether the software finds items automatically is
**how much the background contrasts with what you lay on it.** On the built-in
benchmark, a contrasting background finds 100% of items with tight crops. A
pale background under pale paper drops crop accuracy noticeably and puts items
into the review queue.

---

## Setup

**Background.** One sheet of matte poster board, larger than your biggest item.

| What you're shooting | Use |
|---|---|
| Newspaper clippings, letters, programs, certificates | **Black** or very dark |
| Dark photographs, dark mounts, black card | **White** or very light |

Matte, not glossy — gloss reflects the ceiling light back at the lens and
creates bright patches the detector reads as edges. Black foam board or a
non-reflective black cloth both work well. Avoid wood grain, tablecloths with
patterns, and anything with a printed texture.

**Light.** Bright, even, indirect. Near a window on an overcast day is ideal.
Two lamps at 45° from opposite sides also works.

- **Turn the flash off.** It blows out the centre and hard-edges the shadows.
- Watch for your own shadow falling across the items — stand to the side.
- One soft shadow under each item is fine and expected. A hard shadow that
  *bridges the gap* between two items can make them read as one, which is the
  one error the software cannot recover from.

**Camera.** Your iPhone is fine. Highest resolution, HEIC or JPEG both work.

**Getting the photos off the phone without shrinking them.** This is easy to
get wrong and hard to notice. A 12MP iPhone photo of eight clippings gives each
one about 1400 pixels across, which is roughly what newsprint needs to be
transcribed and to stay readable when a visitor zooms in. Dragging a photo out
of the macOS Photos window, or emailing it to yourself, can hand you a 1024px
copy instead — 0.8MP, about 6% of the original, and each clipping only 240
pixels wide. The crop will look fine as a thumbnail and turn to mush the moment
anyone zooms.

Use **AirDrop**, or **File → Export → Export Unmodified Original** in Photos,
or copy straight off the phone with Image Capture. `rotary ingest` warns you
when a photo arrives under 4MP — take the warning seriously and re-export.

Resolution affects *legibility*, not detection: on the built-in benchmark the
detector finds every item with tight crops even at 0.9MP. Contrast and spacing
are what decide whether the crops are right; resolution decides whether the
result is worth reading.

---

## Laying out the items

- **6 to 10 items per photo** is the sweet spot. More than that and each item
  gets too few pixels to transcribe reliably.
- **Leave a clear gap** — at least a finger's width — between items. This is
  the single most common reason a batch comes out badly.
- **Don't overlap anything.** If an item is partly hidden, the hidden part is
  gone from the archive — no software recovers it.
- Keep everything **fully inside the frame**, with a margin of background all
  the way around. An item running off the edge gets cropped short.
- Group loosely by row. The software numbers items top-to-bottom then
  left-to-right, so a tidy grid makes the results easier to check.

> **What happens if you don't.** Items laid edge to edge merge into one shape
> that no rectangle test accepts. The software notices, splits the mass into
> rough boxes, flags every one, and tells you they need adjusting — but on a
> tightly packed collage it may find four regions where there were eight, and
> you will spend longer dragging corners than you would have spent spreading
> the clippings out in the first place. A finger's width of table between each
> item is worth more than any amount of software.

## Taking the shot

1. Hold the phone **flat above the table**, lens parallel to the surface.
   Some tilt is corrected automatically, but less tilt means a better crop.
2. Get close enough that the items **fill most of the frame** — background
   around the edges is wasted resolution.
3. **Tap to focus** on one of the items before shooting.
4. Take the picture. Check it's sharp before moving on; blurry text can't be
   transcribed by anything.

## Items that need their own photo

Shoot these one at a time, filling the frame:

- Anything larger than about A4 / letter size
- Anything with small or faint print you'd struggle to read at arm's length
- Fragile items you don't want to move much
- Three-dimensional objects — badges, banners, trophies, gavels

The software handles a single item filling the frame automatically; you don't
need to tell it.

## Things that are fine

- **Curled or folded paper.** Flatten it as best you can under glass or a
  weight at the edges; the perspective correction handles the rest.
- **Faded or yellowed items.** Don't try to correct colour while shooting.
- **Handwriting on the back.** Shoot both sides as separate photos.
- **Items already in an album.** Shoot the page; you can add individual crops
  by hand in the review step.

---

## Then what

Copy the photos from your phone into the `inbox/` folder and run:

```
rotary run
```

That ingests, finds the items, crops and straightens them, and opens the review
page in your browser. Anything the software wasn't confident about is shown
first — everything else you can approve in a single click.

If a crop is wrong, drag its corners in the review page and it re-crops from
the original full-resolution photo. If an item was missed entirely, use
**Add missed item** on that photo and draw a box around it.

The original photos are never modified. If detection goes badly on a batch, you
can reshoot or re-run without losing anything.
