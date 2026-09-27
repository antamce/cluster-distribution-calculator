# Synpo Microscopy Processor

Synpo is a staged Python/PySide6 desktop application for paired-channel 3D
microscopy data. It imports and validates batches, preprocesses registered stacks,
detects dendrites, spines, and protein clusters, and provides optional hint-driven
review. Raw TIFFs are never modified or copied into the processing cache; they are
copied only when the user explicitly includes them in a transfer ZIP.

## Current project status

The current version is `0.8.0 Beta`. The end-to-end workflow through flexible
import, preprocessing, detection, correction, measurement, spine review, and
tabular/PDF export is user-approved. See [HANDOFF.md](HANDOFF.md) for architecture,
scientific constraints, and development history.

This repository intentionally excludes microscopy TIFFs, project manifests, and
generated Zarr caches. They are user data or runtime artifacts, not source files.

## Setup and launch

The dedicated environment is `synpo-microscopy`; the separate environment named
`synpo` is not used or modified. Both launchers discover environments in custom
Conda `envs_dirs`, so neither assumes that the environment is inside the Anaconda
or Miniconda installation folder.

### Windows

Double-click `launch_synpo.bat`. The launcher checks the active Conda installation,
its saved choice, `PATH`, Conda's environment registry, the Windows registry, and
common Anaconda, Miniconda, Miniforge, and Mambaforge locations. If discovery still
fails, select the installation folder in the graphical picker. If
`synpo-microscopy` is missing, the launcher offers to create it from
`environment.yml`; it never changes an environment merely because Synpo starts.

The successful choice is stored for this computer in
`%APPDATA%\Synpo\launcher-conda.txt`. Delete that file to choose a different Conda
installation. `conda` does not need to be on the ordinary Command Prompt's `PATH`.
Run `scripts/create_desktop_shortcut.ps1` once to create a desktop shortcut.

Manual setup and update commands remain available from an Anaconda Prompt:

```powershell
conda env create -f environment.yml
conda env update -n synpo-microscopy -f environment.yml --prune
launch_synpo.bat
```

### macOS

The current environment supports Intel and Apple Silicon Macs on macOS 12 or
newer. A separate legacy environment supports macOS 10.15 Catalina on Intel Macs
and macOS 11 on Intel or Apple Silicon. It pins Python 3.10 and PySide6 6.2.4 so
legacy support cannot downgrade packages for modern macOS or Windows users.

In Terminal, make the launcher executable once and start it:

```bash
chmod +x launch_synpo.command
./launch_synpo.command
```

After that, it can be opened from Finder. The launcher searches the active Conda,
its saved choice, `PATH`, initialized zsh/bash shells, and common installation
locations. It uses a macOS folder picker only if needed and offers to create the
correct modern or legacy environment for the detected OS. The choice is stored in
`~/Library/Application Support/Synpo/launcher-conda.txt`; delete that file to
choose again.

For Catalina, use the archived
[Miniforge installer supporting macOS 10.13-10.15](https://github.com/conda-forge/miniforge/releases/tag/26.1.1-3).
If Gatekeeper blocks the unsigned beta script, right-click
`launch_synpo.command`, choose **Open**, and confirm once. Do not disable
Gatekeeper globally. Catalina compatibility is dependency-solved but must still be
confirmed by running the test suite and a real dataset on the Catalina device.

## Filename pairing

The original metadata schema remains supported:

```text
<batch_prefix>_<experimental_group>_<specimen_id>_<ChanA|ChanB>_registered.tif
```

Channel markers are configurable, so metadata-free pairs such as
`Untitled001cy.tif` / `Untitled001cl.tif` can be placed in one editable fallback
group. A manual mode pairs individual A/B TIFFs from the same or different folders.
The importer checks pairing, duplicates, TIFF dimensions, data type, axis metadata,
and channel-shape agreement without loading whole stacks into RAM. SHA-256 source
fingerprints and per-file source paths support verification and recursive relinking.

## Stage 1: project setup

Projects are human-readable `*.synpo.json` manifests containing channel roles,
confirmed physical calibration, source fingerprints, TIFF metadata, checkpoint
state, and editable group/specimen labels.

Use **Verify sources** for full checksum verification. If the TIFF folder moves,
use **Relink source folder**; the source path changes only after every expected
file matches its saved checksum. Relinking first searches exact filenames and then
recovers renamed TIFFs by saved file size and SHA-256. Identical duplicates are
reported and the first deterministic match is used.

For another computer, use **Project → Create transfer ZIP…**. A **Full project
state** transfer includes the project and its preprocessing, detection, correction,
and measurement caches. A **Settings only** transfer preserves parameters,
calibration, labels, exclusions, ROIs, and specimen comments, but deliberately
clears derived results and correction history. Raw TIFFs are optional and are off
by default. Open either a normal project or a `*.synpo-transfer.zip` with **File →
Open project or transfer ZIP…**. Synpo validates the archive before activating it,
rewrites cache paths below the new output folder, and leaves the source ZIP
unchanged. If a full cache is damaged, the app offers an explicit settings-only
recovery instead of silently accepting partial results.

## Stage 2: preprocessing

After saving or opening a project, open **2. Preprocessing**:

1. Choose a specimen and inspect both channels. Each specimen has independent
   settings by default; channel entries are marked as parameters are confirmed,
   and a specimen turns green and says **reviewed** when both channels are marked.
2. Scroll through Z and tune display contrast. Contrast never changes data.
3. Tune adaptive background percentile, physical-unit XY/Z Gaussian smoothing,
   and threshold sensitivity independently for both channels.
4. Optionally select **Make all unset images in this channel use these parameters**.
   It fills only unmarked images and never overwrites reviewed values.
5. Optionally exclude an entire specimen pair with a reason, or define multiple
   rectangular XY analysis ROIs. ROIs apply to both channels and every Z slice;
   touching/overlapping rectangles merge and pixels outside them are not analyzed.
6. Save both channel settings, then select **Preprocess entire batch / resume**.
   Included specimens with an unmarked channel block the batch and are listed.

The Z slider maps clicks directly to the requested slice. Up/Down moves between
specimens when the specimen selector or image canvas has focus. **Preprocess
selected specimen** explicitly reruns preprocessing and then detection for that
pair; after confirmation, its existing corrections and correction history are
permanently discarded.

Background and threshold values are estimated independently for every stack.
Processed voxels are used only for detection; intensity measurements remain tied
to original 16-bit TIFF voxels. Processing is slice-at-a-time and guarded by the
project's 80% RAM ceiling. Changing a specimen setting or its analysis ROIs
invalidates that pair's downstream detection, review, and measurement checkpoints.
Excluded pairs are skipped downstream and retained with their exclusion reason for
audit.

The compressed cache is stored at
`.synpo-cache/<project-id>/preprocessed.zarr` below the output directory. Each
completed channel and written slice is checkpointed for safe resume. Cache deletion
is not offered before verified final export.

On the supplied 40 × 2048 × 2048 example, the defaults took 6.54 seconds for one
channel on the development machine; preview generation took 0.18 seconds.

## Stage 3: automatic detection

Open **3. Automatic detection** after preprocessing. Batch-wide settings include
separate dendrite/spine and cluster sensitivity, maximum terminal spine-branch
length, minimum dendrite length, and minimum candidate sizes.

The same controls can be saved as a specimen-specific override. **Redo detection
for selected specimen** applies it only to that pair. The review tab also links to
both explicit rerun paths. Rerunning detection permanently removes that specimen's
manual corrections and correction history after a warning.

Each completed pair is checkpointed immediately and can be viewed while later
pairs continue. The original 16-bit channel is the background:

- green: dendrite shaft candidates;
- cyan: separated spine candidates;
- magenta: protein-cluster candidates.

The Automatic Detection side panel also provides **Generate XY/XZ/YZ maximum
projections** and **Generate rotatable 3D object view**. Orthogonal projections
preserve the confirmed physical voxel aspect, have linked yellow crosshairs, and
can update the main Z slider. Before any 3D rendering, Synpo asks you to drag a
rectangle on the XY maximum projection and generates only that cropped region.
The 3D view constructs closed triangular surfaces from the cropped 3D masks. The
surface is interpolated between adjacent Z layers rather than displaying separate
voxel points; drag to rotate, scroll to zoom, and double-click to reset. Dendrites
and spines are translucent while protein clusters are opaque, making inclusions
and intersections visible.

Detection intentionally retains extra candidates. Small or short clusters,
possible filopodia, image-edge objects, and possible dendrite ends are flagged for
review. Compressed 3D label masks are stored in `detection.zarr`. A changed
detection setting produces a new reproducibility signature and replaces only stale
automatic masks.

Detection uses the fast in-memory algorithm when it fits below the project's 80%
RAM ceiling. Larger specimens automatically use a slower disk-backed mode that
streams the percentile projection, labels protein clusters in Z slabs, and merges
objects across slab boundaries. **Always use low-memory detection** can be saved as
a per-project execution setting without invalidating completed masks. Synpo checks
temporary disk requirements before each low-memory specimen; a specimen with
insufficient space is left retryable and skipped while the rest of the batch
continues. Other specimen-local failures are likewise recorded without stopping
unrelated specimens.

On the supplied registered pair, defaults completed in 22.88 seconds, below the
three-minute target. Reopening an unchanged checkpoint took 0.019 seconds.

## Stage 4: interactive review

Open **4. Review and correction** when a specimen appears in the queue. Detection
of later specimens may continue simultaneously. Review is optional: a satisfactory
automatic result can be marked complete without drawing anything.

The viewer shows a scrollable original 16-bit Z slice with contrast controls and
colored overlays. Choose dendrite or spine, choose an action, then click or draw a
colored hint. The drawing guides the operation and is not treated as a final mask.
Enable **Distinct colors for individual dendrites and spines** to expose touching
object borders while keeping protein clusters magenta. Dendrite and spine palettes
occupy disjoint color ranges, so the two object types never receive the same RGB
color.
The same orthogonal-maximum and rotatable-3D buttons are available here, using the
current corrected masks when corrections exist and automatic masks otherwise.

The **Main canvas** selector switches between an individual Z slice and a drawable
XY maximum projection. The Z slider remains available as a reference position in
maximum-projection mode. A projection hint searches the 3D labels for touched
objects and infers the relevant Z plane; a missed-object hint uses the strongest
nearby processed signal. Switch back to an individual slice whenever projected
objects overlap ambiguously.

Brush colors and actions are:

- **Add missed object — green:** draw a separate stroke inside each missing
  object. Image-supported new regions that touch become one object. A region that
  touches exactly one existing object of the same category joins that stable ID;
  contact with multiple existing objects is rejected as ambiguous.
- **Assign painted dendrite area to spine — lime:** available only on the drawable
  XY maximum projection. The painted area must touch exactly one spine. Every
  dendrite-mask voxel beneath the painted XY footprint across all Z slices is moved
  literally to that spine; background and other masks are unchanged.
- **Assign painted spine area to dendrite — coral:** the projection-only reverse
  transfer. The painted area must touch exactly one dendrite; touching none or
  multiple dendrites changes nothing. Covered voxels from every touched spine ID
  are moved to that dendrite through all Z slices. Surviving parts retain their
  spine IDs, while fully consumed spines are removed from the object count.
- **Exclude object — red:** touch an unwanted object to remove that complete 3D
  object from the corrected mask.
- **Trim boundary — magenta:** draw across excess segmentation. Synpo removes the
  hinted region and locally retains the largest connected, image-supported
  remainder of that object.
- **Expand boundary — blue:** draw from an existing object toward omitted signal.
  Synpo regrows that object through locally supported preprocessed signal.
- **Split touching objects — yellow:** draw through the neck or contact that should
  divide one object. Exactly two stable IDs are produced: the two largest pieces
  seed the sides and every smaller fragment is assigned to its physically nearest
  side. A stroke without a meaningful second piece is rejected.
- **Eraser:** remove only the painted dendrite and spine voxels, regardless of
  object type. On a Z slice it affects only that slice; on the XY projection it
  clears the painted columns through all Z slices. Surviving pieces retain IDs.
- **Exclude as filopodium — purple:** touch a spine to exclude it while recording
  the specific filopodium decision in the audit trail.
- **Merge objects — `#ED6291` pink:** draw through at least two objects that should
  be one; they are combined under one stable ID.
- **Accept object — cyan:** retain the mask unchanged and record it as accepted.
- **Flag object for attention — orange:** retain the mask unchanged while keeping
  an explicit needs-attention status.

The correction sensitivity slider is enabled for Add, Expand, and Trim and disabled
for other actions. Each Apply captures the currently displayed value, so consecutive
corrections may use different sensitivities. Higher sensitivity grows more signal
for Add/Expand and removes more weak signal for Trim. Brush size is specified as a
diameter from 1 to 1000 pixels.

Brush shortcuts are Ctrl+1 Add, Ctrl+2 Exclude, Ctrl+3 Split, Ctrl+4 Merge,
Ctrl+5 spine-to-dendrite, Ctrl+6 dendrite-to-spine, Ctrl+7 Trim, Ctrl+8 Expand,
Ctrl+9 Eraser, and Ctrl+0 Filopodium. Enter applies the current correction unless
a text field has focus.

**Undo drawn stroke** changes only the current hint. **Undo last applied
correction** restores saved 3D labels and prior object status. Every applied action
immediately writes a specimen checkpoint. Comments and review-complete state are
also saved; a later correction reopens the specimen as in progress.

A persistent **Save project** button remains visible below every workflow tab. It
saves committed project state without applying uncommitted processing or correction
controls.

Corrections are stored separately in `review.zarr`, leaving automatic masks intact.
Review groups are tied to their detection signature. If detection is rerun with
different settings, stale corrections are ignored and the specimen returns to the
review queue.

Correction memory strategy defaults to **Automatic**, which uses the ordinary
in-memory operation when safe and switches oversized object edits to a slower,
disk-backed path. **Always use slow low-memory correction** is available for
low-RAM computers. Exclude, filopodium, merge, split, trim, and expand preserve
undo checkpoints in both modes; bounded add operations remain local and report the
chosen mode.

### Advanced performance diagnostics

Use **Advanced → Diagnostic level** and **Advanced → Diagnostic mode** to capture
a privacy-conscious performance session. The default level is **Full-session
diagnostics**; correction-only capture is also available. Diagnostic mode starts
disabled on every launch. With a project open, sessions are written below
`<output>/Synpo diagnostics/<timestamp>/` as append-only event timings, two-second
CPU/memory/disk samples, a sanitized environment inventory, and a final JSON
summary. Microscopy pixels, project comments, process command lines, and window
titles are not collected, and home-directory prefixes are replaced with
`%USERPROFILE%`.

Choose **Finish and package diagnostic session** to finalize the folder and create
a ZIP beside it. A partially written JSONL/CSV session remains useful after an
unexpected interruption. **Open diagnostic protocol PDF** opens the single bundled
testing and interpretation guide; it is not duplicated into each session folder.
Do not disable workplace security software—use the evidence to request an approved,
narrowly scoped IT test if endpoint scanning appears to dominate cache operations.

Projection/3D generation streams one Z slice at a time, enforces the 80% RAM
ceiling, and adaptively limits the interactive rendering to 60,000 surface faces
inside the selected rectangle. The projection window can independently hide
dendrites, spines, or clusters. The 3D tab has independent color pickers for all
three object classes plus a reset button, and the chosen publication colors are
used when saving the current view as a PNG snapshot. **Z-layer spacing** adjusts
the displayed separation from `0.20×` to `5.00×`; `1.00×` is the confirmed
physical calibration. This is display-only and never changes masks or measurements.
Independent dendrite and spine opacity controls range from 5% to 100%; dendrites
default to 75%, spines to 60%, and protein clusters remain fixed at 100% opacity.
Generation progress and cancellation are shown in the status line at the bottom of
the main window; generation does not open a modal progress popup. Numbered
full-field spine maps draw the current spine with a five-pixel-or-wider bright
green outline and every other visible spine with a similarly thick cyan outline.
The Qt renderer performs its rotations with bounded element-wise array operations;
it does not call a native BLAS matrix routine from the paint event. This avoids a
Windows DLL delay-load failure observed on the development laptop.

## Stage 5: association and measurements

Open **5. Measurements** after at least one specimen has completed preprocessing,
automatic detection, and manual review. Clicking **Mark review complete** qualifies
a specimen even when no corrections were needed. The resumable measurement batch
processes only fully reviewed specimens and skips unfinished pairs until a later
resume. Corrected dendrite/spine masks are used when present; otherwise the
immutable automatic masks are measured. The configurable minimum cluster/spine overlap is
80% by default. Each cluster is assigned to the spine with its greatest overlap,
and only the overlapping portion contributes to cluster volume.

Cluster-end volume can be left untrimmed, trimmed by a fixed number of slices from
the larger terminal end, or trimmed adaptively when consecutive terminal slice
areas exceed the stable-area threshold. The comparison table reports candidate
volumes from fixed and adaptive approaches side by side. For the saved method, the
cluster illustration overlays counted voxels in green and discarded terminal
voxels in magenta on the original 16-bit protein-channel maximum projection.

The resumable batch produces compressed per-specimen raw rows for specimens,
dendrites, spines, individual included clusters, and per-spine cluster sums.
Current metrics include calibrated volumes, approximate skeleton length, spine
density, inclusion percentage, and summed cluster/spine volume ratios.

## Stage 6: curved-axis protein distribution and export

For every cluster-positive spine, Synpo builds a calibrated 3D curved centerline
from the largest shaft-contact region to the longest distal skeleton endpoint. It
divides that path into ten equal physical-length parts and assigns every spine and
qualifying inside-cluster voxel to its nearest centerline position. The individual
distribution table saves, for each part, spine volume, cluster volume, and their
unitless 0–1 ratio. Empty physical parts remain blank and are excluded per-bin;
ambiguous axes and unusable paths are explicitly flagged.

A spine with exactly two meaningful disconnected pieces may use a virtual,
signal-guided bridge up to the configurable **Maximum centerline gap** (1.0 µm by
default). The bridge changes only the analysis centerline—never the spine mask or
volume—and is drawn in the preview and saved in export metadata. Bridged spines
are excluded from distribution summaries until explicitly reviewed. Tiny detached
satellites are ignored only for centerline topology while remaining in measured
mask volume. If a bridge is too long, or more than two substantial components are
present, Synpo retains a usable centerline in the component nearest the dendrite
and flags it for review. A manual endpoint can select which distal component to
bridge before the path is recalculated. Projects containing results from the older
component rule are automatically marked for measurement recalculation when opened;
their saved distribution-review decisions are preserved.

The distribution review opens at the first unreviewed spine. **Include in
distribution summaries** affects only profile aggregation, while **Invalid spine**
excludes the spine from all downstream counts and metrics without deleting its
mask or audit row. Each decision is checkpointed immediately, the current specimen
is recalculated, and the review advances. Group profiles first average eligible
spines within each specimen, then average specimen means by experimental group;
the screen shows SEM across specimen means. Line profiles are the default, with a
bar-chart switch and automatic or fixed 0–1 scaling.

The cropped spine viewer normally remains an XY maximum projection and requires no
manual correction. If the automatic distal endpoint is wrong, press **Centerline
end hint**. The same panel switches to an exact Z-slice viewer with a slider limited
to the first and last slices occupied by that spine; the XY crop includes the full
spine and a 1 µm margin. Click near the spine to place the endpoint. Synpo snaps the
dot to the nearest selected-spine voxel on that slice, rebuilds the path from the
read-only automatic base, checkpoints it, and returns to the maximum projection.
The base is green, the endpoint is magenta, and invalid clicks appear only in the
status line. **Clear end hint** restores automatic endpoint detection.

Placed, replaced, cleared, and resegmentation-invalidated hints remain in audit
history. A hint invalidated by later mask correction falls back to the automatic
endpoint and returns the spine to the unreviewed queue. Excel/CSV rows save both
endpoint coordinates, source and validity; validation PDFs show both endpoint dots.

The export button permits a partial export as soon as one measurement checkpoint is
complete. Its status states how many completed pairs are included and how many
unfinished pairs are omitted. It creates one verified Excel workbook plus CSV
copies of every sheet. Sheets include master specimen/dendrite/spine/cluster rows,
`Distribution_Individual`, `Distribution_Specimen`, `Distribution_Group`,
`Distribution_Excluded`, `Invalid_Spines`, `ROI_Master`, `Excluded_Specimens`,
compact group summaries, and settings. Object rows carry an ROI ID; ROI rows save
the original-image rectangle coordinates and per-ROI metrics, while the specimen
row contains pooled metrics across its ROIs.
Optional PDFs contain one two-panel XY maximum-projection page per spine using the
same fixed shaft-to-tip palette, its curved axis and ten-bin profile. The crop
margin is adjustable; excluded-distribution and invalid-spine audit PDFs are
separate options. Segmentation-mask/ImageJ ROI export remains part of the later
final-export stage.

## Beta workflow additions

Version 0.8.0 adds automatic low-memory detection for stacks that exceed the RAM
safety ceiling, a per-project option to use that strategy for every specimen,
cross-slab object reconciliation, disk-space preflight, retryable skipped/failed
states, and specimen-level batch failure isolation. The current 0.8.0 source also
adds Windows and macOS Conda launchers that discover nonstandard installation and
environment directories, plus an isolated Python 3.10/PySide6 6.2.4 environment
for macOS 10.15 Catalina and macOS 11.

Version 0.7.0 added zoomable preprocessing, detection, correction, projection, and
spine-context views; larger sensitivity ranges; independent X/Y/Z 3D rotation
controls; and screen-aware persistent window sizing. Missed-object correction
strokes now produce separate objects and use preprocessed signal for image-guided
boundaries.

The measurement screen has separate protein-cluster-positive and optional
cluster-less spine queues. Both use stable spine IDs. Clicking a crop opens the
full specimen with the selected spine highlighted, while the numbered spine map
provides maximum-projection or single-Z viewing and category/validity filters.
Invalid cluster-less spines are excluded from every downstream metric and retained
in audit exports.

Preprocessing, automatic detection, and measurement each show a responsive batch
progress bar with work counts, elapsed time, and a smoothed approximate ETA.

## Tests and command-line validation

Run the test suite from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m pytest -q
```

Current Windows baseline: `46 passed, 1 skipped`; the skipped test executes the
macOS launcher with `/bin/bash`. PowerShell and Bash syntax checks, Windows Conda
runtime resolution, Python 3.10 syntax parsing, and offscreen application startup
also pass. Conda dry-run resolution succeeds for Catalina Intel, macOS 11 Apple
Silicon, and macOS 12 on Intel and Apple Silicon. These solves do not replace
running Synpo and representative microscopy data on real Macs.

The non-GUI importer can be exercised with:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m synpo.cli scan "C:\path\to\batch"
```

Add `--skip-checksums` only for a quick structural inspection.
