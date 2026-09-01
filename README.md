# Synpo Microscopy Processor

Synpo is a staged Python/PySide6 desktop application for paired-channel 3D
microscopy data. It imports and validates batches, preprocesses registered stacks,
detects dendrites, spines, and protein clusters, and provides optional hint-driven
review. Raw TIFFs are never modified or copied.

## Current project status

The current version is `0.7.0 Beta`. The end-to-end workflow through flexible
import, preprocessing, detection, correction, measurement, spine review, and
tabular/PDF export is user-approved. See [HANDOFF.md](HANDOFF.md) for architecture,
scientific constraints, and development history.

This repository intentionally excludes microscopy TIFFs, project manifests, and
generated Zarr caches. They are user data or runtime artifacts, not source files.

## Setup and launch

The dedicated environment is `synpo-microscopy`; the separate environment named
`synpo` is not used or modified. From an Anaconda Prompt in this directory:

```powershell
conda env create -f environment.yml
launch_synpo.bat
```

For an existing dedicated environment:

```powershell
conda env update -n synpo-microscopy -f environment.yml --prune
launch_synpo.bat
```

The launcher finds Miniconda or Anaconda directly and starts the dedicated
environment through Conda's runner, so `conda` does not need to be on the ordinary
Windows command prompt's `PATH`. Run
`scripts/create_desktop_shortcut.ps1` once to create a desktop shortcut.

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
file matches its saved checksum.

## Stage 2: preprocessing

After saving or opening a project, open **2. Preprocessing**:

1. Choose representative specimens and inspect both channels.
2. Scroll through Z and tune display contrast. Contrast never changes data.
3. Tune adaptive background percentile, physical-unit XY/Z Gaussian smoothing,
   and threshold sensitivity independently for both channels.
4. Save channel settings, then select **Preprocess entire batch / resume**.

Background and threshold values are estimated independently for every stack.
Processed voxels are used only for detection; intensity measurements remain tied
to original 16-bit TIFF voxels. Processing is slice-at-a-time and guarded by the
project's 80% RAM ceiling.

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

On the supplied registered pair, defaults completed in 22.88 seconds, below the
three-minute target. Reopening an unchanged checkpoint took 0.019 seconds.

## Stage 4: interactive review

Open **4. Review and correction** when a specimen appears in the queue. Detection
of later specimens may continue simultaneously. Review is optional: a satisfactory
automatic result can be marked complete without drawing anything.

The viewer shows a scrollable original 16-bit Z slice with contrast controls and
colored overlays. Choose dendrite or spine, choose an action, then click or draw a
yellow hint. The drawing guides the operation and is not treated as a final mask.
The same orthogonal-maximum and rotatable-3D buttons are available here, using the
current corrected masks when corrections exist and automatic masks otherwise.

The **Main canvas** selector switches between an individual Z slice and a drawable
XY maximum projection. The Z slider remains available as a reference position in
maximum-projection mode. A projection hint searches the 3D labels for touched
objects and infers the relevant Z plane; a missed-object hint uses the strongest
nearby processed signal. Switch back to an individual slice whenever projected
objects overlap ambiguously.

Available actions are:

- add a missed object using nearby image signal;
- exclude an object or record a spine as an excluded filopodium;
- split touching objects or merge objects;
- expand or trim a boundary by re-evaluating a bounded 3D neighborhood;
- accept an object or flag it as needing attention without changing its mask.

**Undo drawn stroke** changes only the current hint. **Undo last applied
correction** restores saved 3D labels and prior object status. Every applied action
immediately writes a specimen checkpoint. Comments and review-complete state are
also saved; a later correction reopens the specimen as in progress.

Corrections are stored separately in `review.zarr`, leaving automatic masks intact.
Review groups are tied to their detection signature. If detection is rerun with
different settings, stale corrections are ignored and the specimen returns to the
review queue.

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
The Qt renderer performs its rotations with bounded element-wise array operations;
it does not call a native BLAS matrix routine from the paint event. This avoids a
Windows DLL delay-load failure observed on the development laptop.

## Stage 5: association and measurements

Open **5. Measurements** after automatic detection. Corrected dendrite/spine masks
are used when a compatible review checkpoint exists; otherwise the immutable
automatic masks are measured. The configurable minimum cluster/spine overlap is
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

The export button creates one verified Excel workbook plus CSV copies of every
sheet. Sheets include master specimen/dendrite/spine/cluster rows,
`Distribution_Individual`, `Distribution_Specimen`, `Distribution_Group`,
`Distribution_Excluded`, `Invalid_Spines`, compact group summaries, and settings.
Optional PDFs contain one two-panel XY maximum-projection page per spine using the
same fixed shaft-to-tip palette, its curved axis and ten-bin profile. The crop
margin is adjustable; excluded-distribution and invalid-spine audit PDFs are
separate options. Segmentation-mask/ImageJ ROI export remains part of the later
final-export stage.

## Beta workflow additions

Version 0.7.0 adds zoomable preprocessing, detection, correction, projection, and
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

The non-GUI importer can be exercised with:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m synpo.cli scan "C:\path\to\batch"
```

Add `--skip-checksums` only for a quick structural inspection.
