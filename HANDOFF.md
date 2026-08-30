# Synpo Development Handoff

## Current state

Synpo is a Python/PySide6 desktop application for paired-channel, registered 3D
microscopy TIFF stacks. Version `0.6.0` implements:

- Stage 1: batch import, filename parsing, pairing, validation, calibration,
  project manifests, fingerprints, reopening, verification, and relinking;
- Stage 2: adaptive preprocessing previews and resumable compressed batch caching;
- Stage 3: automatic dendrite-shaft, spine-candidate, and protein-cluster-candidate
  detection with a slice viewer and overlays (user-approved);
- Stage 4: optional specimen review, drawn hints, local 3D resegmentation,
  add/exclude/split/merge/boundary actions, object flags, undo, and automatic
  per-specimen checkpoints, plus shared orthogonal maxima and rotatable 3D context
  in both Detection and Review (user-approved);
- Stage 5: resumable spine/cluster association and raw specimen-, dendrite-, spine-,
  and cluster-level measurement tables, with cluster-end method comparison and a
  counted/discarded-voxel illustration (user-approved);
- Stage 6: calibrated curved-centerline protein distribution, ten voxel-assigned
  shaft-to-tip parts, distribution/invalid-spine review, specimen-weighted group
  profiles with SEM, Excel/CSV exports, and optional validation/audit PDFs
  (implemented and awaiting user testing).

All settings and actions except the Z slider are in scrollable side panels. Do not
skip the staged approval process: let the user test Stage 6 and wait for approval
before proceeding to the remaining final-export work.

## Running and testing

The dedicated environment is `synpo-microscopy`; do not modify the user's separate
`synpo` environment.

```powershell
conda env update -n synpo-microscopy -f environment.yml --prune
launch_synpo.bat
```

The launcher locates Miniconda or Anaconda by absolute path and uses `conda run`
for the dedicated environment, so it works when `conda` is absent from `PATH` and
sets up native DLL lookup consistently.

Run tests from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m unittest discover -s tests -v
```

Current result: 15 tests pass. An offscreen GUI construction smoke test also passes
with five workflow tabs. The supplied full-resolution pair previously benchmarked
at 0.18 seconds for preprocessing preview, 6.54 seconds per preprocessed channel,
and 22.88 seconds for automatic detection. An unchanged detection checkpoint opens
in about 0.019 seconds.

## Input contract

- All TIFFs are directly inside one selected folder.
- Pair names differ only in `ChanA` versus `ChanB`.
- Schema:
  `<batch_prefix>_<experimental_group>_<specimen_id>_<ChanA|ChanB>_registered.tif`.
- One folder uses one filename schema and one channel-role assignment.
- Supplied data use ChanA for protein clusters and ChanB for dendrites/spines.
- Typical stacks are uint16, 2048 × 2048, and 40–100 Z slices, with about 100 pairs
  per batch.
- Stacks are already XY-registered and paired dimensions must be identical.
- Calibration is entered and confirmed per batch because embedded metadata may be
  wrong. The example is XY `0.0462584 µm/pixel`, Z `0.5 µm`.

## Scientific and workflow requirements

- Intensity measurements always use original uint16 voxels, never processed data.
- The app must use no more than 80% of device RAM.
- Preprocessing target: under one minute per channel.
- Detection target: under three minutes per pair.
- Long operations keep the UI responsive and show progress or ETA.
- Preprocessing, detection, review, and measurements have automatic per-pair checkpoints.
- Detection intentionally keeps extra candidates and flags uncertain objects.
- Fields can contain multiple/crossing dendrites. Cell identity is irrelevant.
  Axons and soma portions must be excludable by hints.
- Spine volume includes the neck up to the shaft. Filopodia are not automatically
  discarded. Closely touching spines should be separate objects.
- Edits apply only to the current specimen and drawings are hints, not final masks.
- Protein clusters have no drawing correction tools.
- Review becomes available as completed pairs arrive while detection continues.

## Architecture and storage

- `src/synpo/app.py`: PySide6 application, independent background workers, five
  workflow tabs, preview/detection viewers, and drawable review canvas.
- `src/synpo/importer.py`: filename parsing, TIFF inspection, pairing, fingerprints.
- `src/synpo/project.py`: schema-1 manifest migration, serialized atomic saves,
  source verification, and relinking.
- `src/synpo/preprocessing.py`: adaptive background/Otsu estimation, physical-unit
  Gaussian filtering, slice processing, RAM guard, Zarr cache, and resume.
- `src/synpo/detection.py`: projection/skeleton shaft and spine separation, 3D
  labels, cluster components, flags, signatures, and checkpoints.
- `src/synpo/review.py`: editable mask initialization, hint interpretation, local
  correction, object statuses, compressed undo patches, and checkpoints.
- `src/synpo/visualization.py`: slice-streamed XY/XZ/YZ maximum projections,
  physical calibration, label projections, bounded interpolated 3D meshes, and
  visualization RAM checks.
- `src/synpo/measurements.py`: slice-streamed volume counts, cluster-end trimming,
  80%-default spine association, dendrite/spine assignment, calibrated raw metrics,
  compressed per-specimen results, resume signatures, and illustration loading.
- `tests/`: importer, project, preprocessing, detection, review, measurement,
  checkpoint, migration, and resume coverage.

Keep project manifests at schema version 1 and extend them through
`migrate_manifest` for backward compatibility.

```text
<output>/.synpo-cache/<project-id>/
├── preprocessed.zarr
├── detection.zarr
├── review.zarr
└── measurements/
```

Automatic and corrected label volumes are separate. Review groups carry the source
detection signature; stale corrections are ignored and review state resets after a
changed detection result. Raw TIFFs are never changed or copied.

## Stage 4 behavior

- The queue gains each detected specimen immediately; its review worker can run
  concurrently with detection of other specimens.
- Dendrite/spine hints support add, exclude, exclude-as-filopodium, split, merge,
  expand, trim, accept, and flag-needs-attention.
- The main Review canvas can switch between a Z slice and a drawable XY maximum.
  Projection hints search object labels through Z and infer a local Z plane from
  label overlap or strongest nearby processed signal. The slider remains a
  reference Z position in projection mode.
- Add and expand use processed dendrite-channel signal in a bounded 3D neighborhood.
  Other actions interpret hints against existing 3D object labels.
- Every applied action stores an undo record and atomically saves the checkpoint.
  Undo restores both label data and previous object status.
- Review may be completed with no edits. Any subsequent correction changes it back
  to in-progress. Comments are per specimen.
- Corrected display still uses original 16-bit TIFF intensity as its background.
- Detection and Review each expose buttons for XY/XZ/YZ maxima and the rotatable
  3D object viewer. Projection crosshairs are linked and update the main Z slice.
- Orthogonal displays respect physical XY/Z calibration. The native Qt 3D renderer
  uses marching cubes to build closed, interpolated triangular surfaces between Z
  layers, then draws depth-sorted shaded faces without OpenGL. It supports
  rotate/zoom/reset, adaptively limits mesh complexity, and can save PNG snapshots.
  Dendrites and spines have adjustable 5%–100% surface opacity, defaulting to 75%
  and 60%; clusters remain fixed at 100% so inclusions remain visible. Separate
  class color pickers and reset controls let users choose publication appearance
  before saving a snapshot. A `0.20×`–`5.00×` Z-layer display multiplier
  changes only the 3D view and snapshot; `1.00×` preserves calibrated spacing and
  no multiplier affects masks, calibration, or measurements.
- The 3D button no longer attempts a full-field render. It first opens an XY maximum
  area-selection dialog; the user drags a rectangle and only that cropped XY region
  is sampled through Z and rendered. This is the laptop-safe path.
- A Windows fatal exception in the first 3D paint was traced to NumPy matrix
  multiplication entering a native BLAS/MKL delay-load path. Renderer rotation now
  uses element-wise float32 coordinate arithmetic instead. The exact visible
  detection-worker, area-selection, and first-paint workflow passed after this fix.

## Stage 5 behavior

- A cluster is assigned to the single spine with greatest voxel overlap and is
  included only when retained-cluster overlap meets the configurable threshold
  (80% default). Only voxels inside that spine contribute to its measured volume.
- Cluster rows preserve every included individual cluster and add a per-spine sum
  row. Stage 6 populates the corresponding ten-part spine-distribution records.
- End handling offers untrimmed, fixed terminal-slice removal, and adaptive removal
  of consecutive oversized terminal slices. The larger terminal end is inferred
  from endpoint areas; at least two slices are retained by default.
- The comparison table shows fixed versus adaptive candidate volume for every
  cluster. The illustration uses original uint16 protein intensity, with counted
  voxels green and discarded voxels magenta.
- Results are gzip-compressed under the project cache and checkpointed after every
  specimen. Signatures include masks, settings, and calibration.

## Stage 6 behavior

- `distribution.py` skeletonizes one cropped 3D spine at a time. Calibrated graph
  edge lengths use `(z_step, xy_size, xy_size)`. The origin is the largest
  spine/dendrite contact patch and the distal endpoint is the longest reachable
  path. Similar contacts/endpoints are flagged `ambiguous_axis`; no path is saved
  as `no_usable_path`.
- Ten equal centerline-length bins receive complete spine and qualifying
  inside-cluster voxels by nearest centerline position. Every voxel is assigned
  once. Zero-spine-voxel bins are blank and flagged
  `insufficient_axis_resolution`; all other bin data are preserved.
- Only cluster-positive spines receive distribution rows. Numerators use inside
  portions of clusters passing the overlap threshold after selected end trimming.
  Rows contain ten ratios, ten spine-part volumes, ten cluster volumes, status,
  review state, and note.
- Valid automatic axes are included by default, so correction is optional.
  Ambiguous axes are excluded pending review. Distribution exclusion leaves other
  metrics intact. Invalid-spine review retains raw rows/masks but immediately
  removes that spine from all specimen/dendrite/group metrics and checkpoints the
  decision before auto-advance.
- Aggregation first averages included cluster-positive spines within each specimen,
  then averages specimen means within experimental groups. Group SD, SEM, and
  separate per-bin `n` are saved; the app displays mean ± SEM with line/bar and
  automatic/fixed-scale controls.
- `exporting.py` writes and verifies an Excel workbook and CSV copy of every sheet.
  Optional multi-page validation PDFs use two original-channel XY maximum panels,
  the ten-color overlay, curved axis, individual profile, identifiers, and status.
  Main, distribution-excluded audit, and invalid-spine audit PDFs are independent;
  crop margin defaults to 1 µm and is adjustable.

## Pending stages and decisions

After Stage 6 approval, remaining work includes:

1. Validate curved axes and ten-bin profiles on representative microscopy data.
2. Complete final export of segmentation masks, ImageJ-compatible ROI ZIPs,
   reproducible project/settings files, and manually saved 3D snapshots. The
   workbook and raw CSV export are already implemented.
3. Keep cache compressed and temporary; offer deletion only after the complete
   final export has been verified. Stage 6 tabular/PDF export alone does not make
   cache deletion eligible.
4. Optionally add an in-app annotation/training utility after the main app. It must
   not depend on Fiji; the user can provide up to two annotated pairs.
5. Statistics remain out of scope. Synpo calculates and exports labeled metrics;
   statistical analysis occurs elsewhere.

The current target is Windows with Anaconda and a desktop shortcut. Keep the design
portable for a later macOS version.
