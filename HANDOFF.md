# Synpo Development Handoff

This document is the context-free continuation record for Synpo. Read it before
changing code, publishing a release, or proposing the next stage. It describes the
current approved behavior, scientific invariants, architecture, repository state,
validation procedure, and remaining work as of 2026-09-03.

## One-minute orientation

- Product: Windows desktop application for paired-channel, registered 3D microscopy
  TIFF stacks, written in Python 3.11/PySide6 and installed with Anaconda/Miniconda.
- Current version: `0.8.0 Beta`.
- Development checkout: `C:\Users\user\Documents\code2\Synpo`.
- Dedicated Conda environment: `synpo-microscopy`. Never modify the user's separate
  environment named `synpo`.
- Current development branch: `lsh`.
- Current development feature commit: `a2ad4dd` (`Improve batch preprocessing and review workflow`).
- Public user repository: <https://github.com/antamce/cluster-distribution>.
- Current public beta commit: `5523392` (`Improve preprocessing and review workflow`).
- Full test result for the current UX update: 41 tests passed.
- Current user status: the 0.8.0 beta now includes pair-specific preprocessing
  overrides, action-colored correction brushes, non-modal context-generation
  progress, high-visibility spine-map outlines, low-memory corrections, and
  reviewed-pair measurement gating with partial export. An actual approximately
  `80 x 2048 x 2048` stack on a 4 GB device remains the preferred field test.
- Development method: build in stages and do not move to a new stage until the user
  explicitly approves the previous one. All behavior through the current beta is
  approved.

## Repository and release safety

The local development repository and public user repository are deliberately not
the same publishing target.

- Active local branch: `lsh`; `master` remains at the 0.7.0 development release.
- Local `origin`: `https://github.com/antamce/cluster-distribution-calculator.git`,
  the earlier raw/development sharing repository. It is no longer the desired public
  user destination.
- The low-memory work and 0.8.0 metadata are pushed to the development remote as
  `origin/lsh`. Do not merge or push development history to the public repository.
- The presentable public repository is
  `https://github.com/antamce/cluster-distribution.git` on branch `main`.
- Beta 0.8.0 was published there from a clean temporary checkout as one curated
  public commit, not by changing the development checkout's `origin`.
- The public tree contains user-facing source, environment/launcher files, icon and
  README. It intentionally excludes `HANDOFF.md`, `tests/`, development history,
  microscopy data, manifests, and Zarr caches.
- `PUBLIC_README.md` is the source for the public repository's `README.md`.
- The temporary public-release checkout used for 0.8.0 was deleted after the remote
  commit was verified. Do not assume a second local checkout still exists.

Before any future public release:

1. Confirm the intended version and release scope with the user.
2. Commit development changes locally and run the complete test suite.
3. Update both `README.md` (developer checkout) and `PUBLIC_README.md` (user copy),
   plus this handoff.
4. Clone or fetch the public user repository into a temporary directory.
5. Copy only distributable files. Keep tests, this handoff, caches, data, and
   developer-only pytest configuration out of the public repository.
6. Commit the curated public delta and push its `main` branch.
7. Verify `refs/heads/main` at the expected commit before deleting the temporary
   checkout.

Do not rewrite the public repository's existing history unless the user explicitly
requests it. The original alpha is `f3bf194`, beta 0.7.0 is `03a9108`, and beta
0.8.0 is `54b77fa`.

## Running, setup, and testing

Create or update only the dedicated environment:

```powershell
conda env create -f environment.yml
# Or, if it already exists:
conda env update -n synpo-microscopy -f environment.yml --prune
```

Launch with:

```powershell
launch_synpo.bat
```

The launcher locates common Miniconda/Anaconda installations directly and invokes
`conda run`, so it works even when `conda` is not on `PATH`. The user considers a
reproducible environment and desktop shortcut sufficient packaging for now.
`scripts/create_desktop_shortcut.ps1` creates the Windows shortcut using the
supplied Synpo `.ico` asset.

Run the complete suite from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m pytest -q
```

Expected release baseline: `41 passed`. Also perform an offscreen application
construction smoke test after material UI changes. The application title should
contain `Synpo Microscopy Processor - Beta 0.8.0` (typographic dash may differ).

For importer-only diagnostics:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m synpo.cli scan "C:\path\to\batch"
```

Use `--skip-checksums` only for a quick structural scan.

## Product workflow

The user-approved workflow is:

1. Select a folder or manually select paired TIFFs.
2. Confirm channel roles, voxel calibration, and output location.
3. Review paired files and parsed experimental groups.
4. Tune preprocessing on representative specimens; save pair-specific ChanA/ChanB
   overrides for unusual pairs when needed.
5. Preprocess the batch unattended and run automatic detection.
6. Review every specimen; corrections are optional and become available while
   later pairs are still being detected.
7. Run measurements, review spine distributions/validity, and export tables/PDFs.
8. A later final-export stage will add mask/ROI packaging and verified cache cleanup.

Long batch operations remain responsive. Preprocessing, automatic detection, and
measurements display completed/total work, elapsed time, and a smoothed approximate
ETA. Automatic checkpoints are mandatory after preprocessing, detection, review,
and measurements for each pair.

## Input and pairing contract

### Original metadata-rich filenames

The original schema remains supported:

```text
<batch_prefix>_<experimental_group>_<specimen_id>_<ChanA|ChanB>_registered.tif
```

Example pair:

```text
exp 3107 - tomato_Glu 0 +_Untitled001_ChanA_registered.tif
exp 3107 - tomato_Glu 0 +_Untitled001_ChanB_registered.tif
```

The first underscore-delimited component is batch metadata, the next is the
experimental group, and the next is the specimen identifier. The channel marker
and optional registered flag follow them.

### Flexible and manual pairing

- Channel filename markers are editable; they are not restricted to `ChanA` and
  `ChanB`.
- Metadata-free files such as `Untitled001cy.tif` and `Untitled001cl.tif` are
  paired after stripping configured channel markers. Their whole batch is assigned
  to one editable fallback experimental group, and filenames become specimen names.
- Manual pairing accepts one TIFF for channel A and one for channel B, including
  files stored in different folders.
- Source paths are stored per file. Verification and recursive relinking support
  moved folders.
- The importer must surface duplicates, missing mates, mismatched dimensions/data
  types, ambiguous markers, and invalid axes before project creation.

### Image characteristics and channel roles

- The stacks are already XY-corrected/registered. Paired dimensions must match.
- Typical data are 16-bit, 2048 x 2048 pixels, and roughly 40-100 Z slices, but Z
  may fall outside that range. A batch is commonly about 100 pairs.
- Channel roles are confirmed for every batch. In the supplied/reference data,
  channel A contains protein clusters and channel B contains dendrites/spines.
- Pixel calibration is entered by the user and confirmed for each new batch because
  embedded TIFF metadata may be incorrect. Named presets are remembered.
- Reference calibration: XY `0.0462584 um/pixel`, Z `0.5 um`.

## Non-negotiable scientific and operational rules

- All intensity-based measurements use original 16-bit voxels. Preprocessed values
  may guide detection and hint-assisted boundaries but never replace raw intensity.
- Raw TIFFs are never modified or copied into the project cache.
- The app must target no more than 80% of physical RAM and avoid whole-batch loading.
- Preprocessing target: under one minute per channel.
- Standard detection target: under three minutes per pair. Low-memory detection may
  be substantially slower in exchange for bounded RAM use.
- Corrections are optional and a satisfactory automatic result can be marked review
  complete without drawing or confirming each object. That explicit specimen-level
  completion is required before measurement.
- Detection intentionally errs toward extra spine and cluster candidates; uncertain
  candidates are flagged and can be excluded later.
- A specimen is a field containing one or more dendrites. Multiple/crossing
  dendrites are all analyzed even if they may come from different cells.
- Axons and soma portions must be excludable through user hints. They are not to be
  inferred as belonging to a particular cell.
- Spine volume includes the neck up to the dendritic shaft. Filopodia are retained
  automatically and can be manually excluded. Touching spines should become
  separate objects.
- Each correction stroke is an instruction, not a final mask. Separate hints must
  create separate objects. Add/expand boundaries follow locally preprocessed image
  signal so rectangular hint shapes do not become rectangular objects.
- Dendrite/spine edits apply only to the current specimen. Protein clusters do not
  have equivalent manual drawing correction.
- A cluster is considered inside a spine when at least a configurable fraction of
  its retained volume overlaps that spine; default is 80%. Only the inside portion
  is measured for the current method.
- Statistical hypothesis testing is out of scope. Synpo exports clearly labeled raw
  and summary metrics; the user performs statistics elsewhere.
- Keep future design portable to macOS, although the current target is Windows.

## Architecture map

- `src/synpo/app.py`: PySide6 application, workflow tabs, background workers,
  progress/ETA plumbing, project state, viewers, review queues, dialogs and export
  orchestration. This is large; search for the relevant widget/class before editing.
- `src/synpo/models.py`: shared import/project model dataclasses.
- `src/synpo/calibration.py`: calibration presets and physical-unit handling.
- `src/synpo/importer.py`: configurable filename parsing, automatic/manual pairing,
  TIFF inspection, checksums, validation, and source metadata.
- `src/synpo/project.py`: schema-1 manifest creation/migration, atomic serialization,
  source verification, relinking, and checkpoint state.
- `src/synpo/preprocessing.py`: per-stack adaptive background/threshold estimation,
  physical-unit Gaussian filtering, slice-wise execution, RAM guards, Zarr cache,
  signatures, and resume.
- `src/synpo/detection.py`: dendrite shaft, separated spine candidate, and protein
  cluster candidate detection; automatic/forced low-memory execution, streamed
  projections, cross-slab component reconciliation, 3D labels, flags, signatures,
  and checkpoints.
- `src/synpo/review.py`: editable mask initialization, signal-guided hint handling,
  local 3D resegmentation, object statuses, undo patches, and checkpoints.
- `src/synpo/visualization.py`: slice-streamed XY/XZ/YZ projections, label overlays,
  physical aspect handling, cropped marching-cubes meshes, RAM/face limits, and the
  native Qt 3D renderer.
- `src/synpo/measurements.py`: association, cluster-end trimming, raw specimen/
  dendrite/spine/cluster metrics, checkpoint signatures, resume, and comparison
  illustrations.
- `src/synpo/distribution.py`: calibrated curved spine axes, endpoint hints, A* path
  completion, ten-bin voxel assignment, review state, aggregation and SEM.
- `src/synpo/exporting.py`: verified Excel/CSV exports and optional validation/audit
  PDF generation.
- `src/synpo/cli.py`: non-GUI scan/inspection command.
- `tests/`: 41-test release suite, including importer, project, preprocessing,
  detection, review, measurement, 3D rendering, and beta UI regression coverage.

Keep project manifests at schema version 1 unless a real schema break is needed.
Extend old manifests through `migrate_manifest`; do not silently make existing
projects unreadable.

## Cache and reproducibility

Runtime data live below the chosen output directory:

```text
<output>/.synpo-cache/<project-id>/
|-- preprocessed.zarr
|-- detection.zarr
|-- review.zarr
`-- measurements/
```

- The cache is compressed, resumable, and temporary.
- Project manifests contain source fingerprints, confirmed calibration, roles,
  settings, labels, paths and checkpoint state.
- Each computation has a signature derived from relevant inputs/settings.
- Automatic detection and corrected label volumes remain separate.
- Review data record the source detection signature. Changed detection invalidates
  stale corrections and returns the specimen to review.
- Distribution endpoint hints are retained in audit history. If resegmentation
  moves a hint off-mask, automatic endpoint selection resumes and that row becomes
  unreviewed.
- Cache deletion may only be offered after the complete final export has been
  verified. Current workbook/CSV/PDF export alone is not yet sufficient.

## Implemented stages and exact behavior

### Stage 1 - import and project setup (approved)

- Automatic metadata-rich pairing, configurable-marker fallback pairing, and manual
  A/B selection from the same or different folders.
- Editable import table, experimental group/specimen labels, channel roles, named
  calibration presets, output path, validations and checksums.
- Human-readable `*.synpo.json` project manifest, reopening, verification and
  recursive per-file relinking.

### Stage 2 - preprocessing (approved)

- Scrollable Z viewer, zoom, contrast adjustment and representative-specimen tuning.
- Independent channel settings: adaptive background percentile, physical XY/Z
  Gaussian smoothing and sensitivity/threshold controls with expanded beta ranges.
- Pair-level special-setting checkmarks activate separate ChanA/ChanB overrides;
  clearing a checkmark leaves its saved values dormant. Effective setting changes
  invalidate only affected preprocessing and downstream checkpoints.
- Background and threshold are estimated per stack.
- Compressed, slice-wise, RAM-guarded batch processing with resume and progress/ETA.
- Preprocessed data are used for detection/boundary guidance only.

Reference benchmark on the supplied full-resolution data: about 0.18 seconds for a
preview and 6.54 seconds per channel for batch preprocessing on the development
laptop. Treat these as historical guidance, not guaranteed performance.

### Stage 3 - automatic detection (approved)

- Detects dendrite shafts, separated spine candidates and cluster candidates in 3D.
- Original uint16 image remains the display background; standard overlays are green
  dendrites, cyan spines and magenta clusters.
- Detection settings expose expanded sensitivity ranges and candidate size/length
  controls.
- Each completed pair is checkpointed and available to review while the worker
  continues later pairs. Progress and ETA are visible.
- Automatic mode keeps the fast in-memory path when its conservative estimate fits
  below the 80% RAM ceiling and otherwise switches that specimen to disk-backed
  low-memory processing without a confirmation dialog.
- A per-project **Always use low-memory detection** option applies the slower path
  to every pending specimen without invalidating completed detection signatures.
- Low-memory detection streams the 95th-percentile projection, labels clusters in
  adaptive Z slabs, and reconciles 26-connected components across slab seams.
  Numeric object IDs may differ from the normal path; voxel masks and connected
  biological objects must remain equivalent.
- Temporary disk space is checked before each low-memory specimen. Insufficient
  space records a retryable `skipped` state and continues the batch. Other
  specimen-local exceptions record `failed` and also continue; project-wide errors
  still stop the batch. Interrupted specimens restart from the beginning.
- Reopening an unchanged detection checkpoint is fast (historically about 0.019 s).
- Historical reference detection time was about 22.88 seconds for one supplied pair.

### Shared projection and 3D viewing (Stages 3 and 4; approved)

- XY/XZ/YZ maximum projections preserve confirmed physical aspect, have linked
  crosshairs, support zoom, and can update the main Z location.
- The main canvas may use an individual Z slice or a drawable XY maximum projection.
- Before 3D generation, the user selects a rectangle on the XY maximum. Only that
  cropped region is sampled/rendered, preventing laptop crashes from full-field
  meshes.
- The Qt renderer builds closed interpolated triangular surfaces using marching
  cubes and adaptively caps complexity (currently 60,000 faces).
- Dendrites and spines have adjustable opacity; clusters remain opaque. All object
  classes have selectable publication colors and reset controls.
- Explicit X/Y/Z rotation sliders allow precise positioning, alongside drag rotate,
  wheel zoom, reset and PNG snapshot saving.
- Display-only Z spacing ranges from 0.20x to 5.00x; 1.00x preserves calibrated
  spacing. It never changes masks or measurements.
- Rotation uses bounded element-wise float32 arithmetic instead of NumPy matrix
  multiplication inside paint events. Do not reintroduce native BLAS calls there:
  they caused a fatal Windows MKL/DLL delay-load crash on the user's laptop.
- Projection/3D generation progress is a cancellable bottom status line, not a
  modal progress dialog. Full-field spine maps use thick green outlines for the
  current spine and thick cyan outlines for all others.

### Stage 4 - optional hint-assisted review (approved)

- Review queue fills as detection pairs complete. Review can run concurrently with
  later detection.
- Actions: add, exclude, exclude as filopodium, split, merge, expand, trim, accept,
  and flag as needing attention.
- Hint colors are add green, exclude red, trim magenta, expand blue, split yellow,
  filopodium purple, merge `#ED6291`, accept cyan, and needs-attention orange.
- Hints can be drawn on Z slices or XY maximum projections. Projection hints search
  touched labels through Z or infer a plane from strongest nearby processed signal.
- Each hint-assisted added object is independent. Boundaries are derived from the
  local preprocessed dendrite-channel signal, not the drawn rectangle/stroke shape.
- Undo exists both for the current drawing and the last applied correction. Applied
  actions atomically checkpoint masks and object statuses.
- Review can be completed without corrections. Any later correction returns it to
  in-progress. Comments are per specimen.
- Correction memory mode automatically moves oversized edits to disk-backed
  workspaces; a forced slow low-memory option is also saved per project. Undo is
  retained for low-memory exclude, filopodium, merge, split, trim, and expand.

### Stage 5 - association and measurements (approved)

- Only specimens with complete preprocessing, detection, and manual-review
  checkpoints are eligible. Resuming skips unfinished pairs, so spine review can
  begin once the first eligible measurement checkpoint exists.
- Uses compatible corrected dendrite/spine masks when present, otherwise immutable
  automatic masks.
- Assigns a cluster to the single spine with greatest voxel overlap, provided the
  retained cluster meets the configurable overlap threshold (80% default).
- Only qualifying cluster voxels inside the spine contribute to volume/ratios.
- Keeps one row per included individual cluster and an additional per-spine sum row.
- Cluster-end choices: untrimmed, fixed terminal-slice removal, or adaptive removal
  of consecutive oversized terminal slices from the inferred larger end. At least
  two slices remain by default.
- Comparison view shows fixed/adaptive candidate volume. Original protein intensity
  is displayed with counted voxels green and discarded voxels magenta.
- Resumable raw tables cover specimen, dendrite, spine, cluster and cluster sums.
- Core metrics include calibrated volumes, approximate dendrite length, spine
  density, average spine volume, inclusion percentage, cluster/spine ratio, average
  cluster volume, and distribution-derived values.

### Stage 6 - curved-axis distribution and exports (approved)

- One cropped 3D spine is skeletonized at a time using calibrated edge lengths
  `(z_step, xy_size, xy_size)`.
- Automatic base: largest spine/dendrite contact patch. Automatic distal endpoint:
  longest reachable medial-skeleton path.
- Optional **Centerline end hint** switches the crop from XY maximum to Z slices only
  while enabled. Its slider is bounded to the spine's occupied Z range and the crop
  remains centered with a 1 um margin.
- The automatic base is a read-only green dot; the distal endpoint is magenta.
  Clicks snap to a same-slice spine voxel within 12 screen/image pixels. Rejections
  use a status line, not a modal popup.
- A manual path follows the reachable skeleton then uses an inside-spine A* path to
  the exact hinted voxel. Place/replace/clear actions checkpoint immediately.
- The centerline is split into ten equal physical-length segments from shaft to tip.
  Every spine and qualifying inside-cluster voxel is assigned exactly once to the
  nearest centerline position.
- Saves ten spine-part volumes, ten cluster volumes and ten cluster/spine ratios per
  individual cluster-positive spine. Cluster-less spines do not get distribution
  rows.
- Empty spine-volume bins remain blank and are flagged as insufficient axis
  resolution; ambiguous/unusable axes are explicit, never silently fabricated.
- Distribution review automatically begins with the first pending spine and advances
  after decisions. Excluding a distribution affects only profile aggregation.
- **Invalid spine** excludes that spine from every downstream count/metric but
  retains masks and raw audit rows.
- There are separate cluster-positive and optional cluster-less spine review queues,
  stable spine IDs, spine number above each crop, click-through full-field context,
  and a zoomable full-field numbered/outlined spine map with filters.
- Group profiles first average included spines within each specimen, then average
  specimen means within experimental group. UI shows mean +/- SEM, line by default,
  with a bar switch and automatic/fixed 0-1 scale.
- Excel export contains all raw specimen/dendrite/spine/cluster measurements,
  individual distributions, specimen/group distributions, exclusion/invalid audits,
  compact group counts/means/variability/inclusion percentages, and settings.
- CSV versions of every sheet are written and verified.
- Partial export is permitted and includes only completed measurement checkpoints;
  the measurement panel reports omitted unfinished pairs.
- Optional multi-page PDFs use two original-channel XY maximum panels plus the
  ten-color mask/centerline and individual line/bar distribution. Main,
  distribution-excluded, and invalid-spine audit PDFs are independent options.

### Beta 0.8.0 additions (approved)

- Automatic and per-project forced low-memory detection.
- Disk-backed Z-slab protein-cluster labeling with cross-slab object merging.
- Conservative temporary disk preflight and automatic work-data cleanup.
- Retryable skipped/failed specimen states and specimen-level batch isolation.
- Memory strategy is operational metadata and does not invalidate completed masks.

### Beta 0.7.0 additions (approved)

- Zoom on preprocessing, detection, correction, orthogonal projection and spine
  context/map views.
- Expanded preprocessing and detection sensitivity ranges.
- Screen-aware persistent sizing; the app should not open larger than the display or
  leave full-screen/maximized mode unexpectedly when changing workflow stages.
- 3D X/Y/Z rotation controls.
- One signal-following object per correction hint.
- Flexible filenames, custom channel markers, manual cross-folder pairing and
  recursive relinking.
- Cluster-positive and optional cluster-less review with global invalid-spine
  exclusion.
- Batch progress bars and approximate ETA for preprocessing, detection and
  measurements.

## Measurement/export expectations

The user ultimately compares protein inclusion percentage, protein inclusion
volume, and distribution across experimental groups. Required table organization:

- Individual spine: volume, cluster presence, cluster/spine volume ratio, and ten
  shaft-to-tip distribution parts where applicable.
- Individual qualifying cluster: inside volume and distribution relative to spine.
- Per-spine cluster sum: summed cluster volume and related ratios/distribution.
- Dendrite: spine density per micrometer, mean spine volume, cluster-positive spine
  percentage, mean cluster/spine ratio, mean cluster volume and mean distribution.
- Specimen: clearly labeled aggregates of dendrite/spine metrics.
- Group: counts, means, variability, inclusion percentages and specimen-weighted
  distribution mean +/- SEM.

Keep every per-specimen, per-spine and per-cluster raw measurement in the Excel/CSV
exports before group averaging.

## Known limitations and pending work

There is no known active regression at this handoff. Remaining planned work is:

1. Validate curved axes and ten-bin profiles further on representative microscopy
   data when the user supplies/chooses examples.
2. Complete the final export package: segmentation masks, ImageJ-compatible ROI ZIP
   files, reproducible project/settings files, and organization of manually saved
   3D snapshots. Workbook, CSV and optional PDF exports already exist.
3. Verify that complete final package before offering cache deletion. Cache cleanup
   must remain optional and occur only after successful verification.
4. Optionally add an in-app annotation/training utility after the core application.
   It must be optional, part of Synpo rather than Fiji-dependent, and may use up to
   two annotated pairs the user can provide.
5. macOS packaging/compatibility is a future target, not current release scope.

Do not infer that the next task is necessarily item 1 or 2. Ask or follow the user's
next explicit stage request.

## Regression hazards and safe-change checklist

- `SliceView` must retain its detection rendering API, including
  `show_detection`; its accidental removal previously broke Z-slider detection
  refresh with `AttributeError`.
- Hint-assisted object masks must use preprocessed signal and keep separate object
  identities; square masks and merged multi-hint objects were prior regressions.
- Avoid unbounded full-field 3D meshes and BLAS-backed rotation in paint events.
- Preserve raw-intensity measurement provenance when refactoring preprocessing or
  visualization.
- Preserve screen maximized/full-screen state across tab/stage changes.
- Keep worker/UI communication thread-safe and checkpoint per pair before reporting
  completion.
- Maintain backward-compatible manifest migration and invalidate only genuinely
  stale derived results.
- Do not add cache/data files to Git. Check staged files before every commit.

For a material change:

1. Inspect `git status` and preserve unrelated user changes.
2. Identify which computation signature/checkpoint must be invalidated.
3. Add or update a focused regression test.
4. Run the full 41-test baseline plus any relevant GUI smoke test.
5. Update version/docs only when preparing a requested release.
6. Summarize behavior and validation for user approval before advancing stages.

## Historical decisions that should not be reopened silently

- Adaptive per-stack background and threshold estimation was explicitly chosen.
- Detection favors extra flagged candidates over missed objects.
- Cluster inclusion threshold defaults to 80%, and only inside voxels currently
  count.
- Cluster volume uses the sum when multiple qualifying clusters occupy one spine,
  while still retaining individual rows.
- Spine distributions are ten equal curved-centerline parts with voxel-by-voxel
  assignment.
- Cluster-less spines have no distribution row, but may optionally be reviewed for
  global invalid-spine exclusion.
- Group distribution summaries are specimen-weighted and use SEM error bars.
- Validation PDF uses two panels, is optional, and defaults to line profiles with a
  bar-chart alternative.
- Statistics remain external to Synpo.

## State immediately after this handoff update

The current 0.8.0 UX feature implementation is `a2ad4dd` on the local `lsh`
branch; the development remote was intentionally not changed by this public-only
publishing request. The curated public beta is verified at `5523392` on
`antamce/cluster-distribution` `main`. `HANDOFF.md` remains development-only and
is never copied into the public repository.
