# Synpo Development Handoff

This document is the context-free continuation record for Synpo. Read it before
changing code, publishing a release, or proposing the next stage. It describes the
current approved behavior, scientific invariants, architecture, repository state,
validation procedure, and remaining work as of 2026-10-09.

## One-minute orientation

- Product: Windows and macOS desktop application for paired-channel, registered 3D
  microscopy TIFF stacks, written in Python/PySide6 and installed with Conda.
  Current systems use Python 3.12/PySide6 6.7+; the isolated macOS 10.15/11 legacy
  environment uses Python 3.10/PySide6 6.2.4.
- Current version: `0.8.0 Beta`.
- Development checkout: `C:\Users\user\Documents\code2\Synpo`.
- Dedicated Conda environment: `synpo-microscopy`. Never modify the user's separate
  environment named `synpo`.
- Current development branch: `lsh`.
- Current development feature commit: `38ad078`, titled `Add feature correlation
  and bulk distribution review`.
- Public user repository: <https://github.com/antamce/cluster-distribution>.
- Current public beta commit: `4b872d815411a7231e6305e0f9f37714864019c6`
  (`Add feature correlation and bulk distribution review`).
- Current full Windows result: 105 passed and 1 platform-specific test skipped.
  PowerShell and Bash syntax checks, Windows runtime resolution, Python 3.10 syntax
  parsing, and the offscreen application construction smoke test also passed.
  Conda dry-run resolution passed for Catalina `osx-64`, macOS 11 `osx-arm64`,
  and macOS 12 on both `osx-64` and `osx-arm64`.
- Current user status: the feature-correlation and project-wide bulk distribution-
  review update was published to public `main` as `4b872d8` on 2026-10-09. An actual approximately
  `80 x 2048 x 2048` stack on a 4 GB device and a Catalina launch remain preferred
  field tests.
- This release adds an Advanced-menu diagnostic mode for an
  intermittent work-PC correction slowdown. Full-session capture is the default
  level, diagnostics remain opt-in per launch, and a standalone bundled PDF gives
  the controlled cross-machine protocol.
- It also contains the user's post-testing
  workflow changes: per-specimen preprocessing/detection reruns, marked/reviewed
  preprocessing state, specimen exclusion and multi-ROI analysis, a literal
  eraser, exactly-two-way splitting, fixed review hotkeys and disjoint palettes,
  and review-required virtual bridging for two-part spines. It also adds validated
  full-state/settings-only transfer ZIPs, optional embedded raw TIFFs, transactional
  import, and checksum recovery for renamed external TIFFs. The user explicitly
  requested publication of this complete scope to the curated public repository,
  and the verified publication is complete.
- Development method: build in stages and do not move to a new stage until the user
  explicitly approves the previous one. All behavior through the current beta is
  approved.
- The current public release adds all-spine calibrated length and morphology
  metrics, editable base/tip and head/neck geometry (including maximum-projection
  brushes and undo/redo), and a sixth interactive morphology-clustering panel.
  Named runs support GMM, Ward, and k-means with 2D/3D PCA, editable colors, and
  separate Excel/CSV/PDF/SVG/600-DPI PNG analysis packages. Protein puncta are an
  explicit descriptive-only layer: they never enter PCA or clustering. Standalone
  re-clustering from exported morphology workbooks is supported.
- The morphology follow-up removes per-voxel Zarr reads from head/neck painting,
  throttles stroke redraws, and moves geometry recalculation to a worker thread.
  Checkpoint-and-advance preserves its queue position, morphology-stage invalidity
  is an auditable all-metric exclusion, named runs can select any subset of
  experimental groups, and group marker shapes appear alongside cluster colors.
  Per-run axes/background colors and alpha values affect both interactive and
  exported plots. The latest plot follow-up adds saved legend visibility/placement,
  an arbitrary X/Y view for any selected clustering features, and repairs the
  Matplotlib protein-puncta-volume box plot while showing its source observations.
  A dedicated PCA-interpretation dashboard adds score/loading vectors, a loading
  heatmap, explained-variance bars, explicit equations, reproducibility metadata,
  and a matching high-resolution exported figure. Three-component runs render the
  score/loading-vector panel as an interactive 3D PC1/PC2/PC3 view and also expose
  a standalone full-size 3D feature-axis biplot. Protein-puncta-volume observations
  use group-specific marker shapes; their legend explains cluster-colored points
  and hollow black box-plot outliers.
  Both PCA feature-vector views persist controls for showing/hiding spine points
  and for their opacity; feature vectors stay fully opaque and exports honor the
  chosen point display.
- Named morphology runs can optionally include only spines explicitly marked
  `geometry_reviewed`; the default remains all otherwise valid complete spines.
  Unreviewed exclusions are retained in the run audit and the toggle is serialized
  in `MorphologyClusteringSettings`.
- Automatic morphology cluster-count selection now offers the existing information
  criterion, maximum silhouette, or a normalized within-cluster-SSE elbow. The
  chosen method is saved for PCA, UMAP, and PCUMAP runs; fixed counts override it.
  Candidate exports include raw SSE, silhouette, elbow distance, acceptance, and
  the selected candidate.
- Tabs 6 and 7 have a `Cluster-count scores` view that reuses the saved candidate
  diagnostics (no refit), annotates exact values, highlights accepted/rejected and
  selected candidates, and tabulates information criterion, silhouette, or the
  within-cluster-SSE elbow curve.
- Tabs 6 and 7 also have a saved `Feature correlations` view. It computes a
  listwise-complete spine-by-feature Pearson matrix from independently selected
  morphology metrics, flags adjustable high-`|r|` pairs and same-source transforms,
  and opens a neutral two-feature scatter when a cell is clicked. Pooled mode ignores
  group labels; an optional group scope restricts rows for biological/QC comparison.
  Styling includes preset or custom three-color maps, alpha, and hierarchical
  ordering. Direct export supports SVG/PDF and 300/600/1200-DPI PNG; full analysis
  exports add exact matrix/pair/settings tables and optional per-group matrices and
  figures.
- A seventh, separate advanced-clustering tab now keeps PCA as the default baseline
  while adding actual higher-dimensional UMAP and Gildenblat/Pahnke PCC/PCUMAP.
  The full 2-20D embedding is passed to GMM, Ward, or k-means; the 2D/3D plot uses
  coordinates from that same fitted space, with cluster colors and experimental-
  group marker shapes. Named advanced runs retain selected groups, reviewed-only
  filtering, reducer parameters, styling, and standalone-workbook support.
- Advanced exports include every embedding coordinate, implementation/effective
  parameters, trustworthiness, distance-rank preservation, repeated-seed embedding
  correlation and cluster ARI, plus high-resolution nonlinear plots. The current
  environment installs `pccdr==1.0.3` (TorchDR/PyTorch) and uses TorchDR for both
  modern UMAP and PCUMAP; `umap-learn` is the legacy macOS fallback. The Catalina/
  macOS 11 environment intentionally omits PCC/PCUMAP.

## Repository and release safety

The local development repository and public user repository are deliberately not
the same publishing target.

- Active local branch: `lsh`; `master` remains at the 0.7.0 development release.
- Local `origin`: `https://github.com/antamce/cluster-distribution-calculator.git`,
  the earlier raw/development sharing repository. It is no longer the desired public
  user destination.
- The low-memory, workflow, and macOS launcher work are pushed to the development
  remote as `origin/lsh`. Do not merge or push development history to the public
  repository.
- The presentable public repository is
  `https://github.com/antamce/cluster-distribution.git` on branch `main`.
- Beta 0.8.0 was published there from a clean temporary checkout as one curated
  public commit, not by changing the development checkout's `origin`.
- The portable Conda/macOS update was published on 2026-09-08 as curated commit
  `27bd6f3`, based on public commit `5523392`. Its temporary checkout was verified
  clean at the remote commit and deleted.
- The manual-correction update was published on 2026-09-09 as curated commit
  `5ad52bc`, based on public commit `27bd6f3`. The public commit and its README
  release note are in English; its temporary checkout was verified and deleted.
- The full post-testing workflow and transfer update was published on 2026-09-27
  as curated commit `a7991f5`, based on `5ad52bc`. Its exact 31-file public tree
  passed the complete 80-test suite (79 passed, 1 platform skip) and an offscreen
  application-construction smoke test before publication.
- The morphology-analysis and spine-volume-filter update was published on
  2026-10-06 as curated commit `aef5ce8`, based on `a7991f5`. Its exact 32-file
  public tree passed the complete 103-test suite (102 passed, 1 platform skip),
  Python compilation, and an offscreen seven-tab application smoke test.
- The feature-correlation and bulk distribution-review update was published on
  2026-10-09 as curated commit `4b872d8`, based on `aef5ce8`. Its exact 32-file
  public tree passed the complete 106-test suite (105 passed, 1 platform skip),
  Python compilation, and an offscreen seven-tab application smoke test.
- The public tree contains user-facing source, both environment files, Windows and
  macOS launcher files, launcher helpers, icons, package metadata, and README. It
  intentionally excludes `HANDOFF.md`, `tests/`, `PUBLIC_README.md`, development
  history, microscopy data, manifests, and Zarr caches.
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
requests it. The original alpha is `f3bf194`, beta 0.7.0 is `03a9108`, the initial
beta 0.8.0 is `54b77fa`, the preprocessing/review workflow update is `5523392`,
the macOS/portable-launcher update is `27bd6f3`, and the manual-correction update
is `5ad52bc`.

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

`launch_synpo.bat` delegates to `scripts/launch_synpo.ps1`. The resolver checks the
active and saved Conda, `PATH`, registered environments, registry/common locations,
and finally a native folder picker. It discovers `synpo-microscopy` with `conda
run`, so custom installation directories and custom `envs_dirs` work. Its per-user
choice lives at `%APPDATA%\Synpo\launcher-conda.txt`.

On macOS, run `chmod +x launch_synpo.command` once. It performs equivalent discovery
through the active/saved Conda, `PATH`, initialized zsh/bash, common locations, and
an AppleScript folder picker. It uses `environment.yml` on macOS 12+ and
`environment-macos-legacy.yml` on macOS 10.15/11. Catalina is Intel-only; macOS 11
and current macOS cover Intel and Apple Silicon. Its choice lives at
`~/Library/Application Support/Synpo/launcher-conda.txt`. This is a source/Conda
launcher, not a signed or notarized `.app` bundle. The current macOS work must be
field-tested before it is described as verified. `scripts/create_desktop_shortcut.ps1`
continues to create the Windows shortcut using the `.ico` asset.

Run the complete suite from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m pytest -q
```

Current Windows baseline: `105 passed, 1 skipped` (the macOS Bash execution test is
the expected skip). Also perform an offscreen application
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
- Preserve both Windows and macOS behavior. macOS 10.15/11 depends on the isolated
  legacy environment; do not loosen its pins without testing the Catalina device.

## Architecture map

- `launch_synpo.bat` and `scripts/launch_synpo.ps1`: Windows entry point and robust
  Conda/environment locator with a remembered per-user selection.
- `launch_synpo.command`: Finder/Terminal entry point, macOS version-aware
  environment creation, Conda discovery, native folder selection, and launch.
- `environment.yml`: current Windows and macOS 12+ environment.
- `environment-macos-legacy.yml`: macOS 10.15/11 compatibility environment using
  Python 3.10 and the official PySide6 6.2.4 wheel.
- `src/synpo/app.py`: PySide6 application, workflow tabs, background workers,
  progress/ETA plumbing, project state, viewers, review queues, dialogs and export
  orchestration. This is large; search for the relevant widget/class before editing.
- `src/synpo/models.py`: shared import/project model dataclasses.
- `src/synpo/calibration.py`: calibration presets and physical-unit handling.
- `src/synpo/importer.py`: configurable filename parsing, automatic/manual pairing,
  TIFF inspection, checksums, validation, and source metadata.
- `src/synpo/project.py`: schema-1 manifest creation/migration, atomic serialization,
  source verification, and recursive filename/checksum relinking (including renamed
  TIFF recovery by size plus SHA-256).
- `src/synpo/transfer.py`: versioned `.synpo-transfer.zip` creation, inventory and
  checksum validation, full-state/settings-only reset policy, transactional import,
  embedded-raw mapping, cache-path rewriting, and cache-damage recovery.
- `src/synpo/preprocessing.py`: per-stack adaptive background/threshold estimation,
  physical-unit Gaussian filtering, slice-wise execution, RAM guards, Zarr cache,
  signatures, and resume.
- `src/synpo/detection.py`: dendrite shaft, separated spine candidate, and protein
  cluster candidate detection; automatic/forced low-memory execution, streamed
  projections, cross-slab component reconciliation, 3D labels, flags, signatures,
  and checkpoints.
- `src/synpo/review.py`: editable mask initialization, signal-guided hint handling,
  local 3D resegmentation, object statuses, undo patches, and checkpoints.
- `src/synpo/diagnostics.py`: thread-safe append-only diagnostic sessions, sanitized
  environment/process inventory, two-second system sampling, JSON summaries, and
  ZIP packaging. Logging failures must never interrupt scientific work.
- `src/synpo/visualization.py`: slice-streamed XY/XZ/YZ projections, label overlays,
  physical aspect handling, cropped marching-cubes meshes, RAM/face limits, and the
  native Qt 3D renderer.
- `src/synpo/measurements.py`: association, cluster-end trimming, raw specimen/
  dendrite/spine/cluster metrics, all-spine length and geometry measurements,
  editable morphology review, checkpoint signatures, resume, and comparison
  illustrations.
- `src/synpo/morphology.py`: named morphology clustering runs, custom PCA,
  GMM/Ward/k-means fitting, candidate and specimen-bootstrap diagnostics,
  descriptive-only protein summaries, arbitrary selected-feature plots, explicit
  reproducible PCA interpretation, standalone workbook input, and separate
  analysis/figure exports.
- `src/synpo/distribution.py`: calibrated curved spine axes, endpoint hints, A* path
  completion, ten-bin voxel assignment, review state, aggregation and SEM.
- `src/synpo/exporting.py`: verified Excel/CSV exports and optional validation/audit
  PDF generation.
- `src/synpo/cli.py`: non-GUI scan/inspection command.
- `tests/`: 106-test development suite, including importer, project, preprocessing,
  detection, diagnostics, review, measurement, 3D rendering, and beta UI regression
  coverage.

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

- Scrollable Z viewer, zoom, contrast adjustment and direct clicked-slice mapping.
- Independent channel settings: adaptive background percentile, physical XY/Z
  Gaussian smoothing and sensitivity/threshold controls with expanded beta ranges.
- Every specimen has independent ChanA/ChanB parameters by default. Channels are
  marked individually and the specimen is green/reviewed only when both are set.
  A fill-unset checkbox copies the current-channel values only to unmarked images.
- Included images with unset parameters block batch preprocessing. Up/Down moves
  between specimens from the combo or canvas.
- Whole specimen pairs may be reversibly excluded with a reason. Multiple XY
  rectangular ROIs apply across both channels/all Z; they clip at the boundary and
  touching/overlapping fields merge. Original coordinates/raw pixels are retained.
- Parameter or ROI changes invalidate affected downstream checkpoints. Explicit
  selected-specimen preprocessing continues into forced detection and permanently
  discards that specimen's corrections/history after confirmation.
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
- Settings may be saved per specimen. Selected-specimen detection reruns only that
  pair and permanently discards its prior review cache/history after confirmation.
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
- Actions: add, projection-only dendrite-to-spine and spine-to-dendrite mask
  transfers, exclude, exclude as filopodium, split, merge, expand, trim, literal
  eraser, accept, and flag as needing attention.
- Hint colors are add green, exclude red, trim magenta, expand blue, split yellow,
  filopodium purple, merge `#ED6291`, accept cyan, and needs-attention orange.
- Hints can be drawn on Z slices or XY maximum projections. Projection hints search
  touched labels through Z or infer a plane from strongest nearby processed signal.
- Added-object boundaries come from the local preprocessed dendrite-channel signal.
  Touching new regions become one object; a region touching exactly one existing
  same-category object joins it, while contact with multiple existing IDs is
  skipped. Pre-existing objects are never implicitly merged with each other.
- The projection-only transfer brush must touch exactly one spine and literally
  moves every covered dendrite voxel through all Z slices to that spine. It changes
  neither background nor other masks and records/restores both label volumes in one
  undo checkpoint.
- The coral reverse-transfer brush must touch exactly one dendrite; zero or multiple
  touched dendrites are rejected without a write. It transfers every covered spine
  voxel from all covered spine IDs through all Z slices to the selected dendrite.
  Partial source objects retain their IDs, fully consumed spines decrement the
  count, and undo restores both masks and counts.
- Add, expand, and intensity-aware trim capture a 0.25–10.0 sensitivity value per
  applied action. Higher values grow Add/Expand and remove more weak Trim signal.
  Other actions visibly disable this control. Brush diameter ranges from 1–1000 px.
- The main review canvas has an optional stable high-contrast color per individual
  dendrite/spine ID; its type palettes are disjoint so the exact same color can
  never represent both types. Clusters retain their category color.
- Split always returns exactly two IDs: the two largest post-stroke components seed
  the result and all smaller pieces join the physically nearest seed. It rejects a
  stroke without a meaningful second component.
- Eraser clears painted dendrite and spine voxels without relabeling survivors:
  current Z only in slice mode, or painted XY columns across Z in projection mode.
- Hotkeys are Ctrl+1 Add, Ctrl+2 Exclude, Ctrl+3 Split, Ctrl+4 Merge, Ctrl+5
  spine-to-dendrite, Ctrl+6 dendrite-to-spine, Ctrl+7 Trim, Ctrl+8 Expand, Ctrl+9
  Eraser, Ctrl+0 Filopodium, and Enter Apply.
- Undo exists both for the current drawing and the last applied correction. Applied
  actions atomically checkpoint masks and object statuses.
- Review can be completed without corrections. Any later correction returns it to
  in-progress. Comments are per specimen.
- Correction memory mode automatically moves oversized edits to disk-backed
  workspaces; a forced slow low-memory option is also saved per project. Undo is
  retained for low-memory exclude, filopodium, merge, split, trim, and expand.
- **Advanced → Diagnostic mode** records correction/cache/projection phase timings
  and, at the default full-session level, other worker activity. Sessions live at
  `<output>/Synpo diagnostics/<timestamp>/`; they exclude image pixels, comments,
  command lines, and window titles, sanitize the user-profile prefix, and can be
  finalized as a ZIP. The single protocol PDF remains an application asset and is
  not copied into every session folder.
- One persistent Save-project button is outside the tab widget and remains
  available at every workflow stage without applying pending controls.

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
- Exactly two meaningful disconnected spine components may be connected by a
  signal-guided virtual centerline bridge up to a configurable 1.0 um default.
  The bridge does not alter masks or volumes, is previewed/exported, and defaults
  to excluded until reviewed. Tiny satellites are ignored for topology but remain
  in volume/bin assignment. A failed bridge or more than two substantial pieces
  retains the base-component path with a review-required status; a manual endpoint
  selects the distal component before bridging. Measurement algorithm version 3
  marks older completed measurement checkpoints for recalculation on project load
  without deleting saved distribution-review decisions.
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
- ROI IDs are attached to object rows. `ROI_Master` includes original rectangle
  coordinates and per-ROI summaries; specimen rows pool across ROIs, and
  `Excluded_Specimens` preserves exclusion reasons.
- CSV versions of every sheet are written and verified.
- Partial export is permitted and includes only completed measurement checkpoints;
  the measurement panel reports omitted unfinished pairs.
- Optional multi-page PDFs use two original-channel XY maximum panels plus the
  ten-color mask/centerline and individual line/bar distribution. Main,
  distribution-excluded, and invalid-spine audit PDFs are independent options.
- A saved post-measurement spine-volume filter uses one calibrated `um3` cutoff and
  the strict rule `volume_um3 < cutoff`; equality is retained. It operates on deep
  copies of saved measurement results and never modifies masks/results on disk.
- Per-spine `volume_filter_force_keep` decisions live beside distribution-review
  decisions. They override only the volume rule; manual invalidity always wins.
  Changing the cutoff/overrides refreshes in-app summaries without remeasurement.
- Stage 5 has a project-wide bulk acceptance checkbox for distribution review. It
  checkpoints all valid, non-volume-filtered spines with usable paths, preserves
  review notes/manual invalidations, and reports invalid, filtered, and pathless
  skips separately.
- Project exports omit volume-filtered spines from all primary/link tables and PDFs,
  recompute ROI/dendrite/specimen/group/profile statistics, and keep manual invalid
  and volume-filter-only audit rows separate. `Spine_Volume_Distribution` uses
  shared Freedman-Diaconis bins with Sturges/constant-data fallbacks and contains
  pooled plus specimen-weighted numeric results.
- **File -> Filter exported measurement workbook...** reconstructs results from
  detailed workbook sheets, writes a new workbook and CSV directory, preserves
  unrecognized sheets, and records exact/unavailable recalculations in
  `Compatibility_Report`. Unsupported legacy summaries are marked unavailable,
  never approximated. This standalone path cannot regenerate PDFs.

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
5. Run the full suite, application smoke test, and a representative small dataset
   on the user's Intel Catalina device. Then repeat at least the launcher/smoke test
   on an Apple Silicon macOS 12+ device. Catalina dependencies were cross-platform
   solved with `CONDA_OVERRIDE_OSX=10.15`; that is not a substitute for execution.
6. A signed/notarized `.app` or DMG remains future packaging work. The approved
   first macOS delivery is the Conda-backed `launch_synpo.command` workflow.

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
4. Run the full 47-test collection plus any relevant GUI smoke test. Exactly one
   platform-specific launcher test is expected to skip on Windows or macOS.
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

## State after the 2026-10-09 public release

- Development implementation: `lsh` commit `38ad078`, titled `Add feature
  correlation and bulk distribution review`; push it only to development
  `origin/lsh`.
- Public repository: `antamce/cluster-distribution`, branch `main`, verified remote
  commit `4b872d815411a7231e6305e0f9f37714864019c6`.
- The public commit contains 32 distributable files. It excludes tests,
  `HANDOFF.md`, `PUBLIC_README.md`, microscopy data, caches, and pytest-only
  dependencies/configuration. The public `README.md` came from `PUBLIC_README.md`
  and documents volume filtering, all-spine geometry review, PCA/UMAP/PCUMAP,
  cluster-count scoring, feature-vector visibility, feature-correlation matrices,
  bulk distribution acceptance, and all earlier workflows.
- Validation immediately before publication: the development tree and exact
  curated public source both passed all 106 tests (105 passed, 1 platform-specific
  launcher skip). The curated public application also passed Python compilation
  and offscreen construction with all seven tabs.
- Public push was verified with `git ls-remote`. The exact temporary checkout
  `C:\Users\user\Documents\code2\Synpo\.public-release-work` was confirmed clean at
  `4b872d8` and then deleted. The development worktree should be clean after this
  handoff record is committed.
- Next expected activity is user testing, particularly on the Intel Catalina
  device. Do not describe macOS as field-validated until those results arrive.
