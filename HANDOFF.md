# Synpo Development Handoff

## Current state

Synpo is a Python/PySide6 desktop application for paired-channel, registered 3D
microscopy TIFF stacks. Version `0.3.0` implements and has user approval for:

- Stage 1: batch import, filename parsing, pairing, validation, calibration,
  project manifests, fingerprints, reopening, verification, and relinking;
- Stage 2: adaptive preprocessing previews and resumable compressed batch caching;
- Stage 3: automatic dendrite-shaft, spine-candidate, and protein-cluster-candidate
  detection with a slice viewer and colored overlays.

The user specifically approved the current detection/segmentation quality. The
latest UI change places all Automatic Detection settings and actions in a
scrollable side panel. Only the Z slider remains over the image.

Stage 4 has not started. Do not skip the staged approval process: complete a stage,
let the user test it, and wait for approval before beginning the next stage.

## Running and testing

The dedicated environment is `synpo-microscopy`; do not modify the user's separate
`synpo` environment.

```powershell
conda env update -f environment.yml --prune
launch_synpo.bat
```

The launcher calls the dedicated environment's Python executable directly, so it
also works when `conda` is absent from `PATH`.

Run tests from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
& "$env:USERPROFILE\miniconda3\envs\synpo-microscopy\python.exe" -m pytest
```

Current result: 11 tests pass. The supplied full-resolution pair benchmarked at
0.18 seconds for preprocessing preview, 6.54 seconds per preprocessed channel, and
22.88 seconds for automatic detection of the pair. Unchanged detection checkpoints
reopen in about 0.019 seconds.

## Input contract

- All TIFFs are directly inside one selected folder.
- Pair names differ only in `ChanA` versus `ChanB`.
- Schema:
  `<batch_prefix>_<experimental_group>_<specimen_id>_<ChanA|ChanB>_registered.tif`.
- A folder uses one filename schema and one channel-role assignment.
- The supplied data use ChanA for protein clusters and ChanB for dendrites/spines.
- Typical data are uint16, 2048 × 2048, and 40–100 Z slices, with about 100 pairs
  per batch.
- Stacks are already XY-registered and paired dimensions must be identical.
- Calibration must be entered/confirmed for every batch because embedded TIFF
  calibration may be wrong. The example preset is XY `0.0462584 µm/pixel` and Z
  `0.5 µm`.

## Important scientific and workflow requirements

- Intensity measurements must always use original uint16 voxels, never processed
  cache values.
- The app must use no more than 80% of device RAM.
- Preprocessing target: under one minute per channel.
- Detection target: under three minutes per pair.
- Long operations must keep the interface responsive and show progress/ETA.
- Preprocessing, detection, and review require automatic per-pair checkpoints.
- Detection deliberately keeps extra candidates for later exclusion and flags
  uncertain objects.
- Dendritic fields may contain multiple or crossing dendrites. Cell identity is
  irrelevant. Axons and soma portions must ultimately be excludable by user hints.
- Spine volume includes the neck up to the shaft. Filopodia are not automatically
  discarded. Closely touching spines should be separate objects.
- Manual edits apply only to the current specimen and trigger local resegmentation;
  drawings are hints, not final masks.
- Manual actions requested for the review stage include add, remove/exclude, split,
  merge, and boundary correction. Protein clusters do not need drawing hints.
- Correction should become usable after the first two detected pairs while later
  pairs continue automatically.

## Architecture and important files

- `src/synpo/app.py`: PySide6 application, background workers, three workflow tabs,
  preview and detection viewers.
- `src/synpo/importer.py`: filename parsing, TIFF metadata inspection, pairing, and
  source fingerprints.
- `src/synpo/project.py`: schema-1 JSON manifests, backward-compatible field
  migration, atomic saves, verification, and relinking.
- `src/synpo/preprocessing.py`: adaptive background/Otsu estimation, physical-unit
  Gaussian filtering, slice-at-a-time processing, RAM guard, Zarr cache, resume.
- `src/synpo/detection.py`: projection/skeleton-based shaft and spine separation,
  3D candidate label volumes, cluster components, flags, signatures, and pair
  checkpoints.
- `tests/`: importer, project, preprocessing, detection, checkpoint, migration, and
  resume coverage.

Project manifests remain schema version 1 and are extended through
`migrate_manifest`. Preserve backward compatibility with Stage 1/2 projects.

Preprocessed arrays are stored below:

```text
<output>/.synpo-cache/<project-id>/preprocessed.zarr
```

Detection label volumes are stored in `detection.zarr` beside that cache. Cache
groups carry settings/source signatures. Stale automatic results are replaced;
matching complete groups are reused. Raw TIFFs are never changed or copied.

## Stage 3 behavior

- Higher sensitivity retains more candidates.
- Dendrite masks are green, spine candidates cyan, and protein clusters magenta.
- Every automatic object begins as unreviewed.
- Additional flags cover possible filopodia, possible dendrite ends/edge objects,
  and weak/small clusters.
- A completed pair becomes viewable while later pairs continue detection.
- The user liked the current automatic segmentation, so retain current defaults
  unless evaluation on additional data justifies a change.

## Pending stages and decisions

The next expected stage is interactive review/correction with local
resegmentation. Later work must also cover:

1. Review queue, user hint drawing, add/remove/split/merge, per-object flags,
   optional filopodia exclusion, and per-specimen review checkpoints.
2. Max projection, linked XZ/YZ views, optional rotatable 3D rendering, and saved
   3D snapshots.
3. Spine/cluster association with a configurable minimum overlap percentage,
   default 80%. Initially count only the cluster portion inside the spine.
4. A comparison of two approaches for discarding the blurry slices at one end of
   clusters, including an illustration of voxels counted by the second approach.
5. Metrics:
   - spine: volume, cluster presence, summed cluster-volume/spine-volume ratio,
     and protein distribution;
   - cluster inside a spine: volume and distribution relative to the spine;
   - dendrite: spine density per µm, mean spine volume, inclusion percentage,
     mean summed inclusion/spine ratio, mean cluster volume, and protein
     distribution;
   - specimen: averages of the requested lower-level metrics.
6. Protein-distribution definitions are intentionally pending user clarification.
7. Export one Excel workbook with raw specimen/spine/cluster sheets, master sheets,
   and compact group summaries containing counts, means, variability, and inclusion
   percentages; also export CSV copies, reproducible settings/masks, ImageJ-compatible
   ROI ZIPs, and manually requested snapshots.
8. Keep the project cache compressed and temporary. Offer deletion only after final
   export has been verified.
9. An optional in-app annotation/training utility may be added after the main app;
   it must not depend on Fiji. The user can provide up to two annotated pairs.
10. Statistics are out of scope: the app calculates and exports clearly labeled
    metrics, while statistical analysis happens elsewhere.

The current target is Windows with a reproducible Anaconda environment and desktop
shortcut. Keep the design portable enough for a future macOS version.
