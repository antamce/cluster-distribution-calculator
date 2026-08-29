# Synpo Microscopy Processor

Synpo is a staged Windows desktop application for paired-channel 3D microscopy
data. Stage 1 provides project creation, TIFF pairing and validation, channel-role
assignment, physical calibration presets, source fingerprints, project reopening,
and source-folder relinking. Stage 2 adds adaptive preprocessing previews and a
resumable compressed batch cache. Stage 3 adds primary dendrite, spine, and
protein-cluster candidate detection.

## Current project status

The current application version is `0.3.0`. Stages 1–3 are implemented and have
been tested by the user. Stage 4 interactive review and local resegmentation have
not started. See [HANDOFF.md](HANDOFF.md) for architecture, scientific constraints,
benchmarks, and the remaining roadmap.

This repository intentionally excludes microscopy TIFFs, project manifests, and
generated Zarr caches. These are user data/runtime artifacts rather than source
files.

## Setup

From an Anaconda Prompt in this directory:

```powershell
conda env create -f environment.yml
launch_synpo.bat
```

For an existing environment, update it with:

```powershell
conda env update -f environment.yml --prune
```

`launch_synpo.bat` launches the source tree with the dedicated `synpo-microscopy`
environment's Python executable directly. It does not require the `conda` command
to be present on the Windows command prompt's `PATH`, and an editable pip
installation is intentionally not required.
Run `scripts/create_desktop_shortcut.ps1` once if a Windows desktop shortcut is
wanted.

## Expected filenames

Files must be directly inside the selected folder and follow:

```text
<batch_prefix>_<experimental_group>_<specimen_id>_<ChanA|ChanB>_registered.tif
```

The importer checks pairing, duplicates, TIFF dimensions, data type, axis metadata,
and channel-shape agreement without loading complete image stacks into RAM. SHA-256
fingerprints are computed in a background worker.

## Project files

Projects are human-readable `*.synpo.json` manifests. They contain paths,
calibration, exact file fingerprints, TIFF metadata, future-stage checkpoint slots,
and editable group/specimen labels. Raw TIFFs are never modified or copied.

Use **Verify sources** for full checksum verification. If a source folder moves,
use **Relink source folder**; the manifest is updated only when every expected file
matches its saved checksum.

## Command-line validation

The non-GUI importer can be exercised with:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
conda run -n synpo-microscopy python -m synpo.cli scan "C:\path\to\batch"
```

Add `--skip-checksums` only for a quick structural inspection.

## Stage 2 preprocessing

After saving or opening a project, open **2. Preprocessing**:

1. Choose one or more representative specimens and inspect both channels.
2. Scroll through Z, adjust black/white contrast, and inspect the magenta candidate
   overlay. Contrast affects display only.
3. Tune background percentile, physical-unit XY/Z Gaussian smoothing, and threshold
   sensitivity separately for ChanA and ChanB. A higher sensitivity retains more
   candidate voxels.
4. Select **Apply channel settings**, repeat for the other channel, then select
   **Preprocess entire batch / resume**.

Background and threshold values are estimated independently for every stack. The
processed data are used only for detection; future intensity measurements remain
tied to the original 16-bit TIFF voxels. Processing is slice-at-a-time and guarded
by the project's 80% RAM ceiling.

The compressed cache is stored under the output folder at
`.synpo-cache/<project-id>/preprocessed.zarr`. Each completed channel is checkpointed
in the project, and each written slice has a resume marker. Cancelling or restarting
the application therefore does not discard completed work. The app does not offer
cache deletion before verified final export.

On the supplied 40 × 2048 × 2048 example stack, the default settings took 6.54
seconds for one channel on the development machine; preview generation took 0.18
seconds. Actual timing depends on storage and CPU speed.

## Stage 3 automatic detection

After one or more pairs have completed preprocessing, open **3. Automatic
detection**. Detection settings are shared across the batch and include separate
dendrite/spine and cluster sensitivity, maximum terminal spine-branch length,
minimum dendrite length, and minimum candidate sizes.

Select **Run automatic detection / resume**. Each completed pair is checkpointed
immediately. Completed pairs become available in the detection viewer while later
pairs continue in the background. The viewer uses the original 16-bit channel as
its background and shows:

- green: dendrite shaft candidates;
- cyan: separated spine candidates;
- magenta: protein-cluster candidates.

The background channel, Z slice, contrast, and each overlay can be changed without
altering any data. Detection masks are stored in the project cache as compressed
3D label volumes. Changing a detection setting gives the pair a new reproducibility
signature and replaces only stale automatic masks.

The detector deliberately keeps extra candidates. Every automatic object starts
as unreviewed, while small/short clusters, possible filopodia, image-edge objects,
and possible dendrite ends receive additional flags. Exclusion, drawing hints,
merge/split corrections, and local resegmentation belong to the next review stage.

On the supplied registered pair, the Stage 3 defaults completed in 22.88 seconds
on the development machine, well below the three-minute target. Reopening an
unchanged completed checkpoint took 0.019 seconds.
