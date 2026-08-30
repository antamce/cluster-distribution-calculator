# Synpo Microscopy Processor

<p align="center">
  <img src="src/synpo/assets/synpo-icon.png" alt="Synpo icon" width="180">
</p>

Synpo is a Windows desktop application for processing paired-channel, 3D microscopy recordings of dendrites, dendritic spines, and protein clusters. It supports batch import, preprocessing, automatic segmentation, optional guided corrections, measurement, distribution review, and tabular export.

This is an alpha release intended for supervised scientific use. Review segmentation and centerline results before relying on exported measurements.

## What Synpo does

- Imports folders containing paired, registered 16-bit TIFF Z-stacks.
- Automatically pairs channel A and channel B files and organizes specimens by experimental group.
- Applies adaptive, per-stack background correction and smoothing without altering the source TIFFs.
- Detects dendrites, spines, and protein clusters in 3D.
- Lets you review Z slices, maximum projections, linked XZ/YZ views, and cropped rotatable 3D surfaces.
- Supports optional drawing hints for local resegmentation rather than requiring hand-drawn masks.
- Measures dendrites, individual spines, and qualifying clusters using the original 16-bit intensities.
- Measures protein distribution in ten equal-length parts along each spine's curved centerline.
- Lets you correct a spine's distal centerline endpoint with a single optional point hint.
- Displays experimental-group distribution profiles with SEM error bars.
- Exports Excel and CSV tables, settings, audit information, and optional validation PDFs.

Synpo calculates measurements only; statistical hypothesis testing should be performed in separate statistics software.

## Requirements

- Windows 10 or Windows 11
- Anaconda or Miniconda
- Enough free disk space for a compressed processing cache and exported results

The application is designed to work on ordinary laptop hardware and limits itself to at most 80% of available RAM. A macOS build is not included in this alpha release.

## Installation

Download or clone this repository, then open an **Anaconda Prompt** in its folder and run:

```powershell
conda env create -f environment.yml
launch_synpo.bat
```

The environment is named `synpo-microscopy`. If it already exists, update it with:

```powershell
conda env update -n synpo-microscopy -f environment.yml --prune
launch_synpo.bat
```

The launcher also works when `conda` is not recognized in an ordinary Command Prompt, provided Miniconda or Anaconda is installed in its usual location.

To create a Synpo desktop shortcut with the supplied icon, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_desktop_shortcut.ps1
```

## Input files

Place all TIFFs for one batch directly in a single folder. Each specimen must have a registered channel pair with identical dimensions:

```text
<batch prefix>_<experimental group>_<specimen ID>_ChanA_registered.tif
<batch prefix>_<experimental group>_<specimen ID>_ChanB_registered.tif
```

In the current workflow:

- Channel A contains protein clusters.
- Channel B contains dendrites and spines.

Channel roles can be confirmed for each batch. Always confirm the XY pixel size and Z step when starting a project; TIFF calibration metadata can be overridden and named presets can be reused.

## Typical workflow

1. Select the folder containing the paired TIFF stacks.
2. Confirm channel roles, voxel calibration, and output location.
3. Review the automatically paired files and parsed experimental groups.
4. Tune preprocessing on one or more representative specimens.
5. Preprocess the batch and run automatic detection.
6. Review detected specimens and optionally apply local corrections.
7. Review spine distributions, then export results.

Automatic checkpoints are written after preprocessing, detection, and review. Detection can continue in the background while completed specimens become available for correction.

## Centerline endpoint hints

The distribution-review screen normally uses the automatically detected base and distal endpoint. If a centerline is already correct, no action is required.

To correct only its distal endpoint:

1. Select **Centerline end hint**.
2. Use the cropped Z viewer; its slider is limited to slices occupied by the current spine.
3. Click the desired distal endpoint on the spine.

The point snaps to the nearest voxel belonging to that spine, the curved centerline is rebuilt from the automatic base, and the view returns to the maximum projection. **Clear end hint** restores automatic endpoint detection. Status messages appear without interrupting review.

## Results

The verified export includes:

- An Excel workbook containing specimen-, dendrite-, spine-, cluster-, and distribution-level sheets.
- CSV copies of the workbook tables.
- Individual ten-part distributions for every qualifying spine.
- Specimen and experimental-group summaries, including counts, means, variability, and inclusion percentages.
- Settings and review/audit records needed to interpret the result.
- Optional two-panel PDF pages showing each reviewed spine and its distribution profile.

Source TIFFs are never modified. Intensity-based measurements always use their original 16-bit voxel values. The compressed project cache can be removed after final export has been verified.

## Alpha-release notes

- Closely touching structures and unusual morphology may require review or correction.
- Filopodia are retained as candidates and can be excluded during review.
- Parts of somata and axons should be excluded with review tools when automatically retained.
- Segmentation-mask and ImageJ ROI ZIP export are not yet included in this alpha release.
- Report problems or suggestions through this repository's GitHub Issues page.
