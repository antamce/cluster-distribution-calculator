from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from openpyxl import Workbook, load_workbook

from .measurements import (
    distribution_summary_rows,
    load_distribution_preview,
    load_measurement_result,
)


Progress = Callable[[str, int, int, str], None]
BIN_COLORS = (
    (35, 0, 75), (75, 3, 110), (112, 14, 117), (147, 37, 103), (177, 63, 82),
    (204, 93, 58), (224, 127, 36), (239, 167, 25), (246, 210, 42), (240, 249, 33),
)


def _cell(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _columns(rows: list[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _group_summary(specimen_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups = sorted({str(row.get("experimental_group", "")) for row in specimen_rows})
    excluded = {"experimental_group", "specimen_id", "average_protein_distribution"}
    numeric = sorted(
        {
            key
            for row in specimen_rows
            for key, value in row.items()
            if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for group in groups:
        members = [row for row in specimen_rows if str(row.get("experimental_group", "")) == group]
        for metric in numeric:
            values = [float(row[metric]) for row in members if row.get(metric) is not None]
            if not values:
                continue
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            output.append(
                {
                    "experimental_group": group,
                    "metric": metric,
                    "n_specimens": len(values),
                    "mean": float(np.mean(values)),
                    "sd": sd,
                    "sem": sd / np.sqrt(len(values)),
                }
            )
    return output


def collect_export_tables(manifest: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    results = [
        load_measurement_result(manifest, index)
        for index, specimen in enumerate(manifest["specimens"])
        if specimen["checkpoints"].get("measurements", {}).get("state") == "complete"
    ]
    specimen = [row for result in results for row in result.get("specimen_rows", [])]
    roi = [row for result in results for row in result.get("roi_rows", [])]
    dendrite = [row for result in results for row in result.get("dendrite_rows", [])]
    spine = [row for result in results for row in result.get("spine_rows", [])]
    clusters = [
        row
        for result in results
        for row in result.get("cluster_rows", [])
        if row.get("row_type") == "individual_cluster"
    ]
    sums = [
        row
        for result in results
        for row in result.get("cluster_rows", [])
        if row.get("row_type") == "spine_cluster_sum"
    ]
    distributions = [row for result in results for row in result.get("distribution_rows", [])]
    distribution_specimen, distribution_group = distribution_summary_rows(results)
    excluded = [
        row
        for row in distributions
        if bool(row.get("spine_valid", True)) and not bool(row.get("distribution_included", False))
    ]
    invalid_ids = {
        (str(row.get("experimental_group", "")), str(row.get("specimen_id", "")), int(row["spine_id"]))
        for row in spine
        if not bool(row.get("spine_valid", True))
    }
    invalid = [
        row
        for row in spine
        if (str(row.get("experimental_group", "")), str(row.get("specimen_id", "")), int(row["spine_id"])) in invalid_ids
    ]
    reviewed_spines = [
        row
        for row in spine
        if bool(row.get("validity_reviewed", False))
        or bool(row.get("distribution_reviewed", False))
    ]
    settings = [
        {"section": "calibration", **dict(manifest["calibration"])},
        {"section": "measurements", **dict(manifest["measurements"]["settings"])},
        {
            "section": "distribution",
            "bin_count": 10,
            "orientation": "largest shaft contact to longest curved distal skeleton path",
            "aggregation": "spine means within specimen, then specimen means within group",
            "group_error_bars": "SEM across specimen means",
            "ratio_units": "fraction 0-1",
        },
    ]
    excluded_specimens = [
        {
            "experimental_group": specimen.get("experimental_group", ""),
            "specimen_id": specimen.get("specimen_id", ""),
            "reason": specimen.get("analysis", {}).get("exclusion_reason", ""),
        }
        for specimen in manifest["specimens"]
        if bool(specimen.get("analysis", {}).get("excluded", False))
    ]
    return {
        "Specimen_Master": sorted(specimen, key=lambda row: (str(row.get("experimental_group")), str(row.get("specimen_id")))),
        "ROI_Master": sorted(roi, key=lambda row: (str(row.get("experimental_group")), str(row.get("specimen_id")), int(row.get("roi_id", 0)))),
        "Dendrite_Master": sorted(dendrite, key=lambda row: (str(row.get("experimental_group")), str(row.get("specimen_id")), int(row.get("dendrite_id", 0)))),
        "Spine_Master": sorted(spine, key=lambda row: (str(row.get("experimental_group")), str(row.get("specimen_id")), int(row.get("spine_id", 0)))),
        "Cluster_Individual": clusters,
        "Cluster_Sums": sums,
        "Distribution_Individual": distributions,
        "Distribution_Specimen": distribution_specimen,
        "Distribution_Group": distribution_group,
        "Distribution_Excluded": excluded,
        "Invalid_Spines": invalid,
        "Spine_Review_Audit": reviewed_spines,
        "Excluded_Specimens": excluded_specimens,
        "Group_Summary": _group_summary(specimen),
        "Settings": settings,
    }


def _write_workbook(path: Path, tables: dict[str, list[dict[str, object]]]) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows in tables.items():
        sheet = workbook.create_sheet(name[:31])
        columns = _columns(rows)
        if columns:
            sheet.append(columns)
            for row in rows:
                sheet.append([_cell(row.get(column)) for column in columns])
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
        else:
            sheet.append(["No rows"])
    workbook.save(path)


def _write_csvs(directory: Path, tables: dict[str, list[dict[str, object]]]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, rows in tables.items():
        path = directory / f"{name}.csv"
        columns = _columns(rows)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            if columns:
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: _cell(row.get(key)) for key in columns})
        written.append(path)
    return written


def _pdf_pages_matplotlib_legacy(
    manifest: dict[str, object],
    path: Path,
    selected: list[tuple[int, dict[str, object]]],
    group_rows: list[dict[str, object]],
    margin_um: float,
    *,
    include_group_summary: bool,
    progress: Progress | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = np.asarray(BIN_COLORS, dtype=np.float64) / 255.0
    with PdfPages(path) as pdf:
        wrote_page = False
        if include_group_summary:
            for group in group_rows:
                means = np.asarray(
                    [
                        np.nan if group.get(f"bin_{index:02d}_mean") is None else float(group[f"bin_{index:02d}_mean"])
                        for index in range(1, 11)
                    ],
                    dtype=np.float64,
                )
                sem = np.asarray(
                    [
                        np.nan if group.get(f"bin_{index:02d}_sem") is None else float(group[f"bin_{index:02d}_sem"])
                        for index in range(1, 11)
                    ],
                    dtype=np.float64,
                )
                figure, axis = plt.subplots(figsize=(9, 6))
                x_values = np.arange(1, 11, dtype=np.float64)
                axis.plot(x_values, means, marker="o")
                for x_value, mean, error in zip(x_values, means, sem):
                    if not np.isfinite(mean) or not np.isfinite(error):
                        continue
                    axis.plot([x_value, x_value], [mean - error, mean + error], color="C0")
                    axis.plot([x_value - 0.08, x_value + 0.08], [mean - error, mean - error], color="C0")
                    axis.plot([x_value - 0.08, x_value + 0.08], [mean + error, mean + error], color="C0")
                axis.set(xlabel="Spine part (shaft → tip)", ylabel="Cluster volume / spine-part volume", title=f"{group['experimental_group']} — specimen-weighted mean ± SEM")
                axis.set_xlim(0.5, 10.5)
                axis.set_ylim(bottom=0)
                axis.grid(alpha=0.25)
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
                wrote_page = True
        for position, (specimen_index, row) in enumerate(selected):
            preview = load_distribution_preview(
                manifest, specimen_index, int(row["spine_id"]), margin_um=margin_um
            )
            figure = plt.figure(figsize=(11.7, 8.3))
            grid = figure.add_gridspec(2, 2, height_ratios=(3, 2))
            dendrite_axis = figure.add_subplot(grid[0, 0])
            protein_axis = figure.add_subplot(grid[0, 1])
            profile_axis = figure.add_subplot(grid[1, :])
            for axis, raw, bins, title in (
                (dendrite_axis, preview.dendrite_projection, preview.spine_bins_projection, "Dendrite/spine channel"),
                (protein_axis, preview.protein_projection, preview.cluster_bins_projection, "Protein channel"),
            ):
                axis.imshow(raw, cmap="gray", vmin=np.percentile(raw, 0.5), vmax=np.percentile(raw, 99.8))
                overlay = np.zeros((*bins.shape, 4), dtype=np.float64)
                for bin_index in range(1, 11):
                    overlay[bins == bin_index] = (*colors[bin_index - 1, :3], 0.55 if axis is dendrite_axis else 0.85)
                axis.imshow(overlay)
                if np.any(bins):
                    axis.contour(bins > 0, levels=[0.5], colors="white", linewidths=0.7)
                axis.set_title(title)
                axis.axis("off")
            if preview.axis_xy:
                dendrite_axis.plot(
                    [point[0] for point in preview.axis_xy],
                    [point[1] for point in preview.axis_xy],
                    color="white",
                    linewidth=1.2,
                )
            if preview.base_point_local_zyx is not None:
                dendrite_axis.scatter(
                    [preview.base_point_local_zyx[2]],
                    [preview.base_point_local_zyx[1]],
                    s=42,
                    c="#20d060",
                    edgecolors="black",
                    linewidths=0.6,
                    label="automatic base",
                    zorder=5,
                )
            if preview.endpoint_local_zyx is not None:
                dendrite_axis.scatter(
                    [preview.endpoint_local_zyx[2]],
                    [preview.endpoint_local_zyx[1]],
                    s=42,
                    c="#ed32c8",
                    edgecolors="black",
                    linewidths=0.6,
                    label=f"{row.get('centerline_endpoint_source', 'automatic')} endpoint",
                    zorder=5,
                )
            if preview.base_point_local_zyx is not None or preview.endpoint_local_zyx is not None:
                dendrite_axis.legend(loc="lower right", fontsize=7)
            ratios = [row.get(f"bin_{index:02d}_ratio") for index in range(1, 11)]
            profile_axis.plot(range(1, 11), ratios, marker="o")
            profile_axis.set(xlabel="Spine part (shaft → tip)", ylabel="Cluster volume / spine-part volume")
            profile_axis.set_ylim(bottom=0)
            profile_axis.grid(alpha=0.25)
            figure.suptitle(
                f"{row['experimental_group']} | {row['specimen_id']} | spine {row['spine_id']}\n"
                f"Axis: {row['distribution_axis_status']} | included: {row.get('distribution_included')} | "
                f"valid: {row.get('spine_valid')} | endpoint: "
                f"{row.get('centerline_endpoint_source', 'automatic')} | {row.get('review_note', '')}"
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            wrote_page = True
            if progress:
                progress("Exporting validation PDF", position + 1, len(selected), f"Spine {row['spine_id']}")
        if not wrote_page:
            figure, axis = plt.subplots(figsize=(9, 6))
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No spines matched this optional PDF category.",
                ha="center",
                va="center",
                fontsize=14,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def _pdf_pages(
    manifest: dict[str, object],
    path: Path,
    selected: list[tuple[int, dict[str, object]]],
    group_rows: list[dict[str, object]],
    margin_um: float,
    *,
    include_group_summary: bool,
    progress: Progress | None,
) -> None:
    """Render validation pages without depending on matplotlib native DLLs."""
    from PIL import Image, ImageDraw, ImageFont

    page_size = (1400, 1000)
    font = ImageFont.load_default()

    def page_and_draw():  # type: ignore[no-untyped-def]
        page = Image.new("RGB", page_size, "white")
        return page, ImageDraw.Draw(page)

    def safe_text(value: object) -> str:
        return str(value).encode("ascii", "replace").decode("ascii")

    def draw_chart(draw, values: np.ndarray, errors: np.ndarray | None, box):  # type: ignore[no-untyped-def]
        left, top, right, bottom = box
        finite = values[np.isfinite(values)]
        maximum = float(np.max(finite)) if finite.size else 1.0
        if errors is not None:
            upper = values + np.nan_to_num(errors, nan=0.0)
            upper = upper[np.isfinite(upper)]
            maximum = max(maximum, float(np.max(upper)) if upper.size else 1.0)
        maximum = max(maximum, 1e-12)
        draw.rectangle(box, outline=(80, 80, 80), width=2)
        points: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            x = int(left + (index + 0.5) * (right - left) / len(values))
            y = int(bottom - float(value) / maximum * (bottom - top))
            points.append((x, y))
            if errors is not None and np.isfinite(errors[index]):
                delta = int(float(errors[index]) / maximum * (bottom - top))
                draw.line((x, y - delta, x, y + delta), fill=(35, 100, 175), width=2)
                draw.line((x - 5, y - delta, x + 5, y - delta), fill=(35, 100, 175), width=2)
                draw.line((x - 5, y + delta, x + 5, y + delta), fill=(35, 100, 175), width=2)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(35, 100, 175))
            draw.text((x - 3, bottom + 8), str(index + 1), fill="black", font=font)
        if len(points) > 1:
            draw.line(points, fill=(35, 100, 175), width=3)

    def projection(raw: np.ndarray, bins: np.ndarray, alpha: float):  # type: ignore[no-untyped-def]
        array = np.asarray(raw, dtype=np.float64)
        low, high = np.percentile(array, (0.5, 99.8))
        gray = np.clip((array - low) / max(float(high - low), 1e-12) * 255.0, 0, 255)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        for bin_index, color in enumerate(BIN_COLORS, start=1):
            mask = np.asarray(bins) == bin_index
            rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.asarray(color)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    def paste_projection(page, source, box):  # type: ignore[no-untyped-def]
        left, top, right, bottom = box
        available = (right - left, bottom - top)
        resized = source.copy()
        resized.thumbnail(available, Image.Resampling.NEAREST)
        x = left + (available[0] - resized.width) // 2
        y = top + (available[1] - resized.height) // 2
        page.paste(resized, (x, y))
        return x, y, resized.width / source.width, resized.height / source.height

    pages = []
    if include_group_summary:
        for group in group_rows:
            means = np.asarray([
                np.nan if group.get(f"bin_{index:02d}_mean") is None else float(group[f"bin_{index:02d}_mean"])
                for index in range(1, 11)
            ])
            sem = np.asarray([
                np.nan if group.get(f"bin_{index:02d}_sem") is None else float(group[f"bin_{index:02d}_sem"])
                for index in range(1, 11)
            ])
            page, draw = page_and_draw()
            title = f"{group['experimental_group']} - specimen-weighted mean +/- SEM"
            draw.text((70, 45), safe_text(title), fill="black", font=font)
            draw_chart(draw, means, sem, (100, 130, 1320, 850))
            draw.text((520, 900), "Spine part (shaft -> tip)", fill="black", font=font)
            pages.append(page)

    for position, (specimen_index, row) in enumerate(selected):
        preview = load_distribution_preview(
            manifest, specimen_index, int(row["spine_id"]), margin_um=margin_um
        )
        page, draw = page_and_draw()
        title = (
            f"{row['experimental_group']} | {row['specimen_id']} | spine {row['spine_id']} | "
            f"axis: {row['distribution_axis_status']} | included: {row.get('distribution_included')} | "
            f"valid: {row.get('spine_valid')} | endpoint: {row.get('centerline_endpoint_source', 'automatic')}"
        )
        draw.text((45, 30), safe_text(title), fill="black", font=font)
        draw.text((45, 55), safe_text(row.get("review_note", "")), fill="black", font=font)
        left_box = (40, 105, 680, 570)
        right_box = (720, 105, 1360, 570)
        left_image = projection(preview.dendrite_projection, preview.spine_bins_projection, 0.55)
        right_image = projection(preview.protein_projection, preview.cluster_bins_projection, 0.85)
        left_x, left_y, scale_x, scale_y = paste_projection(page, left_image, left_box)
        paste_projection(page, right_image, right_box)
        draw.text((40, 85), "Dendrite/spine channel", fill="black", font=font)
        draw.text((720, 85), "Protein channel", fill="black", font=font)
        if preview.axis_xy:
            axis_points = [
                (left_x + int(point[0] * scale_x), left_y + int(point[1] * scale_y))
                for point in preview.axis_xy
            ]
            if len(axis_points) > 1:
                draw.line(axis_points, fill="white", width=3)
        for point, color in (
            (preview.base_point_local_zyx, (32, 208, 96)),
            (preview.endpoint_local_zyx, (237, 50, 200)),
        ):
            if point is not None:
                x = left_x + int(point[2] * scale_x)
                y = left_y + int(point[1] * scale_y)
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="black", width=2)
        ratios = np.asarray([
            np.nan if row.get(f"bin_{index:02d}_ratio") is None else float(row[f"bin_{index:02d}_ratio"])
            for index in range(1, 11)
        ])
        draw_chart(draw, ratios, None, (100, 650, 1320, 910))
        draw.text((520, 940), "Spine part (shaft -> tip)", fill="black", font=font)
        pages.append(page)
        if progress:
            progress("Exporting validation PDF", position + 1, len(selected), f"Spine {row['spine_id']}")

    if not pages:
        page, draw = page_and_draw()
        draw.text((500, 480), "No spines matched this optional PDF category.", fill="black", font=font)
        pages.append(page)
    pages[0].save(path, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])


def export_measurements(
    manifest: dict[str, object],
    workbook_path: str | Path,
    *,
    validation_pdf: bool = False,
    excluded_audit_pdf: bool = False,
    invalid_audit_pdf: bool = False,
    pdf_margin_um: float = 1.0,
    progress: Progress | None = None,
) -> dict[str, object]:
    path = Path(workbook_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = collect_export_tables(manifest)
    _write_workbook(path, tables)
    csv_directory = path.parent / f"{path.stem}_csv"
    csv_paths = _write_csvs(csv_directory, tables)
    written_pdfs: list[Path] = []
    indexed_rows: list[tuple[int, dict[str, object]]] = []
    for specimen_index, _specimen in enumerate(manifest["specimens"]):
        try:
            result = load_measurement_result(manifest, specimen_index)
        except ValueError:
            continue
        indexed_rows.extend((specimen_index, row) for row in result.get("distribution_rows", []))
    selections = []
    if validation_pdf:
        selections.append((path.with_name(f"{path.stem}_distribution_validation.pdf"), [item for item in indexed_rows if bool(item[1].get("spine_valid", True)) and bool(item[1].get("distribution_included", False))], True))
    if excluded_audit_pdf:
        selections.append((path.with_name(f"{path.stem}_distribution_excluded_audit.pdf"), [item for item in indexed_rows if bool(item[1].get("spine_valid", True)) and not bool(item[1].get("distribution_included", False))], False))
    if invalid_audit_pdf:
        selections.append((path.with_name(f"{path.stem}_invalid_spines_audit.pdf"), [item for item in indexed_rows if not bool(item[1].get("spine_valid", True))], False))
    for pdf_path, selected, summaries in selections:
        _pdf_pages(
            manifest,
            pdf_path,
            selected,
            tables["Distribution_Group"],
            pdf_margin_um,
            include_group_summary=summaries,
            progress=progress,
        )
        written_pdfs.append(pdf_path)

    # Verification happens before the UI may offer any future cache deletion.
    load_workbook(path, read_only=True).close()
    if not csv_paths or any(not csv_path.is_file() for csv_path in csv_paths):
        raise OSError("CSV export verification failed.")
    for pdf_path in written_pdfs:
        with pdf_path.open("rb") as stream:
            if stream.read(4) != b"%PDF":
                raise OSError(f"PDF verification failed: {pdf_path.name}")
    return {
        "workbook": str(path),
        "csv_directory": str(csv_directory),
        "pdfs": [str(item) for item in written_pdfs],
        "verified": True,
    }
