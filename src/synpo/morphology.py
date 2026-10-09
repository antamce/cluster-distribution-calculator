from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.special import logsumexp

from .measurements import filtered_measurement_result
from .preprocessing import project_cache_path
from .project import save_project


MORPHOLOGY_FEATURES: dict[str, tuple[str, str]] = {
    "log_volume": ("Log spine volume", "volume_um3"),
    "linear_volume": ("Linear spine volume", "volume_um3"),
    "sqrt_volume": ("Square-root spine volume", "volume_um3"),
    "curvilinear_length": ("Curvilinear length", "spine_curvilinear_length_um"),
    "base_to_tip_distance": ("Base-to-tip distance", "spine_base_to_tip_distance_um"),
    "tortuosity": ("Centerline tortuosity", "centerline_tortuosity"),
    "maximum_width": ("Maximum width", "maximum_width_um"),
    "elongation": ("Principal-axis elongation", "principal_axis_elongation"),
    "head_width": ("Head maximum width", "head_maximum_width_um"),
    "neck_width": ("Neck median width", "neck_median_width_um"),
    "head_neck_ratio": ("Head/neck width ratio", "head_to_neck_width_ratio"),
    "head_volume": ("Head volume", "head_volume_um3"),
    "neck_volume": ("Neck volume", "neck_volume_um3"),
    "surface_area": ("Surface area", "surface_area_um2"),
    "sphericity": ("Sphericity", "sphericity"),
}

DEFAULT_FEATURES = (
    "log_volume",
    "curvilinear_length",
    "base_to_tip_distance",
    "tortuosity",
    "maximum_width",
    "elongation",
)

DEFAULT_COLORS = (
    "#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#9b5de5",
    "#e9c46a", "#00b4d8", "#f15bb5", "#6a994e", "#7f5539",
)

DEFAULT_PLOT_STYLE = {
    "axes_color": "#202020",
    "axes_alpha": 1.0,
    "background_color": "#ffffff",
    "background_alpha": 1.0,
    "show_legend": True,
    "legend_position": "outside_right",
    "custom_x_feature": "",
    "custom_y_feature": "",
    "pca_show_points": True,
    "pca_point_alpha": 0.68,
    "correlation_features": list(DEFAULT_FEATURES),
    "correlation_scope": "__pooled__",
    "correlation_threshold": 0.80,
    "correlation_colormap": "coolwarm",
    "correlation_negative_color": "#2166ac",
    "correlation_zero_color": "#f7f7f7",
    "correlation_positive_color": "#b2182b",
    "correlation_alpha": 1.0,
    "correlation_reorder": False,
    "correlation_export_group_matrices": False,
}


@dataclass(frozen=True)
class MorphologyClusteringSettings:
    algorithm: str = "gaussian_mixture"
    features: tuple[str, ...] = DEFAULT_FEATURES
    pca_dimensions: int = 2
    use_pca_for_clustering: bool = True
    minimum_clusters: int = 1
    maximum_clusters: int = 6
    fixed_cluster_count: int = 0
    cluster_count_selection: str = "information_criterion"
    scaling: str = "robust"
    random_seed: int = 42
    minimum_cluster_spines: int = 10
    minimum_cluster_fraction: float = 0.05
    included_groups: tuple[str, ...] = ()
    reviewed_only: bool = False
    reduction_method: str = "pca"
    embedding_dimensions: int = 5
    embedding_plot_dimensions: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_iterations: int = 500
    pcumap_reference_points: int = 100
    pcumap_beta: float = 10.0
    pcumap_correlation_weight: float = 90000.0
    pcumap_correlation_start: int = 10
    pcumap_device: str = "cpu"
    embedding_stability_repetitions: int = 3

    def validate(self) -> None:
        if self.algorithm not in {"gaussian_mixture", "ward", "kmeans"}:
            raise ValueError("Unknown morphology clustering algorithm.")
        if not self.features:
            raise ValueError("Select at least one morphology feature.")
        unknown = sorted(set(self.features) - set(MORPHOLOGY_FEATURES))
        if unknown:
            raise ValueError("Unknown morphology feature(s): " + ", ".join(unknown))
        volume = {"log_volume", "linear_volume", "sqrt_volume"} & set(self.features)
        if len(volume) > 1:
            raise ValueError("Choose only one spine-volume transformation.")
        if self.pca_dimensions not in {2, 3}:
            raise ValueError("PCA dimensionality must be 2 or 3.")
        if not 1 <= self.minimum_clusters <= self.maximum_clusters <= 10:
            raise ValueError("Candidate cluster range must be between 1 and 10.")
        if self.fixed_cluster_count and not (
            self.minimum_clusters <= self.fixed_cluster_count <= self.maximum_clusters
        ):
            raise ValueError("Fixed cluster count must fall inside the candidate range.")
        if self.cluster_count_selection not in {
            "information_criterion", "silhouette", "elbow"
        }:
            raise ValueError("Unknown automatic cluster-count selection method.")
        if not self.fixed_cluster_count:
            if self.cluster_count_selection == "silhouette" and self.maximum_clusters < 2:
                raise ValueError("Silhouette selection requires at least one candidate with two or more clusters.")
            if (
                self.cluster_count_selection == "elbow"
                and self.maximum_clusters - self.minimum_clusters < 2
            ):
                raise ValueError("Elbow selection requires at least three candidate cluster counts.")
        if self.scaling not in {"robust", "zscore", "none"}:
            raise ValueError("Unknown feature scaling method.")
        if self.minimum_cluster_spines < 2:
            raise ValueError("Minimum cluster size must be at least two spines.")
        if not 0.0 <= self.minimum_cluster_fraction <= 0.5:
            raise ValueError("Minimum cluster fraction must be between 0 and 0.5.")
        if any(not str(group).strip() for group in self.included_groups):
            raise ValueError("Experimental-group names cannot be blank.")
        if self.reduction_method not in {"pca", "umap", "pcumap"}:
            raise ValueError("Dimensionality reduction must be PCA, UMAP, or PCC/PCUMAP.")
        if not 2 <= self.embedding_dimensions <= 20:
            raise ValueError("Nonlinear embedding dimensionality must be between 2 and 20.")
        if (
            self.reduction_method in {"umap", "pcumap"}
            and self.embedding_dimensions > len(self.features)
        ):
            raise ValueError(
                "Nonlinear embedding dimensionality cannot exceed the number of "
                "selected morphology features."
            )
        if self.embedding_plot_dimensions not in {2, 3}:
            raise ValueError("Embedding plot dimensionality must be 2 or 3.")
        if self.embedding_plot_dimensions > self.embedding_dimensions:
            raise ValueError("Plot dimensionality cannot exceed the clustering embedding dimensionality.")
        if self.umap_n_neighbors < 2:
            raise ValueError("UMAP neighbors must be at least 2.")
        if not 0.0 <= self.umap_min_dist < 1.0:
            raise ValueError("UMAP minimum distance must be at least 0 and below 1.")
        if self.umap_metric not in {"euclidean", "manhattan"}:
            raise ValueError("The shared UMAP/PCUMAP metric must be Euclidean or Manhattan.")
        if not 50 <= self.umap_iterations <= 10_000:
            raise ValueError("Embedding iterations must be between 50 and 10,000.")
        if self.pcumap_reference_points < 2:
            raise ValueError("PCUMAP reference points must be at least 2.")
        if self.pcumap_beta <= 0.0:
            raise ValueError("PCUMAP beta must be positive.")
        if self.pcumap_correlation_weight < 0.0:
            raise ValueError("PCUMAP correlation weight cannot be negative.")
        if self.pcumap_correlation_start < 0:
            raise ValueError("PCUMAP correlation-loss start epoch cannot be negative.")
        if self.pcumap_device not in {"cpu", "auto", "cuda"}:
            raise ValueError("PCUMAP device must be CPU, automatic, or CUDA.")
        if not 1 <= self.embedding_stability_repetitions <= 10:
            raise ValueError("Embedding stability repetitions must be between 1 and 10.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "MorphologyClusteringSettings":
        settings = cls(
            algorithm=str(value.get("algorithm", "gaussian_mixture")),
            features=tuple(value.get("features", DEFAULT_FEATURES)),
            pca_dimensions=int(value.get("pca_dimensions", 2)),
            use_pca_for_clustering=bool(value.get("use_pca_for_clustering", True)),
            minimum_clusters=int(value.get("minimum_clusters", 1)),
            maximum_clusters=int(value.get("maximum_clusters", 6)),
            fixed_cluster_count=int(value.get("fixed_cluster_count", 0)),
            cluster_count_selection=str(
                value.get("cluster_count_selection", "information_criterion")
            ),
            scaling=str(value.get("scaling", "robust")),
            random_seed=int(value.get("random_seed", 42)),
            minimum_cluster_spines=int(value.get("minimum_cluster_spines", 10)),
            minimum_cluster_fraction=float(value.get("minimum_cluster_fraction", 0.05)),
            included_groups=tuple(str(group) for group in value.get("included_groups", ())),
            reviewed_only=bool(value.get("reviewed_only", False)),
            reduction_method=str(value.get("reduction_method", "pca")),
            embedding_dimensions=int(value.get("embedding_dimensions", 5)),
            embedding_plot_dimensions=int(value.get("embedding_plot_dimensions", 2)),
            umap_n_neighbors=int(value.get("umap_n_neighbors", 15)),
            umap_min_dist=float(value.get("umap_min_dist", 0.1)),
            umap_metric=str(value.get("umap_metric", "euclidean")),
            umap_iterations=int(value.get("umap_iterations", 500)),
            pcumap_reference_points=int(value.get("pcumap_reference_points", 100)),
            pcumap_beta=float(value.get("pcumap_beta", 10.0)),
            pcumap_correlation_weight=float(value.get("pcumap_correlation_weight", 90000.0)),
            pcumap_correlation_start=int(value.get("pcumap_correlation_start", 10)),
            pcumap_device=str(value.get("pcumap_device", "cpu")),
            embedding_stability_repetitions=int(value.get("embedding_stability_repetitions", 3)),
        )
        settings.validate()
        return settings


def morphology_analysis_directory(manifest: dict[str, object]) -> Path:
    return project_cache_path(manifest).parent / "morphology-analysis"


def morphology_run_path(manifest: dict[str, object], run_id: str) -> Path:
    return morphology_analysis_directory(manifest) / f"{run_id}.json.gz"


def collect_morphology_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for specimen_index, specimen in enumerate(manifest["specimens"]):
        if specimen["checkpoints"].get("measurements", {}).get("state") != "complete":
            continue
        result, _audit = filtered_measurement_result(manifest, specimen_index)
        spine_by_id = {int(row["spine_id"]): row for row in result.get("spine_rows", [])}
        distribution_by_id = {
            int(row["spine_id"]): row for row in result.get("distribution_rows", [])
        }
        cluster_by_spine: dict[int, list[dict[str, object]]] = {}
        for cluster in result.get("cluster_rows", []):
            if cluster.get("row_type") == "individual_cluster":
                cluster_by_spine.setdefault(int(cluster.get("spine_id") or 0), []).append(cluster)
        for morphology in result.get("morphology_rows", []):
            spine_id = int(morphology["spine_id"])
            spine = spine_by_id.get(spine_id, {})
            distribution = distribution_by_id.get(spine_id, {})
            clusters = cluster_by_spine.get(spine_id, [])
            bin_volumes = [
                float(distribution.get(f"bin_{index:02d}_cluster_volume_um3") or 0.0)
                for index in range(1, 11)
            ]
            total_bin_volume = sum(bin_volumes)
            centroid = (
                sum((index - 0.5) / 10.0 * value for index, value in enumerate(bin_volumes, start=1))
                / total_bin_volume
                if total_bin_volume > 0
                else None
            )
            rows.append(
                {
                    **dict(morphology),
                    "volume_um3": spine.get("volume_um3"),
                    "spine_valid": spine.get("spine_valid", True),
                    "manual_spine_valid": spine.get("manual_spine_valid", spine.get("spine_valid", True)),
                    "volume_filter_excluded": spine.get("volume_filter_excluded", False),
                    "volume_filter_force_keep": spine.get("volume_filter_force_keep", False),
                    "has_protein_cluster": spine.get("has_protein_cluster", False),
                    "protein_puncta_count": len(clusters),
                    "protein_puncta_total_volume_um3": spine.get("inside_cluster_volume_sum_um3", 0.0),
                    "protein_puncta_to_spine_volume_ratio": spine.get("cluster_to_spine_volume_ratio"),
                    "protein_puncta_mean_volume_um3": (
                        float(np.mean([float(item.get("volume_inside_spine_um3") or 0.0) for item in clusters]))
                        if clusters else None
                    ),
                    "protein_puncta_axis_centroid_fraction": centroid,
                    "protein_distribution_in_spine": spine.get("protein_distribution_in_spine"),
                    "distribution_axis_status": distribution.get("distribution_axis_status"),
                }
            )
    return rows


def _feature_value(row: dict[str, object], feature: str) -> float | None:
    _label, column = MORPHOLOGY_FEATURES[feature]
    value = row.get(column)
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    if feature == "log_volume":
        return float(np.log10(number)) if number > 0 else None
    if feature == "sqrt_volume":
        return float(np.sqrt(number)) if number >= 0 else None
    return number


def morphology_feature_value(
    row: dict[str, object], feature: str
) -> float | None:
    """Return the exact transformed value displayed for a clustering feature.

    New analysis runs store these values explicitly. Falling back to the raw
    morphology row keeps custom plots working for runs saved by older versions.
    """
    stored = row.get(f"clustering_feature_{feature}")
    if stored is not None:
        number = float(stored)
        return number if np.isfinite(number) else None
    return _feature_value(row, feature)


def _feature_expression(feature: str) -> str:
    _label, column = MORPHOLOGY_FEATURES[feature]
    if feature == "log_volume":
        return f"log10({column})"
    if feature == "sqrt_volume":
        return f"sqrt({column})"
    return column


def feature_correlation_data(
    result: dict[str, object],
    features: list[str] | tuple[str, ...],
    *,
    experimental_group: str | None = None,
    threshold: float = 0.80,
    reorder: bool = False,
) -> dict[str, object]:
    """Calculate a listwise-complete Pearson matrix across spine features.

    Experimental groups can optionally select a subset of spines, but group labels
    never enter the correlation calculation itself.
    """
    selected = list(dict.fromkeys(str(feature) for feature in features))
    unknown = [feature for feature in selected if feature not in MORPHOLOGY_FEATURES]
    if unknown:
        raise ValueError("Unknown correlation feature(s): " + ", ".join(unknown))
    if len(selected) < 2:
        raise ValueError("Select at least two metrics for the correlation matrix.")
    threshold = min(1.0, max(0.0, float(threshold)))
    rows = [
        dict(row)
        for row in result.get("assignments", [])
        if experimental_group is None
        or str(row.get("experimental_group", "")) == experimental_group
    ]
    complete_rows: list[dict[str, object]] = []
    vectors: list[list[float]] = []
    for row in rows:
        values = [morphology_feature_value(row, feature) for feature in selected]
        if all(value is not None for value in values):
            complete_rows.append(row)
            vectors.append([float(value) for value in values if value is not None])
    if len(complete_rows) < 3:
        scope = (
            f"experimental group {experimental_group!r}"
            if experimental_group is not None
            else "the pooled analysis population"
        )
        raise ValueError(
            f"Pearson correlation requires at least three listwise-complete spines; "
            f"{scope} has {len(complete_rows)}."
        )
    matrix_values = np.asarray(vectors, dtype=np.float64)
    standard_deviations = np.std(matrix_values, axis=0, ddof=1)
    correlation = np.full((len(selected), len(selected)), np.nan, dtype=np.float64)
    variable = standard_deviations > 1e-12
    if np.count_nonzero(variable):
        variable_values = matrix_values[:, variable]
        variable_correlation = np.atleast_2d(np.corrcoef(variable_values, rowvar=False))
        variable_indices = np.flatnonzero(variable)
        for local_row, matrix_row in enumerate(variable_indices):
            for local_column, matrix_column in enumerate(variable_indices):
                correlation[matrix_row, matrix_column] = float(
                    variable_correlation[local_row, local_column]
                )

    if reorder and len(selected) > 2:
        distance = 1.0 - np.abs(np.nan_to_num(correlation, nan=0.0))
        np.fill_diagonal(distance, 0.0)
        distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
        order = list(
            int(value)
            for value in leaves_list(
                linkage(squareform(distance, checks=False), method="average")
            )
        )
        selected = [selected[index] for index in order]
        matrix_values = matrix_values[:, order]
        standard_deviations = standard_deviations[order]
        correlation = correlation[np.ix_(order, order)]

    pairs: list[dict[str, object]] = []
    for row_index, first in enumerate(selected):
        for column_index in range(row_index + 1, len(selected)):
            second = selected[column_index]
            value = correlation[row_index, column_index]
            same_source = MORPHOLOGY_FEATURES[first][1] == MORPHOLOGY_FEATURES[second][1]
            high_correlation = bool(np.isfinite(value) and abs(float(value)) >= threshold)
            pairs.append(
                {
                    "feature_1": first,
                    "feature_1_label": MORPHOLOGY_FEATURES[first][0],
                    "feature_2": second,
                    "feature_2_label": MORPHOLOGY_FEATURES[second][0],
                    "pearson_r": float(value) if np.isfinite(value) else None,
                    "absolute_pearson_r": abs(float(value)) if np.isfinite(value) else None,
                    "high_correlation": high_correlation,
                    "same_source_measurement": same_source,
                    "warning": (
                        "same source measurement and high absolute correlation"
                        if same_source and high_correlation
                        else "same source measurement"
                        if same_source
                        else "high absolute correlation"
                        if high_correlation
                        else ""
                    ),
                }
            )
    return {
        "features": selected,
        "feature_labels": [MORPHOLOGY_FEATURES[feature][0] for feature in selected],
        "feature_sources": [MORPHOLOGY_FEATURES[feature][1] for feature in selected],
        "matrix": [
            [float(value) if np.isfinite(value) else None for value in row]
            for row in correlation
        ],
        "values_by_feature": {
            feature: [float(value) for value in matrix_values[:, index]]
            for index, feature in enumerate(selected)
        },
        "pairs": pairs,
        "included_spine_count": len(complete_rows),
        "candidate_spine_count": len(rows),
        "missing_spine_count": len(rows) - len(complete_rows),
        "experimental_group": experimental_group,
        "threshold": threshold,
        "reordered": bool(reorder),
        "constant_features": [
            feature
            for feature, spread in zip(selected, standard_deviations)
            if spread <= 1e-12
        ],
    }


def draw_feature_correlation(
    figure: object,
    result: dict[str, object],
    *,
    features: list[str] | tuple[str, ...],
    experimental_group: str | None = None,
    threshold: float = 0.80,
    reorder: bool = False,
    colormap: str = "coolwarm",
    negative_color: str = "#2166ac",
    zero_color: str = "#f7f7f7",
    positive_color: str = "#b2182b",
    alpha: float = 1.0,
    scatter_pair: tuple[str, str] | None = None,
) -> dict[str, object]:
    """Draw an annotated Pearson heatmap and optional selected-pair scatter."""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    data = feature_correlation_data(
        result,
        features,
        experimental_group=experimental_group,
        threshold=threshold,
        reorder=reorder,
    )
    figure.clear()
    show_scatter = bool(
        scatter_pair
        and scatter_pair[0] in data["features"]
        and scatter_pair[1] in data["features"]
    )
    grid = figure.add_gridspec(1, 2, width_ratios=(1.5, 1.0)) if show_scatter else None
    axis = figure.add_subplot(grid[0, 0] if grid is not None else 111)
    if colormap == "custom":
        cmap = LinearSegmentedColormap.from_list(
            "synpo_custom_correlation",
            [negative_color, zero_color, positive_color],
        )
    else:
        try:
            cmap = colormaps.get_cmap(colormap).copy()
        except ValueError:
            cmap = colormaps.get_cmap("coolwarm").copy()
    cmap.set_bad("#bdbdbd")
    matrix = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in data["matrix"]
        ],
        dtype=np.float64,
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        alpha=min(1.0, max(0.0, float(alpha))),
        aspect="equal",
    )
    labels = list(data["feature_labels"])
    axis.set_xticks(np.arange(len(labels)), labels, rotation=42, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    pair_lookup = {
        frozenset((str(row["feature_1"]), str(row["feature_2"]))): row
        for row in data["pairs"]
    }
    selected_features = list(data["features"])
    for row_index, first in enumerate(selected_features):
        for column_index, second in enumerate(selected_features):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                text_value = "N/A"
                flagged = False
                same_source = False
            else:
                pair = pair_lookup.get(frozenset((first, second)), {})
                flagged = bool(pair.get("high_correlation", False))
                same_source = bool(pair.get("same_source_measurement", False))
                text_value = f"{value:.2f}" + ("†" if same_source else "")
            axis.text(
                column_index,
                row_index,
                text_value,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if flagged or same_source else "normal",
                color="white" if np.isfinite(value) and abs(value) >= 0.58 else "black",
            )
    scope = (
        f"group: {experimental_group or '(blank)'}"
        if experimental_group is not None
        else "all included spines"
    )
    axis.set_title(
        f"Pearson correlation of morphology metrics ({scope}, n={data['included_spine_count']})"
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson r")
    if show_scatter and grid is not None and scatter_pair is not None:
        scatter_axis = figure.add_subplot(grid[0, 1])
        first, second = scatter_pair
        x_values = list(data["values_by_feature"][first])
        y_values = list(data["values_by_feature"][second])
        scatter_axis.scatter(x_values, y_values, s=18, alpha=0.58, color="#356b8c")
        first_index = selected_features.index(first)
        second_index = selected_features.index(second)
        value = matrix[first_index, second_index]
        scatter_axis.set(
            xlabel=MORPHOLOGY_FEATURES[first][0],
            ylabel=MORPHOLOGY_FEATURES[second][0],
            title=(
                f"Selected pair\nPearson r = {value:.4f}"
                if np.isfinite(value)
                else "Selected pair\nPearson r is undefined"
            ),
        )
        scatter_axis.grid(alpha=0.2)
    figure.text(
        0.01,
        0.01,
        "Bold values meet the warning threshold; † marks features derived from the same source measurement.",
        fontsize=8,
    )
    return data


def pca_interpretation_data(result: dict[str, object]) -> dict[str, object]:
    """Build reproducible PCA equations and feature-preparation metadata."""
    settings = dict(result.get("settings", {}))
    features = [
        str(feature)
        for feature in settings.get("features", [])
        if str(feature) in MORPHOLOGY_FEATURES
    ]
    loading_rows = {
        str(row.get("feature", "")): row
        for row in result.get("pca_loadings", [])
    }
    component_count = 0
    for component in range(1, 4):
        if any(
            row.get(f"pca_{component}_loading") is not None
            for row in loading_rows.values()
        ):
            component_count = component
    if not features or not component_count:
        return {"features": [], "feature_rows": [], "components": []}

    assignments = list(result.get("assignments", []))
    complete_values = []
    for row in assignments:
        values = [morphology_feature_value(row, feature) for feature in features]
        if all(value is not None for value in values):
            complete_values.append([float(value) for value in values])
    matrix = np.asarray(complete_values, dtype=np.float64)
    stored_centers = list(result.get("feature_centers", []))
    stored_scales = list(result.get("feature_scales", []))
    if len(stored_centers) == len(features) and len(stored_scales) == len(features):
        centers = np.asarray(stored_centers, dtype=np.float64)
        scales = np.asarray(stored_scales, dtype=np.float64)
    elif len(matrix):
        _scaled, centers, scales = _scale(
            matrix, str(settings.get("scaling", "robust"))
        )
    else:
        centers = np.zeros(len(features), dtype=np.float64)
        scales = np.ones(len(features), dtype=np.float64)
    scales = np.where(np.isfinite(scales) & (np.abs(scales) > 1e-12), scales, 1.0)

    stored_pca_means = list(result.get("pca_feature_means", []))
    if len(stored_pca_means) == len(features):
        pca_means = np.asarray(stored_pca_means, dtype=np.float64)
    elif len(matrix):
        pca_means = np.mean((matrix - centers) / scales, axis=0)
    else:
        pca_means = np.zeros(len(features), dtype=np.float64)

    loadings = np.asarray(
        [
            [
                float(loading_rows.get(feature, {}).get(f"pca_{component}_loading") or 0.0)
                for feature in features
            ]
            for component in range(1, component_count + 1)
        ],
        dtype=np.float64,
    )
    explained = []
    for component in range(1, component_count + 1):
        value = next(
            (
                row.get(f"pca_{component}_explained_fraction")
                for row in loading_rows.values()
                if row.get(f"pca_{component}_explained_fraction") is not None
            ),
            0.0,
        )
        explained.append(float(value))

    feature_rows = []
    for index, feature in enumerate(features):
        feature_rows.append(
            {
                "feature_number": index + 1,
                "feature": feature,
                "label": MORPHOLOGY_FEATURES[feature][0],
                "transformed_expression": _feature_expression(feature),
                "scaling_method": str(settings.get("scaling", "robust")),
                "scaling_center": float(centers[index]),
                "scaling_scale": float(scales[index]),
                "pca_center_after_scaling": float(pca_means[index]),
                **{
                    f"pca_{component}_loading": float(
                        loadings[component - 1, index]
                    )
                    for component in range(1, component_count + 1)
                },
            }
        )

    components = []
    for component_index in range(component_count):
        coefficients = loadings[component_index] / scales
        intercept = float(
            np.sum(
                loadings[component_index]
                * (-centers / scales - pca_means)
            )
        )
        centered_terms = [
            (
                f"{loadings[component_index, index]:+.8g} * "
                f"((({_feature_expression(feature)} - {centers[index]:.8g}) / "
                f"{scales[index]:.8g}) - {pca_means[index]:.8g})"
            )
            for index, feature in enumerate(features)
        ]
        expanded_terms = [
            f"{coefficient:+.8g} * {_feature_expression(feature)}"
            for coefficient, feature in zip(coefficients, features)
        ]
        components.append(
            {
                "component": f"PC{component_index + 1}",
                "explained_fraction": explained[component_index],
                "explained_percent": explained[component_index] * 100.0,
                "intercept": intercept,
                "centered_scaled_equation": (
                    f"PC{component_index + 1} = " + " ".join(centered_terms)
                ),
                "expanded_equation": (
                    f"PC{component_index + 1} = {intercept:.8g} "
                    + " ".join(expanded_terms)
                ),
                "loadings": [float(value) for value in loadings[component_index]],
            }
        )
    return {
        "features": features,
        "feature_rows": feature_rows,
        "components": components,
        "loadings": loadings,
        "explained": np.asarray(explained, dtype=np.float64),
    }


def draw_pca_interpretation(figure: object, result: dict[str, object]) -> None:
    """Draw a reproducible PCA interpretation dashboard on a Matplotlib figure."""
    from matplotlib.colors import to_rgba

    figure.clear()
    data = pca_interpretation_data(result)
    features = list(data.get("features", []))
    components = list(data.get("components", []))
    style = {**DEFAULT_PLOT_STYLE, **dict(result.get("plot_style", {}))}
    axes_rgba = to_rgba(
        str(style["axes_color"]), alpha=float(style["axes_alpha"])
    )
    background_rgba = to_rgba(
        str(style["background_color"]), alpha=float(style["background_alpha"])
    )
    figure.patch.set_facecolor(background_rgba)
    if not features or not components:
        axis = figure.add_subplot(111)
        axis.set_facecolor(background_rgba)
        axis.text(
            0.5,
            0.5,
            "PCA interpretation is unavailable for this saved analysis.",
            ha="center",
            va="center",
            color=axes_rgba,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return

    grid = figure.add_gridspec(2, 2, height_ratios=(1.12, 1.0))
    use_3d = len(components) >= 3 and any(
        row.get("pca_3") is not None for row in result.get("assignments", [])
    )
    score_axis = figure.add_subplot(
        grid[0, 0], projection="3d" if use_3d else None
    )
    loading_axis = figure.add_subplot(grid[0, 1])
    variance_axis = figure.add_subplot(grid[1, 0])
    equation_axis = figure.add_subplot(grid[1, 1])
    assignments = list(result.get("assignments", []))
    definitions = list(result.get("cluster_definitions", []))
    colors = {
        int(row["morphology_cluster_id"]): str(row.get("color", "#457b9d"))
        for row in definitions
    }
    groups = sorted(
        {str(row.get("experimental_group", "")) for row in assignments}
    )
    markers = ("o", "^", "s", "D", "P", "X", "v", "<", ">", "*")
    marker_by_group = {
        group: markers[index % len(markers)] for index, group in enumerate(groups)
    }
    show_points = bool(style.get("pca_show_points", True))
    point_alpha = min(
        1.0, max(0.0, float(style.get("pca_point_alpha", 0.68)))
    )
    for cluster in sorted(colors):
        for group in groups:
            members = [
                row
                for row in assignments
                if int(row.get("morphology_cluster_id", 0)) == cluster
                and str(row.get("experimental_group", "")) == group
                and row.get("pca_1") is not None
                and show_points
            ]
            if members:
                coordinates = (
                    [float(row["pca_1"]) for row in members],
                    [float(row.get("pca_2") or 0.0) for row in members],
                )
                if use_3d:
                    score_axis.scatter(
                        *coordinates,
                        [float(row.get("pca_3") or 0.0) for row in members],
                        color=colors[cluster],
                        marker=marker_by_group[group],
                        alpha=point_alpha,
                        s=22,
                        label=f"Cluster {cluster} · {group}",
                    )
                else:
                    score_axis.scatter(
                        *coordinates,
                        color=colors[cluster],
                        marker=marker_by_group[group],
                        alpha=point_alpha,
                        s=22,
                        label=f"Cluster {cluster} · {group}",
                    )

    loadings = np.asarray(data["loadings"], dtype=np.float64)
    score_x = np.asarray(
        [float(row["pca_1"]) for row in assignments if row.get("pca_1") is not None],
        dtype=np.float64,
    )
    score_y = np.asarray(
        [float(row.get("pca_2") or 0.0) for row in assignments if row.get("pca_1") is not None],
        dtype=np.float64,
    )
    score_z = np.asarray(
        [float(row.get("pca_3") or 0.0) for row in assignments if row.get("pca_1") is not None],
        dtype=np.float64,
    )
    x_span = float(np.ptp(score_x)) if len(score_x) else 1.0
    y_span = float(np.ptp(score_y)) if len(score_y) else x_span
    x_span = x_span if x_span > 1e-12 else 1.0
    y_span = y_span if y_span > 1e-12 else x_span
    z_span = float(np.ptp(score_z)) if len(score_z) else x_span
    z_span = z_span if z_span > 1e-12 else x_span
    max_x_loading = max(float(np.max(np.abs(loadings[0]))), 1e-12)
    max_y_loading = (
        max(float(np.max(np.abs(loadings[1]))), 1e-12)
        if len(loadings) > 1
        else 1.0
    )
    max_z_loading = (
        max(float(np.max(np.abs(loadings[2]))), 1e-12)
        if use_3d
        else 1.0
    )
    for index, feature in enumerate(features):
        color = DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        end_x = float(loadings[0, index]) / max_x_loading * 0.30 * x_span
        end_y = (
            float(loadings[1, index]) / max_y_loading * 0.30 * y_span
            if len(loadings) > 1
            else 0.0
        )
        end_z = (
            float(loadings[2, index]) / max_z_loading * 0.30 * z_span
            if use_3d
            else 0.0
        )
        if use_3d:
            score_axis.quiver(
                0.0,
                0.0,
                0.0,
                end_x,
                end_y,
                end_z,
                color=color,
                linewidth=1.6,
                arrow_length_ratio=0.12,
            )
            score_axis.text(
                end_x,
                end_y,
                end_z,
                f"F{index + 1}",
                color=color,
                fontsize=8,
            )
        else:
            score_axis.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(0.0, 0.0),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6},
            )
            score_axis.annotate(
                f"F{index + 1}",
                xy=(end_x, end_y),
                xytext=(
                    4 if end_x >= 0 else -4,
                    ((index % 5) - 2) * 6,
                ),
                textcoords="offset points",
                color=color,
                fontsize=8,
                ha="left" if end_x >= 0 else "right",
                va="center",
            )
    explained = np.asarray(data["explained"], dtype=np.float64)
    pc1_percent = explained[0] * 100.0
    pc2_percent = explained[1] * 100.0 if len(explained) > 1 else 0.0
    score_axis.set(
        xlabel=f"PC1 ({pc1_percent:.1f}%)",
        ylabel=(f"PC2 ({pc2_percent:.1f}%)" if len(components) > 1 else "No PC2"),
        title=(
            "3D PCA scores and loading vectors (vectors scaled to panel)"
            if use_3d
            else "PCA scores and loading vectors (vectors scaled to panel)"
        ),
    )
    if use_3d:
        score_axis.set_zlabel(f"PC3 ({explained[2] * 100.0:.1f}%)")
    else:
        score_axis.axhline(0.0, color=axes_rgba, lw=0.7, alpha=0.35)
        score_axis.axvline(0.0, color=axes_rgba, lw=0.7, alpha=0.35)

    heatmap = loading_axis.imshow(
        loadings,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max(1e-12, float(np.max(np.abs(loadings)))),
        vmax=max(1e-12, float(np.max(np.abs(loadings)))),
    )
    loading_axis.set_xticks(
        np.arange(len(features)), [f"F{index + 1}" for index in range(len(features))]
    )
    loading_axis.set_yticks(
        np.arange(len(components)), [str(row["component"]) for row in components]
    )
    loading_axis.set_title("PCA loadings")
    colorbar = figure.colorbar(heatmap, ax=loading_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Loading")

    positions = np.arange(1, len(components) + 1)
    variance_axis.bar(
        positions,
        explained * 100.0,
        color=[DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index in range(len(components))],
    )
    variance_axis.set_xticks(positions, [str(row["component"]) for row in components])
    variance_axis.set(
        ylabel="Explained variance (%)",
        title="Variance explained by each displayed component",
    )
    for position, value in zip(positions, explained * 100.0):
        variance_axis.text(
            position,
            float(value),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            color=axes_rgba,
            fontsize=8,
        )

    equation_axis.set_axis_off()
    formula_lines = ["Explicit equations in transformed feature units"]
    for component in components:
        readable = str(component["expanded_equation"])
        formula_lines.append(
            textwrap.fill(
                f"{component['component']} ({float(component['explained_percent']):.1f}%): "
                + readable.split(" = ", 1)[-1],
                width=70,
                subsequent_indent="    ",
            )
        )
    formula_lines.append("")
    formula_lines.append("Feature key:")
    feature_key = "; ".join(
        f"F{index + 1}={_feature_expression(feature)}"
        for index, feature in enumerate(features)
    )
    formula_lines.append(textwrap.fill(feature_key, width=70, subsequent_indent="    "))
    formula_lines.append("Full-precision centered/scaled equations are in the PCA_Equations export sheet.")
    equation_axis.text(
        0.0,
        1.0,
        "\n".join(formula_lines),
        transform=equation_axis.transAxes,
        ha="left",
        va="top",
        color=axes_rgba,
        fontsize=7 if len(features) > 8 else 8,
        family="monospace",
    )

    for axis in figure.axes:
        axis.set_facecolor(background_rgba)
        axis.tick_params(colors=axes_rgba)
        axis.xaxis.label.set_color(axes_rgba)
        axis.yaxis.label.set_color(axes_rgba)
        axis.title.set_color(axes_rgba)
        if hasattr(axis, "zaxis"):
            axis.zaxis.label.set_color(axes_rgba)
        for spine in axis.spines.values():
            spine.set_color(axes_rgba)
        axis.grid(alpha=0.16, color=axes_rgba)
    handles, labels = score_axis.get_legend_handles_labels()
    legend = None
    if bool(style.get("show_legend", True)) and handles:
        legend_position = str(style.get("legend_position", "outside_right"))
        if legend_position == "outside_bottom":
            legend = figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncols=min(3, len(handles)),
            )
        elif legend_position == "inside":
            legend = score_axis.legend(handles, labels, loc="upper right")
        else:
            legend = figure.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.0, 0.98),
            )
        legend.get_frame().set_facecolor(background_rgba)
        for legend_text in legend.get_texts():
            legend_text.set_color(axes_rgba)


def draw_protein_puncta_volume(axis: object, result: dict[str, object]) -> None:
    """Draw puncta volumes with cluster colors and group-specific point markers."""
    from matplotlib.lines import Line2D

    assignments = list(result.get("assignments", []))
    definitions = list(result.get("cluster_definitions", []))
    colors = {
        int(row["morphology_cluster_id"]): str(row.get("color", "#457b9d"))
        for row in definitions
    }
    groups = sorted(
        {str(row.get("experimental_group", "")) for row in assignments}
    )
    markers = ("o", "^", "s", "D", "P", "X", "v", "<", ">", "*")
    marker_by_group = {
        group: markers[index % len(markers)] for index, group in enumerate(groups)
    }
    cluster_values = []
    for cluster in sorted(colors):
        points = [
            row
            for row in assignments
            if int(row.get("morphology_cluster_id", 0)) == cluster
            and bool(row.get("has_protein_cluster"))
            and row.get("protein_puncta_total_volume_um3") is not None
        ]
        if points:
            cluster_values.append((cluster, points))
    if cluster_values:
        positions = list(range(1, len(cluster_values) + 1))
        artists = axis.boxplot(
            [
                [float(row["protein_puncta_total_volume_um3"]) for row in points]
                for _cluster, points in cluster_values
            ],
            positions=positions,
            tick_labels=[str(cluster) for cluster, _points in cluster_values],
            patch_artist=True,
            flierprops={
                "marker": "o",
                "markerfacecolor": "none",
                "markeredgecolor": "#111111",
                "markersize": 5,
                "linestyle": "none",
            },
        )
        for box, (cluster, _points) in zip(artists["boxes"], cluster_values):
            box.set_facecolor(colors[cluster])
            box.set_label(f"Cluster {cluster} box")
        for position, (cluster, points) in zip(positions, cluster_values):
            offsets = (
                np.linspace(-0.14, 0.14, len(points))
                if len(points) > 1
                else np.zeros(1)
            )
            for group in groups:
                indices = [
                    index
                    for index, row in enumerate(points)
                    if str(row.get("experimental_group", "")) == group
                ]
                if not indices:
                    continue
                axis.scatter(
                    [position + float(offsets[index]) for index in indices],
                    [
                        float(points[index]["protein_puncta_total_volume_um3"])
                        for index in indices
                    ],
                    s=22,
                    alpha=0.75,
                    color=colors[cluster],
                    marker=marker_by_group[group],
                    zorder=3,
                )
        axis.add_line(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="#777777",
                markeredgecolor="#777777",
                label="Colored points: individual protein-positive spines",
            )
        )
        axis.add_line(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#111111",
                label="Hollow black points: box-plot outliers (same spines)",
            )
        )
        for group in groups:
            axis.add_line(
                Line2D(
                    [],
                    [],
                    marker=marker_by_group[group],
                    linestyle="none",
                    markerfacecolor="#777777",
                    markeredgecolor="#777777",
                    label=f"Experimental group: {group or '(blank)'}",
                )
            )
    else:
        axis.text(
            0.5,
            0.5,
            "No protein-positive puncta volumes are available.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        xlabel="Morphology cluster",
        ylabel="Protein puncta volume in spine (µm³)",
        title="Protein puncta volume among protein-positive spines",
    )


def draw_pca_3d_feature_axes(figure: object, result: dict[str, object]) -> None:
    """Draw a full-size rotatable PC1/PC2/PC3 biplot with feature axes."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    figure.clear()
    data = pca_interpretation_data(result)
    features = list(data.get("features", []))
    components = list(data.get("components", []))
    assignments = list(result.get("assignments", []))
    style = {**DEFAULT_PLOT_STYLE, **dict(result.get("plot_style", {}))}
    axes_rgba = to_rgba(
        str(style["axes_color"]), alpha=float(style["axes_alpha"])
    )
    background_rgba = to_rgba(
        str(style["background_color"]), alpha=float(style["background_alpha"])
    )
    figure.patch.set_facecolor(background_rgba)
    if len(components) < 3 or not any(
        row.get("pca_3") is not None for row in assignments
    ):
        axis = figure.add_subplot(111)
        axis.set_facecolor(background_rgba)
        axis.text(
            0.5,
            0.5,
            "This view requires a morphology run calculated with 3D PCA.",
            ha="center",
            va="center",
            color=axes_rgba,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return

    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor(background_rgba)
    definitions = list(result.get("cluster_definitions", []))
    colors = {
        int(row["morphology_cluster_id"]): str(row.get("color", "#457b9d"))
        for row in definitions
    }
    groups = sorted(
        {str(row.get("experimental_group", "")) for row in assignments}
    )
    markers = ("o", "^", "s", "D", "P", "X", "v", "<", ">", "*")
    marker_by_group = {
        group: markers[index % len(markers)] for index, group in enumerate(groups)
    }
    show_points = bool(style.get("pca_show_points", True))
    point_alpha = min(
        1.0, max(0.0, float(style.get("pca_point_alpha", 0.68)))
    )
    for cluster in sorted(colors):
        for group in groups:
            members = [
                row
                for row in assignments
                if int(row.get("morphology_cluster_id", 0)) == cluster
                and str(row.get("experimental_group", "")) == group
                and row.get("pca_1") is not None
                and row.get("pca_2") is not None
                and row.get("pca_3") is not None
                and show_points
            ]
            if members:
                axis.scatter(
                    [float(row["pca_1"]) for row in members],
                    [float(row["pca_2"]) for row in members],
                    [float(row["pca_3"]) for row in members],
                    color=colors[cluster],
                    marker=marker_by_group[group],
                    alpha=point_alpha,
                    s=26,
                    label=f"Cluster {cluster} · {group}",
                )

    score_arrays = [
        np.asarray(
            [float(row[f"pca_{component}"]) for row in assignments],
            dtype=np.float64,
        )
        for component in (1, 2, 3)
    ]
    spans = [
        float(np.ptp(values)) if float(np.ptp(values)) > 1e-12 else 1.0
        for values in score_arrays
    ]
    loadings = np.asarray(data["loadings"], dtype=np.float64)
    maximum_loadings = [
        max(float(np.max(np.abs(loadings[index]))), 1e-12)
        for index in range(3)
    ]
    feature_handles = []
    for feature_index, feature in enumerate(features):
        color = DEFAULT_COLORS[feature_index % len(DEFAULT_COLORS)]
        endpoint = [
            float(loadings[component, feature_index])
            / maximum_loadings[component]
            * 0.30
            * spans[component]
            for component in range(3)
        ]
        axis.quiver(
            0.0,
            0.0,
            0.0,
            *endpoint,
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.12,
        )
        axis.text(
            *endpoint,
            f"F{feature_index + 1}",
            color=color,
            fontsize=9,
        )
        feature_handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=f"F{feature_index + 1}: {MORPHOLOGY_FEATURES[feature][0]}",
            )
        )

    explained = np.asarray(data["explained"], dtype=np.float64)
    axis.set(
        xlabel=f"PC1 ({explained[0] * 100.0:.1f}%)",
        ylabel=f"PC2 ({explained[1] * 100.0:.1f}%)",
        zlabel=f"PC3 ({explained[2] * 100.0:.1f}%)",
        title="3D PCA scores with morphology-feature loading axes",
    )
    axis.tick_params(colors=axes_rgba)
    axis.xaxis.label.set_color(axes_rgba)
    axis.yaxis.label.set_color(axes_rgba)
    axis.zaxis.label.set_color(axes_rgba)
    axis.title.set_color(axes_rgba)
    axis.grid(alpha=0.2, color=axes_rgba)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(feature_handles)
    labels.extend([handle.get_label() for handle in feature_handles])
    if bool(style.get("show_legend", True)) and handles:
        legend_position = str(style.get("legend_position", "outside_right"))
        if legend_position == "outside_bottom":
            legend = figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncols=min(3, len(handles)),
            )
        elif legend_position == "inside":
            legend = axis.legend(handles, labels, loc="upper right")
        else:
            legend = figure.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.0, 0.95),
            )
        legend.get_frame().set_facecolor(background_rgba)
        for legend_text in legend.get_texts():
            legend_text.set_color(axes_rgba)


def _scale(values: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "none":
        center = np.zeros(values.shape[1])
        scale = np.ones(values.shape[1])
    elif method == "zscore":
        center = np.mean(values, axis=0)
        scale = np.std(values, axis=0, ddof=1)
    else:
        center = np.median(values, axis=0)
        q1, q3 = np.percentile(values, [25.0, 75.0], axis=0)
        scale = q3 - q1
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (values - center) / scale, center, scale


def _matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(sum(matrix[row, col] * vector[col] for col in range(len(vector)))) for row in range(len(vector))]
    )


def _pca(values: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=0)
    feature_count = values.shape[1]
    covariance = np.asarray(
        [
            [float(np.sum(centered[:, row] * centered[:, col])) / max(1, len(values) - 1) for col in range(feature_count)]
            for row in range(feature_count)
        ],
        dtype=np.float64,
    )
    working = covariance.copy()
    components: list[np.ndarray] = []
    eigenvalues: list[float] = []
    for component_index in range(min(dimensions, feature_count)):
        vector = np.asarray([1.0 + ((index + component_index) % feature_count) for index in range(feature_count)])
        vector /= max(float(np.sqrt(np.sum(vector * vector))), 1e-12)
        for _ in range(250):
            candidate = _matvec(working, vector)
            norm = float(np.sqrt(np.sum(candidate * candidate)))
            if norm <= 1e-12:
                break
            candidate /= norm
            if float(np.sqrt(np.sum((candidate - vector) ** 2))) < 1e-10:
                vector = candidate
                break
            vector = candidate
        eigenvalue = float(np.dot(vector, _matvec(working, vector)))
        components.append(vector)
        eigenvalues.append(max(0.0, eigenvalue))
        working -= eigenvalue * np.outer(vector, vector)
    component_matrix = np.asarray(components)
    scores = centered @ component_matrix.T
    total = max(float(np.trace(covariance)), 1e-12)
    explained = np.asarray(eigenvalues) / total
    return scores, component_matrix, explained


def _package_version(distribution: str) -> str:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return "unknown"


def _fit_nonlinear_embedding(
    values: np.ndarray,
    settings: MorphologyClusteringSettings,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit the selected nonlinear reducer and return its effective parameters."""
    method = settings.reduction_method
    actual_seed = settings.random_seed if seed is None else int(seed)
    neighbors = min(settings.umap_n_neighbors, max(2, len(values) - 1))
    common_metadata: dict[str, object] = {
        "method": method,
        "dimensions": settings.embedding_dimensions,
        "requested_neighbors": settings.umap_n_neighbors,
        "effective_neighbors": neighbors,
        "minimum_distance": settings.umap_min_dist,
        "metric": settings.umap_metric,
        "iterations": settings.umap_iterations,
        "random_seed": actual_seed,
    }
    if method == "umap":
        try:
            from torchdr import UMAP as TorchUMAP  # type: ignore[import-not-found]
        except ImportError:
            if not os.environ.get("NUMBA_CACHE_DIR"):
                cache_root = Path(
                    os.environ.get("LOCALAPPDATA")
                    or os.environ.get("XDG_CACHE_HOME")
                    or tempfile.gettempdir()
                )
                numba_cache = cache_root / "Synpo" / "numba-cache"
                try:
                    numba_cache.mkdir(parents=True, exist_ok=True)
                    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
                except OSError:
                    pass
            try:
                import umap  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ValueError(
                    "UMAP is not installed. Update the Synpo environment from "
                    "environment.yml."
                ) from exc
            reducer = umap.UMAP(
                n_neighbors=neighbors,
                n_components=settings.embedding_dimensions,
                min_dist=settings.umap_min_dist,
                metric=settings.umap_metric,
                n_epochs=settings.umap_iterations,
                random_state=actual_seed,
                n_jobs=1,
                low_memory=True,
            )
            embedded = reducer.fit_transform(values)
            common_metadata["implementation"] = "umap-learn"
            common_metadata["implementation_version"] = _package_version("umap-learn")
        else:
            reducer = TorchUMAP(
                n_neighbors=neighbors,
                n_components=settings.embedding_dimensions,
                min_dist=settings.umap_min_dist,
                metric=settings.umap_metric,
                max_iter=settings.umap_iterations,
                random_state=actual_seed,
                device="cpu",
                backend=None,
                verbose=False,
            )
            embedded = reducer.fit_transform(values)
            common_metadata["implementation"] = "torchdr.UMAP"
            common_metadata["implementation_version"] = _package_version("torchdr")
    elif method == "pcumap":
        try:
            from pcc import PCUMAP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                "PCC/PCUMAP is not installed. Update the Synpo environment from "
                "environment.yml or install the 'pccdr' package."
            ) from exc
        reference_points = min(settings.pcumap_reference_points, len(values))
        numpy_state = np.random.get_state()
        np.random.seed(actual_seed)
        try:
            try:
                import torch  # type: ignore[import-not-found]

                torch.manual_seed(actual_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(actual_seed)
            except ImportError:
                pass
            reducer = PCUMAP(
                num_points=reference_points,
                regularization_strength=0.005,
                sampling="random",
                n_components=settings.embedding_dimensions,
                beta=settings.pcumap_beta,
                spearman=False,
                pearson=True,
                epoch_to_start_correlation_loss=settings.pcumap_correlation_start,
                correlation_loss_weight=settings.pcumap_correlation_weight,
                n_neighbors=neighbors,
                min_dist=settings.umap_min_dist,
                metric=settings.umap_metric,
                max_iter=settings.umap_iterations,
                random_state=actual_seed,
                init="pca",
                device=settings.pcumap_device,
                backend=None,
                verbose=False,
            )
            embedded = reducer.fit_transform(values)
        finally:
            np.random.set_state(numpy_state)
        common_metadata.update(
            {
                "implementation": "pccdr.PCUMAP",
                "implementation_version": _package_version("pccdr"),
                "reference_points": reference_points,
                "beta": settings.pcumap_beta,
                "correlation_loss_weight": settings.pcumap_correlation_weight,
                "correlation_loss_start_epoch": settings.pcumap_correlation_start,
                "device": settings.pcumap_device,
            }
        )
    else:
        raise ValueError("A nonlinear embedding can only use UMAP or PCC/PCUMAP.")
    array = np.asarray(embedded, dtype=np.float64)
    expected = (len(values), settings.embedding_dimensions)
    if array.shape != expected or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{method.upper()} returned an invalid embedding with shape {array.shape}; "
            f"expected {expected}."
        )
    return array, common_metadata


def _distance_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    if len(first) < 3:
        return None
    first_distances = pdist(first)
    second_distances = pdist(second)
    if not len(first_distances):
        return None
    value = float(spearmanr(first_distances, second_distances).statistic)
    return value if np.isfinite(value) else None


def _embedding_quality(
    source: np.ndarray,
    embedded: np.ndarray,
    neighbors: int,
) -> dict[str, object]:
    try:
        from sklearn.manifold import trustworthiness
    except ImportError as exc:
        raise ValueError(
            "Advanced embedding diagnostics require scikit-learn. Update the Synpo "
            "environment before running UMAP or PCC/PCUMAP."
        ) from exc
    diagnostic_neighbors = min(neighbors, max(1, (len(source) - 1) // 2))
    indices = np.arange(len(source))
    if len(indices) > 2000:
        indices = np.random.default_rng(42).choice(indices, 2000, replace=False)
    sampled_source = source[indices]
    sampled_embedding = embedded[indices]
    return {
        "trustworthiness": float(
            trustworthiness(
                sampled_source,
                sampled_embedding,
                n_neighbors=min(diagnostic_neighbors, max(1, len(indices) // 2 - 1)),
            )
        ),
        "distance_rank_correlation": _distance_correlation(
            sampled_source, sampled_embedding
        ),
        "diagnostic_sample_size": len(indices),
        "diagnostic_neighbors": diagnostic_neighbors,
    }


def _kmeans(values: np.ndarray, clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    centers = [values[int(rng.integers(len(values)))]]
    while len(centers) < clusters:
        distance = np.min(
            np.asarray([np.sum((values - center) ** 2, axis=1) for center in centers]), axis=0
        )
        total = float(np.sum(distance))
        index = int(rng.choice(len(values), p=distance / total)) if total > 0 else len(centers) % len(values)
        centers.append(values[index])
    center_array = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(len(values), dtype=np.int32)
    for _ in range(300):
        distances = np.asarray([np.sum((values - center) ** 2, axis=1) for center in center_array]).T
        updated_labels = np.argmin(distances, axis=1).astype(np.int32)
        updated = center_array.copy()
        for cluster in range(clusters):
            members = values[updated_labels == cluster]
            if len(members):
                updated[cluster] = np.mean(members, axis=0)
            else:
                updated[cluster] = values[int(np.argmax(np.min(distances, axis=1)))]
        if np.array_equal(updated_labels, labels) and np.max(np.abs(updated - center_array)) < 1e-8:
            labels = updated_labels
            center_array = updated
            break
        labels, center_array = updated_labels, updated
    sse = float(sum(np.sum((values[labels == cluster] - center_array[cluster]) ** 2) for cluster in range(clusters)))
    return labels, center_array, sse


def _gmm(values: np.ndarray, clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    labels, means, _sse = _kmeans(values, clusters, seed)
    count, dimensions = values.shape
    variances = np.tile(np.var(values, axis=0) + 1e-4, (clusters, 1))
    weights = np.asarray([max(1, np.count_nonzero(labels == cluster)) for cluster in range(clusters)], dtype=np.float64)
    weights /= np.sum(weights)
    previous = -np.inf
    responsibilities = np.zeros((count, clusters), dtype=np.float64)
    for _ in range(300):
        log_probability = np.empty((count, clusters), dtype=np.float64)
        for cluster in range(clusters):
            variance = np.maximum(variances[cluster], 1e-6)
            log_probability[:, cluster] = (
                np.log(max(weights[cluster], 1e-12))
                - 0.5 * np.sum(np.log(2.0 * np.pi * variance))
                - 0.5 * np.sum((values - means[cluster]) ** 2 / variance, axis=1)
            )
        normalizer = logsumexp(log_probability, axis=1)
        likelihood = float(np.sum(normalizer))
        responsibilities = np.exp(log_probability - normalizer[:, None])
        effective = np.maximum(np.sum(responsibilities, axis=0), 1e-8)
        weights = effective / count
        means = np.asarray([
            np.sum(responsibilities[:, cluster, None] * values, axis=0) / effective[cluster]
            for cluster in range(clusters)
        ])
        variances = np.asarray([
            np.sum(responsibilities[:, cluster, None] * (values - means[cluster]) ** 2, axis=0) / effective[cluster] + 1e-6
            for cluster in range(clusters)
        ])
        if abs(likelihood - previous) < 1e-6 * (1.0 + abs(likelihood)):
            break
        previous = likelihood
    return np.argmax(responsibilities, axis=1).astype(np.int32), means, responsibilities, previous


def _candidate_labels(values: np.ndarray, settings: MorphologyClusteringSettings) -> tuple[np.ndarray, int, list[dict[str, object]], np.ndarray]:
    candidates = [settings.fixed_cluster_count] if settings.fixed_cluster_count else list(range(settings.minimum_clusters, settings.maximum_clusters + 1))
    evaluated: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for clusters in candidates:
        if clusters > len(values):
            continue
        probabilities = np.ones((len(values), clusters), dtype=np.float64)
        if settings.algorithm == "gaussian_mixture":
            labels, _centers, probabilities, likelihood = _gmm(values, clusters, settings.random_seed + clusters)
            parameters = clusters * (2 * values.shape[1]) + clusters - 1
            criterion = -2.0 * likelihood + parameters * np.log(len(values))
            criterion_name = "BIC"
        else:
            if clusters == 1:
                labels = np.zeros(len(values), dtype=np.int32)
                center = np.mean(values, axis=0)
                sse = float(np.sum((values - center) ** 2))
            elif settings.algorithm == "kmeans":
                labels, _centers, sse = _kmeans(values, clusters, settings.random_seed + clusters)
            else:
                tree = linkage(values, method="ward")
                labels = fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
                sse = 0.0
                for cluster in range(clusters):
                    members = values[labels == cluster]
                    if len(members):
                        sse += float(np.sum((members - np.mean(members, axis=0)) ** 2))
            criterion = len(values) * np.log(max(sse / len(values), 1e-12)) + clusters * values.shape[1] * np.log(len(values))
            criterion_name = "SSE-BIC"
        sizes = [int(np.count_nonzero(labels == cluster)) for cluster in range(clusters)]
        minimum = max(settings.minimum_cluster_spines, int(math.ceil(settings.minimum_cluster_fraction * len(values))))
        accepted = min(sizes, default=0) >= minimum
        silhouette = _silhouette_score(values, labels) if clusters > 1 else None
        within_cluster_sse = float(
            sum(
                np.sum(
                    (members - np.mean(members, axis=0)) ** 2
                )
                for cluster in range(clusters)
                if len(members := values[labels == cluster])
            )
        )
        diagnostic: dict[str, object] = {
            "cluster_count": clusters,
            "criterion": float(criterion),
            "criterion_name": criterion_name,
            "silhouette": silhouette,
            "within_cluster_sse": within_cluster_sse,
            "elbow_distance": None,
            "cluster_sizes": sizes,
            "minimum_size_required": minimum,
            "accepted": accepted,
            "selected": False,
        }
        diagnostics.append(diagnostic)
        if accepted:
            evaluated.append(
                {
                    "criterion": float(criterion),
                    "cluster_count": clusters,
                    "labels": labels,
                    "probabilities": probabilities,
                    "silhouette": silhouette,
                    "within_cluster_sse": within_cluster_sse,
                    "diagnostic": diagnostic,
                }
            )
    if not evaluated:
        raise ValueError("No candidate solution met the configured minimum cluster size.")
    if settings.fixed_cluster_count:
        winner = evaluated[0]
    elif settings.cluster_count_selection == "silhouette":
        eligible = [item for item in evaluated if item["silhouette"] is not None]
        if not eligible:
            raise ValueError(
                "No accepted candidate with at least two clusters is available for silhouette selection."
            )
        winner = max(
            eligible,
            key=lambda item: (
                float(item["silhouette"]),
                -int(item["cluster_count"]),
            ),
        )
    elif settings.cluster_count_selection == "elbow":
        ordered = sorted(evaluated, key=lambda item: int(item["cluster_count"]))
        if len(ordered) < 3:
            raise ValueError(
                "Elbow selection requires at least three accepted candidate cluster counts. "
                "Widen the candidate range or relax the minimum cluster-size settings."
            )
        first_k = int(ordered[0]["cluster_count"])
        last_k = int(ordered[-1]["cluster_count"])
        first_sse = float(ordered[0]["within_cluster_sse"])
        last_sse = float(ordered[-1]["within_cluster_sse"])
        sse_range = first_sse - last_sse
        if last_k == first_k or sse_range <= 1e-12:
            raise ValueError(
                "The within-cluster SSE curve is too flat to identify an elbow."
            )
        for item in ordered:
            normalized_k = (
                int(item["cluster_count"]) - first_k
            ) / (last_k - first_k)
            normalized_sse = (
                float(item["within_cluster_sse"]) - last_sse
            ) / sse_range
            distance = float((1.0 - normalized_k) - normalized_sse)
            item["diagnostic"]["elbow_distance"] = distance  # type: ignore[index]
        winner = max(
            ordered,
            key=lambda item: (
                float(item["diagnostic"]["elbow_distance"]),  # type: ignore[index]
                -int(item["cluster_count"]),
            ),
        )
    else:
        winner = min(evaluated, key=lambda item: float(item["criterion"]))
    winner["diagnostic"]["selected"] = True  # type: ignore[index]
    return (
        np.asarray(winner["labels"], dtype=np.int32),
        int(winner["cluster_count"]),
        diagnostics,
        np.asarray(winner["probabilities"], dtype=np.float64),
    )


def _silhouette_score(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Return an exact silhouette score without adding a scikit-learn dependency."""
    unique = sorted(set(int(value) for value in labels))
    if len(unique) < 2 or len(values) < 3:
        return None
    # Bound quadratic work for large experiments while keeping a deterministic sample.
    indices = np.arange(len(values))
    if len(indices) > 2000:
        indices = np.random.default_rng(42).choice(indices, 2000, replace=False)
    sample = values[indices]
    sample_labels = labels[indices]
    distances = np.sqrt(np.maximum(0.0, np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2)))
    scores: list[float] = []
    for index, cluster in enumerate(sample_labels):
        same = sample_labels == cluster
        same[index] = False
        if not np.any(same):
            scores.append(0.0)
            continue
        within = float(np.mean(distances[index, same]))
        other = [
            float(np.mean(distances[index, sample_labels == candidate]))
            for candidate in unique
            if candidate != int(cluster) and np.any(sample_labels == candidate)
        ]
        nearest = min(other)
        scores.append((nearest - within) / max(nearest, within, 1e-12))
    return float(np.mean(scores))


def _adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    first_values = sorted(set(int(value) for value in first))
    second_values = sorted(set(int(value) for value in second))
    contingency = np.asarray(
        [
            [int(np.count_nonzero((first == left) & (second == right))) for right in second_values]
            for left in first_values
        ],
        dtype=np.int64,
    )
    choose_two = lambda values: np.sum(values * (values - 1) // 2)
    sum_cells = float(choose_two(contingency))
    sum_rows = float(choose_two(np.sum(contingency, axis=1)))
    sum_columns = float(choose_two(np.sum(contingency, axis=0)))
    total_pairs = float(len(first) * (len(first) - 1) // 2)
    if total_pairs <= 0:
        return 1.0
    expected = sum_rows * sum_columns / total_pairs
    maximum = 0.5 * (sum_rows + sum_columns)
    return (sum_cells - expected) / max(maximum - expected, 1e-12)


def _bootstrap_stability(
    values: np.ndarray,
    rows: list[dict[str, object]],
    reference_labels: np.ndarray,
    cluster_count: int,
    algorithm: str,
    seed: int,
    repetitions: int = 12,
) -> dict[str, object]:
    if cluster_count <= 1:
        return {"repetitions": repetitions, "mean_adjusted_rand": 1.0, "median_adjusted_rand": 1.0, "minimum_adjusted_rand": 1.0}
    specimen_keys = sorted({(str(row.get("experimental_group", "")), str(row.get("specimen_id", ""))) for row in rows})
    indices_by_specimen = {
        key: np.asarray([index for index, row in enumerate(rows) if (str(row.get("experimental_group", "")), str(row.get("specimen_id", ""))) == key], dtype=np.int64)
        for key in specimen_keys
    }
    random = np.random.default_rng(seed)
    scores: list[float] = []
    for repetition in range(repetitions):
        sampled_keys = random.choice(len(specimen_keys), len(specimen_keys), replace=True)
        sampled_indices = np.concatenate([indices_by_specimen[specimen_keys[int(index)]] for index in sampled_keys])
        sampled = values[sampled_indices]
        if algorithm == "gaussian_mixture":
            _labels, centers, _probabilities, _likelihood = _gmm(sampled, cluster_count, seed + repetition + 101)
        elif algorithm == "kmeans":
            _labels, centers, _sse = _kmeans(sampled, cluster_count, seed + repetition + 101)
        else:
            sampled_labels = fcluster(linkage(sampled, method="ward"), cluster_count, criterion="maxclust").astype(np.int32) - 1
            centers = np.asarray([np.mean(sampled[sampled_labels == cluster], axis=0) for cluster in range(cluster_count)])
        predicted = np.argmin(
            np.asarray([np.sum((values - center) ** 2, axis=1) for center in centers]).T,
            axis=1,
        ).astype(np.int32)
        scores.append(_adjusted_rand_index(reference_labels, predicted))
    return {
        "repetitions": repetitions,
        "mean_adjusted_rand": float(np.mean(scores)),
        "median_adjusted_rand": float(np.median(scores)),
        "minimum_adjusted_rand": float(np.min(scores)),
    }


def _canonicalize(labels: np.ndarray, rows: list[dict[str, object]], probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    old = sorted(set(int(value) for value in labels))
    old.sort(key=lambda cluster: (
        float(np.median([float(rows[index]["volume_um3"]) for index in np.flatnonzero(labels == cluster)])),
        float(np.median([
            float(rows[index].get("spine_curvilinear_length_um") or rows[index].get("spine_base_to_tip_distance_um") or 0.0)
            for index in np.flatnonzero(labels == cluster)
        ])),
    ))
    mapping = {cluster: index + 1 for index, cluster in enumerate(old)}
    canonical = np.asarray([mapping[int(value)] for value in labels], dtype=np.int32)
    reordered = probabilities[:, old] if probabilities.shape[1] == len(old) else probabilities
    return canonical, reordered, mapping


def _descriptive(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(np.mean(array)) if len(array) else None,
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else (0.0 if len(array) else None),
        "median": float(np.median(array)) if len(array) else None,
        "q1": float(np.percentile(array, 25.0)) if len(array) else None,
        "q3": float(np.percentile(array, 75.0)) if len(array) else None,
        "minimum": float(np.min(array)) if len(array) else None,
        "maximum": float(np.max(array)) if len(array) else None,
    }


def run_morphology_clustering(
    rows: list[dict[str, object]], settings: MorphologyClusteringSettings
) -> dict[str, object]:
    settings.validate()
    included: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    vectors: list[list[float]] = []
    for row in rows:
        reason = ""
        if settings.included_groups and str(row.get("experimental_group", "")) not in settings.included_groups:
            reason = "experimental group not selected for this run"
        elif not bool(row.get("spine_valid", True)):
            reason = "not effectively valid after manual and volume filtering"
        elif settings.reviewed_only and not bool(row.get("geometry_reviewed", False)):
            reason = "geometry review not completed"
        values = [_feature_value(row, feature) for feature in settings.features]
        if not reason and any(value is None for value in values):
            missing = [feature for feature, value in zip(settings.features, values) if value is None]
            reason = "missing selected feature(s): " + ", ".join(missing)
        if reason:
            excluded.append({**row, "clustering_exclusion_reason": reason})
        else:
            included.append(dict(row))
            vectors.append([float(value) for value in values if value is not None])
    minimum_sample = max(50, 10 * len(settings.features))
    if len(included) < minimum_sample:
        raise ValueError(
            f"Clustering requires at least {minimum_sample} complete, effectively valid spines "
            f"for {len(settings.features)} selected feature(s); only {len(included)} are available."
        )
    matrix = np.asarray(vectors, dtype=np.float64)
    scaled, centers, scales = _scale(matrix, settings.scaling)
    pca_scores, pca_components, explained = _pca(
        scaled, min(settings.pca_dimensions, scaled.shape[1])
    )
    embedding: np.ndarray | None = None
    embedding_metadata: dict[str, object] = {
        "method": "pca",
        "dimensions": pca_scores.shape[1],
        "used_for_clustering": bool(settings.use_pca_for_clustering),
    }
    if settings.reduction_method in {"umap", "pcumap"}:
        embedding, embedding_metadata = _fit_nonlinear_embedding(scaled, settings)
        embedding_metadata["used_for_clustering"] = True
        embedding_metadata.update(
            _embedding_quality(
                scaled,
                embedding,
                int(embedding_metadata["effective_neighbors"]),
            )
        )
        clustering_values = embedding
    else:
        clustering_values = pca_scores if settings.use_pca_for_clustering else scaled
    labels, cluster_count, diagnostics, probabilities = _candidate_labels(
        clustering_values, settings
    )
    embedding_stability: dict[str, object] = {
        "requested_repetitions": 1,
        "completed_repetitions": 1,
        "distance_rank_correlations": [],
        "cluster_adjusted_rand_scores": [],
        "mean_distance_rank_correlation": None,
        "mean_cluster_adjusted_rand": None,
    }
    if embedding is not None:
        distance_scores: list[float] = []
        cluster_scores: list[float] = []
        for repetition in range(1, settings.embedding_stability_repetitions):
            repeat_seed = settings.random_seed + repetition
            repeated_embedding, _metadata = _fit_nonlinear_embedding(
                scaled, settings, seed=repeat_seed
            )
            distance_score = _distance_correlation(embedding, repeated_embedding)
            if distance_score is not None:
                distance_scores.append(distance_score)
            repeated_settings = replace(
                settings,
                random_seed=repeat_seed,
                fixed_cluster_count=cluster_count,
            )
            repeated_labels, _count, _diagnostics, _probabilities = _candidate_labels(
                repeated_embedding, repeated_settings
            )
            cluster_scores.append(_adjusted_rand_index(labels, repeated_labels))
        embedding_stability = {
            "requested_repetitions": settings.embedding_stability_repetitions,
            "completed_repetitions": 1 + len(cluster_scores),
            "distance_rank_correlations": distance_scores,
            "cluster_adjusted_rand_scores": cluster_scores,
            "mean_distance_rank_correlation": (
                float(np.mean(distance_scores)) if distance_scores else None
            ),
            "minimum_distance_rank_correlation": (
                float(np.min(distance_scores)) if distance_scores else None
            ),
            "mean_cluster_adjusted_rand": (
                float(np.mean(cluster_scores)) if cluster_scores else None
            ),
            "minimum_cluster_adjusted_rand": (
                float(np.min(cluster_scores)) if cluster_scores else None
            ),
        }
    stability = _bootstrap_stability(
        clustering_values,
        included,
        labels,
        cluster_count,
        settings.algorithm,
        settings.random_seed,
    )
    labels, probabilities, _mapping = _canonicalize(labels, included, probabilities)
    assignments: list[dict[str, object]] = []
    for index, (row, label) in enumerate(zip(included, labels)):
        assignment = {
            **row,
            "morphology_cluster_id": int(label),
            "morphology_cluster_probability": (
                float(np.max(probabilities[index]))
                if settings.algorithm == "gaussian_mixture"
                else None
            ),
        }
        for component in range(pca_scores.shape[1]):
            assignment[f"pca_{component + 1}"] = float(pca_scores[index, component])
        if embedding is not None:
            for component in range(embedding.shape[1]):
                value = float(embedding[index, component])
                assignment[f"embedding_{component + 1}"] = value
                assignment[
                    f"{settings.reduction_method}_{component + 1}"
                ] = value
        for feature_index, feature in enumerate(settings.features):
            assignment[f"clustering_feature_{feature}"] = float(
                vectors[index][feature_index]
            )
        assignments.append(assignment)

    definitions: list[dict[str, object]] = []
    for cluster in range(1, cluster_count + 1):
        members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster]
        definition: dict[str, object] = {
            "morphology_cluster_id": cluster,
            "color": DEFAULT_COLORS[(cluster - 1) % len(DEFAULT_COLORS)],
            "spine_count": len(members),
        }
        for feature in settings.features:
            values = [_feature_value(row, feature) for row in members]
            finite = [float(value) for value in values if value is not None]
            stats = _descriptive(finite)
            for key, value in stats.items():
                definition[f"{feature}_{key}"] = value
        definitions.append(definition)

    protein_rows: list[dict[str, object]] = []
    protein_metrics = (
        "protein_puncta_count",
        "protein_puncta_total_volume_um3",
        "protein_puncta_to_spine_volume_ratio",
        "protein_puncta_mean_volume_um3",
        "protein_puncta_axis_centroid_fraction",
    )
    for cluster in range(1, cluster_count + 1):
        members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster]
        positive = [row for row in members if bool(row.get("has_protein_cluster", False))]
        for subset_name, subset in (("all_spines", members), ("protein_positive", positive)):
            base = {
                "morphology_cluster_id": cluster,
                "subset": subset_name,
                "spine_count": len(subset),
                "protein_positive_spine_count": len(positive),
                "protein_positive_percent": 100.0 * len(positive) / len(members) if members else None,
            }
            for metric in protein_metrics:
                values = [
                    float(row.get(metric) or 0.0)
                    for row in subset
                    if subset_name == "all_spines" or row.get(metric) is not None
                ]
                stats = _descriptive(values)
                for key, value in stats.items():
                    base[f"{metric}_{key}"] = value
            for bin_index in range(1, 11):
                values = [
                    float(row["protein_distribution_in_spine"][bin_index - 1])
                    for row in positive
                    if isinstance(row.get("protein_distribution_in_spine"), list)
                    and len(row["protein_distribution_in_spine"]) >= bin_index
                    and row["protein_distribution_in_spine"][bin_index - 1] is not None
                ]
                base[f"bin_{bin_index:02d}_mean"] = float(np.mean(values)) if values else None
                base[f"bin_{bin_index:02d}_sem"] = (
                    float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else (0.0 if values else None)
                )
                base[f"bin_{bin_index:02d}_n"] = len(values)
            protein_rows.append(base)

    group_rows: list[dict[str, object]] = []
    groups = sorted({str(row.get("experimental_group", "")) for row in assignments})
    for group in groups:
        group_members = [row for row in assignments if str(row.get("experimental_group", "")) == group]
        specimen_ids = sorted({str(row.get("specimen_id", "")) for row in group_members})
        for cluster in range(1, cluster_count + 1):
            pooled = sum(int(row["morphology_cluster_id"]) == cluster for row in group_members)
            specimen_percentages = []
            for specimen_id in specimen_ids:
                specimen_rows = [row for row in group_members if str(row.get("specimen_id", "")) == specimen_id]
                specimen_percentages.append(100.0 * sum(int(row["morphology_cluster_id"]) == cluster for row in specimen_rows) / len(specimen_rows))
            sd = float(np.std(specimen_percentages, ddof=1)) if len(specimen_percentages) > 1 else 0.0
            group_rows.append({
                "experimental_group": group,
                "morphology_cluster_id": cluster,
                "pooled_spine_count": pooled,
                "pooled_percent": 100.0 * pooled / len(group_members) if group_members else None,
                "specimen_percentage_mean": float(np.mean(specimen_percentages)),
                "specimen_percentage_sd": sd,
                "specimen_percentage_sem": sd / np.sqrt(len(specimen_percentages)),
                "n_specimens": len(specimen_percentages),
            })

    loadings = [
        {
            "feature": feature,
            **{f"pca_{index + 1}_loading": float(pca_components[index, feature_index]) for index in range(len(pca_components))},
            **{f"pca_{index + 1}_explained_fraction": float(explained[index]) for index in range(len(explained))},
        }
        for feature_index, feature in enumerate(settings.features)
    ]
    input_signature = hashlib.sha256(
        json.dumps(
            [
                [row.get("experimental_group"), row.get("specimen_id"), row.get("spine_id"), *[_feature_value(row, feature) for feature in settings.features]]
                for row in included
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm_version": 2,
        "created_at": time.time(),
        "settings": settings.to_dict(),
        "input_signature": input_signature,
        "selected_cluster_count": cluster_count,
        "included_spine_count": len(included),
        "excluded_spine_count": len(excluded),
        "feature_centers": [float(value) for value in centers],
        "feature_scales": [float(value) for value in scales],
        "pca_feature_means": [float(value) for value in np.mean(scaled, axis=0)],
        "assignments": assignments,
        "cluster_definitions": definitions,
        "group_summary": group_rows,
        "protein_summary": protein_rows,
        "excluded_audit": excluded,
        "candidate_diagnostics": diagnostics,
        "bootstrap_stability": stability,
        "embedding_metadata": embedding_metadata,
        "embedding_stability": embedding_stability,
        "pca_loadings": loadings,
        "protein_metrics_used_for_clustering": False,
    }


def save_named_morphology_run(
    manifest: dict[str, object],
    project_path: str | Path,
    name: str,
    settings: MorphologyClusteringSettings,
) -> dict[str, object]:
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("Enter a name for this morphology analysis run.")
    result = run_morphology_clustering(collect_morphology_rows(manifest), settings)
    analysis = manifest.setdefault("morphology_analysis", {"runs": [], "active_run_id": None})
    existing = next((item for item in analysis.setdefault("runs", []) if item.get("name") == clean_name), None)
    run_id = str(existing.get("run_id")) if existing else uuid.uuid4().hex
    previous_style = dict(existing.get("plot_style", {})) if existing else {}
    result.update(
        {
            "run_id": run_id,
            "name": clean_name,
            "plot_style": {**DEFAULT_PLOT_STYLE, **previous_style},
        }
    )
    path = morphology_run_path(manifest, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(result, stream, separators=(",", ":"))
    os.replace(temporary, path)
    record = {
        "run_id": run_id,
        "name": clean_name,
        "created_at": result["created_at"],
        "input_signature": result["input_signature"],
        "selected_cluster_count": result["selected_cluster_count"],
        "settings": settings.to_dict(),
        "colors": [item["color"] for item in result["cluster_definitions"]],
        "plot_style": dict(result["plot_style"]),
    }
    if existing:
        existing.clear()
        existing.update(record)
    else:
        analysis["runs"].append(record)
    analysis["active_run_id"] = run_id
    save_project(project_path, manifest)
    return result


def load_morphology_run(manifest: dict[str, object], run_id: str) -> dict[str, object]:
    path = morphology_run_path(manifest, run_id)
    if not path.is_file():
        raise ValueError("The saved morphology analysis data are unavailable.")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def update_run_colors(
    manifest: dict[str, object], project_path: str | Path, run_id: str, colors: list[str]
) -> dict[str, object]:
    result = load_morphology_run(manifest, run_id)
    for index, definition in enumerate(result.get("cluster_definitions", [])):
        if index < len(colors):
            definition["color"] = str(colors[index])
    path = morphology_run_path(manifest, run_id)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(result, stream, separators=(",", ":"))
    os.replace(temporary, path)
    for record in manifest.get("morphology_analysis", {}).get("runs", []):
        if str(record.get("run_id")) == run_id:
            record["colors"] = list(colors)
    save_project(project_path, manifest)
    return result


def update_run_plot_style(
    manifest: dict[str, object],
    project_path: str | Path,
    run_id: str,
    plot_style: dict[str, object],
) -> dict[str, object]:
    result = load_morphology_run(manifest, run_id)
    axes_alpha = min(1.0, max(0.0, float(plot_style.get("axes_alpha", 1.0))))
    background_alpha = min(
        1.0, max(0.0, float(plot_style.get("background_alpha", 1.0)))
    )
    result["plot_style"] = {
        "axes_color": str(plot_style.get("axes_color", "#202020")),
        "axes_alpha": axes_alpha,
        "background_color": str(plot_style.get("background_color", "#ffffff")),
        "background_alpha": background_alpha,
        "show_legend": bool(plot_style.get("show_legend", True)),
        "legend_position": str(
            plot_style.get("legend_position", "outside_right")
        ),
        "custom_x_feature": str(plot_style.get("custom_x_feature", "")),
        "custom_y_feature": str(plot_style.get("custom_y_feature", "")),
        "pca_show_points": bool(plot_style.get("pca_show_points", True)),
        "pca_point_alpha": min(
            1.0, max(0.0, float(plot_style.get("pca_point_alpha", 0.68)))
        ),
        "correlation_features": [
            str(feature)
            for feature in plot_style.get("correlation_features", DEFAULT_FEATURES)
            if str(feature) in MORPHOLOGY_FEATURES
        ],
        "correlation_scope": str(
            plot_style.get("correlation_scope", "__pooled__")
        ),
        "correlation_threshold": min(
            1.0,
            max(0.0, float(plot_style.get("correlation_threshold", 0.80))),
        ),
        "correlation_colormap": str(
            plot_style.get("correlation_colormap", "coolwarm")
        ),
        "correlation_negative_color": str(
            plot_style.get("correlation_negative_color", "#2166ac")
        ),
        "correlation_zero_color": str(
            plot_style.get("correlation_zero_color", "#f7f7f7")
        ),
        "correlation_positive_color": str(
            plot_style.get("correlation_positive_color", "#b2182b")
        ),
        "correlation_alpha": min(
            1.0, max(0.0, float(plot_style.get("correlation_alpha", 1.0)))
        ),
        "correlation_reorder": bool(
            plot_style.get("correlation_reorder", False)
        ),
        "correlation_export_group_matrices": bool(
            plot_style.get("correlation_export_group_matrices", False)
        ),
    }
    path = morphology_run_path(manifest, run_id)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(result, stream, separators=(",", ":"))
    os.replace(temporary, path)
    for record in manifest.get("morphology_analysis", {}).get("runs", []):
        if str(record.get("run_id")) == run_id:
            record["plot_style"] = dict(result["plot_style"])
    save_project(project_path, manifest)
    return result


def _columns(rows: list[dict[str, object]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        for key in row:
            if key not in output:
                output.append(key)
    return output


def _cell(value: object) -> object:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, tuple, dict)) else value


def export_morphology_analysis(
    result: dict[str, object], workbook_path: str | Path
) -> dict[str, object]:
    from openpyxl import Workbook, load_workbook
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import to_rgba

    path = Path(workbook_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    reduction_method = str(result.get("settings", {}).get("reduction_method", "pca"))
    embedding_dimensions = int(
        result.get("settings", {}).get("embedding_dimensions", 0)
        if reduction_method in {"umap", "pcumap"}
        else 0
    )
    settings_rows = [
        {
            "section": "clustering",
            **dict(result["settings"]),
            "protein_metrics_used_for_clustering": False,
        },
        {"section": "run", "name": result.get("name", ""), "run_id": result.get("run_id", ""), "input_signature": result.get("input_signature", ""), "selected_cluster_count": result.get("selected_cluster_count")},
        {"section": "plot_style", **{**DEFAULT_PLOT_STYLE, **dict(result.get("plot_style", {}))}},
    ]
    plot_rows = [
        {
            "experimental_group": row.get("experimental_group"),
            "specimen_id": row.get("specimen_id"),
            "spine_id": row.get("spine_id"),
            "morphology_cluster_id": row.get("morphology_cluster_id"),
            "cluster_color": next(
                (item.get("color") for item in result.get("cluster_definitions", []) if item.get("morphology_cluster_id") == row.get("morphology_cluster_id")),
                None,
            ),
            "volume_um3": row.get("volume_um3"),
            "spine_curvilinear_length_um": row.get("spine_curvilinear_length_um"),
            "pca_1": row.get("pca_1"),
            "pca_2": row.get("pca_2"),
            "pca_3": row.get("pca_3"),
            **{
                f"embedding_{index}": row.get(f"embedding_{index}")
                for index in range(1, embedding_dimensions + 1)
            },
            "has_protein_cluster": row.get("has_protein_cluster"),
            "protein_puncta_count": row.get("protein_puncta_count"),
            "protein_puncta_total_volume_um3": row.get("protein_puncta_total_volume_um3"),
            "protein_puncta_axis_centroid_fraction": row.get("protein_puncta_axis_centroid_fraction"),
            **{
                f"clustering_feature_{feature}": row.get(
                    f"clustering_feature_{feature}"
                )
                for feature in result.get("settings", {}).get("features", [])
            },
        }
        for row in result.get("assignments", [])
    ]
    interpretation = pca_interpretation_data(result)
    equation_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "loadings"
        }
        for row in interpretation.get("components", [])
    ]
    correlation_style = {
        **DEFAULT_PLOT_STYLE,
        **dict(result.get("plot_style", {})),
    }
    correlation_features = [
        str(feature)
        for feature in correlation_style.get(
            "correlation_features", DEFAULT_FEATURES
        )
        if str(feature) in MORPHOLOGY_FEATURES
    ]
    if len(correlation_features) < 2:
        correlation_features = list(DEFAULT_FEATURES)
    correlation_scope = str(
        correlation_style.get("correlation_scope", "__pooled__")
    )
    correlation_group = None if correlation_scope == "__pooled__" else correlation_scope
    correlation_threshold = float(
        correlation_style.get("correlation_threshold", 0.80)
    )
    correlation_error = ""
    correlation_data: dict[str, object] | None = None
    try:
        correlation_data = feature_correlation_data(
            result,
            correlation_features,
            experimental_group=correlation_group,
            threshold=correlation_threshold,
            reorder=bool(correlation_style.get("correlation_reorder", False)),
        )
    except ValueError as exc:
        correlation_error = str(exc)

    def correlation_matrix_rows(
        data: dict[str, object], group: str | None
    ) -> list[dict[str, object]]:
        feature_keys = list(data["features"])
        feature_labels = list(data["feature_labels"])
        matrix_rows: list[dict[str, object]] = []
        for row_index, feature in enumerate(feature_keys):
            matrix_rows.append(
                {
                    "experimental_group": group if group is not None else "__pooled__",
                    "feature": feature,
                    "feature_label": feature_labels[row_index],
                    **{
                        column_feature: data["matrix"][row_index][column_index]
                        for column_index, column_feature in enumerate(feature_keys)
                    },
                }
            )
        return matrix_rows

    correlation_matrix_table = (
        correlation_matrix_rows(correlation_data, correlation_group)
        if correlation_data is not None
        else []
    )
    correlation_pair_table = (
        [
            {
                "experimental_group": (
                    correlation_group
                    if correlation_group is not None
                    else "__pooled__"
                ),
                "included_spine_count": correlation_data["included_spine_count"],
                **dict(row),
            }
            for row in correlation_data["pairs"]
        ]
        if correlation_data is not None
        else []
    )
    correlation_group_table: list[dict[str, object]] = []
    correlation_group_data: list[tuple[str, dict[str, object]]] = []
    if bool(correlation_style.get("correlation_export_group_matrices", False)):
        for group in sorted(
            {str(row.get("experimental_group", "")) for row in result.get("assignments", [])}
        ):
            try:
                group_data = feature_correlation_data(
                    result,
                    correlation_features,
                    experimental_group=group,
                    threshold=correlation_threshold,
                    reorder=bool(correlation_style.get("correlation_reorder", False)),
                )
            except ValueError:
                continue
            correlation_group_data.append((group, group_data))
            correlation_group_table.extend(
                correlation_matrix_rows(group_data, group)
            )
    tables = {
        "Spine_Morphology": list(result.get("assignments", [])),
        "Spine_Cluster_Assignments": [
            {
                key: row.get(key)
                for key in (
                    "experimental_group", "specimen_id", "roi_id", "dendrite_id",
                    "spine_id", "morphology_cluster_id",
                    "morphology_cluster_probability", "pca_1", "pca_2", "pca_3",
                    *tuple(
                        f"embedding_{index}"
                        for index in range(1, embedding_dimensions + 1)
                    ),
                )
                if key in row
            }
            for row in result.get("assignments", [])
        ],
        "Morphology_Cluster_Definitions": list(result.get("cluster_definitions", [])),
        "Morphology_Cluster_Groups": list(result.get("group_summary", [])),
        "Morphology_Cluster_Protein": list(result.get("protein_summary", [])),
        "Morphology_Cluster_Audit": list(result.get("excluded_audit", [])),
        "Morphology_Cluster_Settings": settings_rows,
        "PCA_Loadings": list(result.get("pca_loadings", [])),
        "PCA_Equations": equation_rows,
        "PCA_Feature_Preparation": list(
            interpretation.get("feature_rows", [])
        ),
        "Candidate_Models": list(result.get("candidate_diagnostics", [])),
        "Bootstrap_Stability": [dict(result.get("bootstrap_stability", {}))],
        "Embedding_Diagnostics": [dict(result.get("embedding_metadata", {}))],
        "Embedding_Stability": [dict(result.get("embedding_stability", {}))],
        "Plot_Data": plot_rows,
        "Feature_Correlation": correlation_matrix_table,
        "Feature_Corr_Pairs": correlation_pair_table,
        "Feature_Corr_Settings": [
            {
                "features": correlation_features,
                "scope": correlation_scope,
                "pearson_threshold": correlation_threshold,
                "listwise_complete": True,
                "included_spine_count": (
                    correlation_data.get("included_spine_count")
                    if correlation_data is not None
                    else 0
                ),
                "missing_spine_count": (
                    correlation_data.get("missing_spine_count")
                    if correlation_data is not None
                    else None
                ),
                "reordered_by_absolute_correlation": bool(
                    correlation_style.get("correlation_reorder", False)
                ),
                "export_group_matrices": bool(
                    correlation_style.get("correlation_export_group_matrices", False)
                ),
                "error": correlation_error,
            }
        ],
        "Feature_Corr_By_Group": correlation_group_table,
    }
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows in tables.items():
        sheet = workbook.create_sheet(name[:31])
        columns = _columns(rows)
        if not columns:
            sheet.append(["No rows"])
            continue
        sheet.append(columns)
        for row in rows:
            sheet.append([_cell(row.get(column)) for column in columns])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)
    load_workbook(path, read_only=True).close()
    csv_directory = path.parent / f"{path.stem}_csv"
    csv_directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        columns = _columns(rows)
        with (csv_directory / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            if columns:
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: _cell(row.get(key)) for key in columns})

    plot_directory = path.parent / f"{path.stem}_plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    assignments = list(result.get("assignments", []))
    definitions = list(result.get("cluster_definitions", []))
    colors = {int(row["morphology_cluster_id"]): str(row["color"]) for row in definitions}
    groups = sorted({str(row.get("experimental_group", "")) for row in assignments})
    marker_values = ("o", "^", "s", "D", "P", "X", "v", "<", ">", "*")
    marker_by_group = {
        group: marker_values[index % len(marker_values)]
        for index, group in enumerate(groups)
    }
    figures: list[tuple[str, object]] = []
    figure, axis = plt.subplots(figsize=(8, 6))
    for cluster in sorted(colors):
        for group in groups:
            members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster and str(row.get("experimental_group", "")) == group and row.get("volume_um3") is not None and row.get("spine_curvilinear_length_um") is not None]
            if members:
                axis.scatter([float(row["volume_um3"]) for row in members], [float(row["spine_curvilinear_length_um"]) for row in members], s=18, alpha=0.7, color=colors[cluster], marker=marker_by_group[group], label=f"Cluster {cluster} · {group}")
    axis.set_xscale("log")
    axis.set(xlabel="Spine volume (µm³, log scale)", ylabel="Curvilinear length (µm)", title="Spine volume versus length")
    axis.legend()
    axis.grid(alpha=0.2)
    figures.append(("volume_vs_length", figure))
    figure, axis = plt.subplots(figsize=(8, 6))
    for cluster in sorted(colors):
        for group in groups:
            members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster and str(row.get("experimental_group", "")) == group and row.get("volume_um3") is not None and row.get("spine_base_to_tip_distance_um") is not None]
            if members:
                axis.scatter([float(row["volume_um3"]) for row in members], [float(row["spine_base_to_tip_distance_um"]) for row in members], s=18, alpha=0.7, color=colors[cluster], marker=marker_by_group[group], label=f"Cluster {cluster} · {group}")
    axis.set_xscale("log")
    axis.set(xlabel="Spine volume (µm³, log scale)", ylabel="Base-to-tip distance (µm)", title="Spine volume versus base-to-tip distance")
    axis.legend()
    axis.grid(alpha=0.2)
    figures.append(("volume_vs_base_to_tip_distance", figure))
    dimensions = int(result["settings"].get("pca_dimensions", 2))
    if dimensions == 3 and assignments and "pca_3" in assignments[0]:
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")
        for cluster in sorted(colors):
            for group in groups:
                members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster and str(row.get("experimental_group", "")) == group]
                if members:
                    axis.scatter([row["pca_1"] for row in members], [row["pca_2"] for row in members], [row["pca_3"] for row in members], color=colors[cluster], marker=marker_by_group[group], label=f"Cluster {cluster} · {group}")
        axis.set(xlabel="PCA 1", ylabel="PCA 2", zlabel="PCA 3", title="3D PCA morphology plot")
    else:
        figure, axis = plt.subplots(figsize=(8, 6))
        for cluster in sorted(colors):
            for group in groups:
                members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster and str(row.get("experimental_group", "")) == group]
                if members:
                    axis.scatter([row.get("pca_1") for row in members], [row.get("pca_2") for row in members], color=colors[cluster], marker=marker_by_group[group], label=f"Cluster {cluster} · {group}")
        axis.set(xlabel="PCA 1", ylabel="PCA 2", title="2D PCA morphology plot")
    axis.legend()
    figures.append((f"pca_{dimensions}d", figure))

    if reduction_method in {"umap", "pcumap"} and assignments:
        plot_dimensions = int(
            result.get("settings", {}).get("embedding_plot_dimensions", 2)
        )
        plot_dimensions = 3 if plot_dimensions == 3 and embedding_dimensions >= 3 else 2
        if plot_dimensions == 3:
            figure = plt.figure(figsize=(8, 7))
            axis = figure.add_subplot(111, projection="3d")
        else:
            figure, axis = plt.subplots(figsize=(8, 6))
        for cluster in sorted(colors):
            for group in groups:
                members = [
                    row
                    for row in assignments
                    if int(row["morphology_cluster_id"]) == cluster
                    and str(row.get("experimental_group", "")) == group
                ]
                if not members:
                    continue
                arguments = (
                    [row.get("embedding_1") for row in members],
                    [row.get("embedding_2") for row in members],
                )
                if plot_dimensions == 3:
                    arguments += ([row.get("embedding_3") for row in members],)
                axis.scatter(
                    *arguments,
                    s=18,
                    alpha=0.72,
                    color=colors[cluster],
                    marker=marker_by_group[group],
                    label=f"Cluster {cluster} В· {group}",
                )
        method_label = "PCC/PCUMAP" if reduction_method == "pcumap" else "UMAP"
        axis.set_xlabel(f"{method_label} 1")
        axis.set_ylabel(f"{method_label} 2")
        if plot_dimensions == 3:
            axis.set_zlabel(f"{method_label} 3")
        axis.set_title(
            f"{plot_dimensions}D view of the {embedding_dimensions}D {method_label} clustering space"
        )
        axis.legend()
        figures.append((f"{reduction_method}_{plot_dimensions}d_embedding", figure))

    figure = plt.figure(figsize=(13, 9))
    draw_pca_interpretation(figure, result)
    figures.append(("pca_interpretation", figure))
    if dimensions == 3:
        figure = plt.figure(figsize=(10, 8))
        draw_pca_3d_feature_axes(figure, result)
        figures.append(("pca_3d_feature_axes", figure))

    if correlation_data is not None:
        figure = plt.figure(
            figsize=(
                max(8.0, len(correlation_features) * 0.85),
                max(7.0, len(correlation_features) * 0.72),
            )
        )
        draw_feature_correlation(
            figure,
            result,
            features=correlation_features,
            experimental_group=correlation_group,
            threshold=correlation_threshold,
            reorder=bool(correlation_style.get("correlation_reorder", False)),
            colormap=str(correlation_style.get("correlation_colormap", "coolwarm")),
            negative_color=str(correlation_style.get("correlation_negative_color", "#2166ac")),
            zero_color=str(correlation_style.get("correlation_zero_color", "#f7f7f7")),
            positive_color=str(correlation_style.get("correlation_positive_color", "#b2182b")),
            alpha=float(correlation_style.get("correlation_alpha", 1.0)),
        )
        figures.append(("feature_correlation_matrix", figure))
    for group, _group_data in correlation_group_data:
        figure = plt.figure(
            figsize=(
                max(8.0, len(correlation_features) * 0.85),
                max(7.0, len(correlation_features) * 0.72),
            )
        )
        draw_feature_correlation(
            figure,
            result,
            features=correlation_features,
            experimental_group=group,
            threshold=correlation_threshold,
            reorder=bool(correlation_style.get("correlation_reorder", False)),
            colormap=str(correlation_style.get("correlation_colormap", "coolwarm")),
            negative_color=str(correlation_style.get("correlation_negative_color", "#2166ac")),
            zero_color=str(correlation_style.get("correlation_zero_color", "#f7f7f7")),
            positive_color=str(correlation_style.get("correlation_positive_color", "#b2182b")),
            alpha=float(correlation_style.get("correlation_alpha", 1.0)),
        )
        safe_group = "".join(
            character if character.isalnum() else "_" for character in group
        ).strip("_") or "blank"
        figures.append((f"feature_correlation_{safe_group[:48]}", figure))

    cluster_ids = sorted(colors)
    figure, axis = plt.subplots(figsize=(8, 6))
    positive = []
    for cluster in cluster_ids:
        members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster]
        positive.append(100.0 * sum(bool(row.get("has_protein_cluster")) for row in members) / len(members) if members else 0.0)
    axis.bar([str(value) for value in cluster_ids], positive, color=[colors[value] for value in cluster_ids])
    axis.set(xlabel="Morphology cluster", ylabel="Protein-positive spines (%)", title="Protein-positive fraction by morphology cluster")
    axis.grid(axis="y", alpha=0.2)
    figures.append(("protein_positive_fraction", figure))

    figure, axis = plt.subplots(figsize=(8, 6))
    draw_protein_puncta_volume(axis, result)
    axis.grid(axis="y", alpha=0.2)
    figures.append(("protein_puncta_volume", figure))

    figure, axis = plt.subplots(figsize=(9, 6))
    protein_summary = list(result.get("protein_summary", []))
    x_values = np.arange(1, 11)
    for cluster in cluster_ids:
        row = next((item for item in protein_summary if int(item.get("morphology_cluster_id", 0)) == cluster and item.get("subset") == "protein_positive"), None)
        if row:
            means = [row.get(f"bin_{index:02d}_mean") for index in x_values]
            if any(value is not None for value in means):
                axis.plot(x_values, [np.nan if value is None else float(value) for value in means], marker="o", color=colors[cluster], label=f"Cluster {cluster}")
    axis.set(xlabel="Normalized shaft-to-tip bin", ylabel="Mean protein distribution", title="Protein position profiles by morphology cluster")
    axis.set_xticks(x_values)
    axis.legend()
    axis.grid(alpha=0.2)
    figures.append(("protein_position_profiles", figure))

    figure, axis = plt.subplots(figsize=(9, 6))
    group_rows = list(result.get("group_summary", []))
    groups = sorted({str(row.get("experimental_group", "")) for row in group_rows})
    positions = np.arange(len(groups), dtype=float)
    width = 0.8 / max(1, len(cluster_ids))
    for offset, cluster in enumerate(cluster_ids):
        values = [
            next((float(row.get("specimen_percentage_mean") or 0.0) for row in group_rows if str(row.get("experimental_group", "")) == group and int(row.get("morphology_cluster_id", 0)) == cluster), 0.0)
            for group in groups
        ]
        axis.bar(positions + (offset - (len(cluster_ids) - 1) / 2.0) * width, values, width=width, color=colors[cluster], label=f"Cluster {cluster}")
    axis.set_xticks(positions, groups, rotation=25, ha="right")
    axis.set(xlabel="Experimental group", ylabel="Mean specimen cluster proportion (%)", title="Morphology cluster proportions by group")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figures.append(("group_cluster_proportions", figure))

    selected_features = list(result.get("settings", {}).get("features", []))
    pending_plot_style = {
        **DEFAULT_PLOT_STYLE,
        **dict(result.get("plot_style", {})),
    }
    custom_x = str(pending_plot_style.get("custom_x_feature", ""))
    custom_y = str(pending_plot_style.get("custom_y_feature", ""))
    if custom_x not in selected_features:
        custom_x = selected_features[0] if selected_features else ""
    if custom_y not in selected_features:
        custom_y = (
            selected_features[1]
            if len(selected_features) > 1
            else custom_x
        )
    if custom_x and custom_y:
        figure, axis = plt.subplots(figsize=(8, 6))
        for cluster in sorted(colors):
            for group in groups:
                points = []
                for row in assignments:
                    if (
                        int(row["morphology_cluster_id"]) != cluster
                        or str(row.get("experimental_group", "")) != group
                    ):
                        continue
                    x_value = morphology_feature_value(row, custom_x)
                    y_value = morphology_feature_value(row, custom_y)
                    if x_value is not None and y_value is not None:
                        points.append((x_value, y_value))
                if points:
                    axis.scatter(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        s=18,
                        alpha=0.72,
                        color=colors[cluster],
                        marker=marker_by_group[group],
                        label=f"Cluster {cluster} · {group}",
                    )
        axis.set(
            xlabel=MORPHOLOGY_FEATURES[custom_x][0],
            ylabel=MORPHOLOGY_FEATURES[custom_y][0],
            title=(
                f"{MORPHOLOGY_FEATURES[custom_y][0]} versus "
                f"{MORPHOLOGY_FEATURES[custom_x][0]}"
            ),
        )
        axis.legend()
        axis.grid(alpha=0.2)
        figures.append((f"morphology_{custom_x}_vs_{custom_y}", figure))
    if selected_features and definitions:
        matrix = np.asarray(
            [
                [float(definition.get(f"{feature}_median") or 0.0) for feature in selected_features]
                for definition in definitions
            ],
            dtype=np.float64,
        )
        center = np.mean(matrix, axis=0)
        spread = np.std(matrix, axis=0)
        standardized = (matrix - center) / np.where(spread > 1e-12, spread, 1.0)
        figure, axis = plt.subplots(figsize=(max(8, len(selected_features) * 1.3), max(4, len(definitions) * 0.65)))
        image = axis.imshow(standardized, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        axis.set_xticks(np.arange(len(selected_features)), [MORPHOLOGY_FEATURES[value][0] for value in selected_features], rotation=35, ha="right")
        axis.set_yticks(np.arange(len(definitions)), [f"Cluster {row['morphology_cluster_id']}" for row in definitions])
        axis.set_title("Standardized morphology-cluster median profiles")
        figure.colorbar(image, ax=axis, label="Across-cluster standardized median")
        figures.append(("cluster_profile_heatmap", figure))
    plot_style = {**DEFAULT_PLOT_STYLE, **dict(result.get("plot_style", {}))}
    axes_rgba = to_rgba(
        str(plot_style["axes_color"]), alpha=float(plot_style["axes_alpha"])
    )
    background_rgba = to_rgba(
        str(plot_style["background_color"]),
        alpha=float(plot_style["background_alpha"]),
    )
    for _name, figure in figures:
        if _name in {"pca_interpretation", "pca_3d_feature_axes"}:
            continue
        figure.patch.set_facecolor(background_rgba)
        for styled_axis in figure.axes:
            styled_axis.set_facecolor(background_rgba)
            styled_axis.tick_params(colors=axes_rgba)
            styled_axis.xaxis.label.set_color(axes_rgba)
            styled_axis.yaxis.label.set_color(axes_rgba)
            styled_axis.title.set_color(axes_rgba)
            if hasattr(styled_axis, "zaxis"):
                styled_axis.zaxis.label.set_color(axes_rgba)
            for spine in styled_axis.spines.values():
                spine.set_color(axes_rgba)
            handles, labels = styled_axis.get_legend_handles_labels()
            existing_legend = styled_axis.get_legend()
            if existing_legend is not None:
                existing_legend.remove()
            legend = None
            if bool(plot_style.get("show_legend", True)) and handles:
                legend_position = str(
                    plot_style.get("legend_position", "outside_right")
                )
                if legend_position == "outside_bottom":
                    legend = styled_axis.legend(
                        handles,
                        labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.16),
                        ncols=min(3, len(handles)),
                        borderaxespad=0.0,
                    )
                elif legend_position == "inside":
                    legend = styled_axis.legend(
                        handles, labels, loc="upper right"
                    )
                else:
                    legend = styled_axis.legend(
                        handles,
                        labels,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0,
                    )
            if legend is not None:
                legend.get_frame().set_facecolor(background_rgba)
                for legend_text in legend.get_texts():
                    legend_text.set_color(axes_rgba)
    pdf_path = path.with_name(f"{path.stem}_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for name, figure in figures:
            figure.savefig(plot_directory / f"{name}.png", dpi=600, bbox_inches="tight", facecolor=background_rgba)
            figure.savefig(plot_directory / f"{name}.svg", bbox_inches="tight", facecolor=background_rgba)
            figure.savefig(plot_directory / f"{name}.pdf", bbox_inches="tight", facecolor=background_rgba)
            pdf.savefig(figure, bbox_inches="tight", facecolor=background_rgba)
            plt.close(figure)
    return {"workbook": str(path), "csv_directory": str(csv_directory), "plot_directory": str(plot_directory), "report_pdf": str(pdf_path), "verified": True}


def load_morphology_rows_from_workbook(path: str | Path) -> list[dict[str, object]]:
    """Load all-spine morphology from a Synpo analysis or measurement workbook."""
    from openpyxl import load_workbook

    source = Path(path).resolve()
    workbook = load_workbook(source, read_only=True, data_only=True)
    try:
        sheet_name = "Spine_Morphology" if "Spine_Morphology" in workbook.sheetnames else "Spine_Master"
        if sheet_name not in workbook.sheetnames:
            raise ValueError("The workbook has neither Spine_Morphology nor Spine_Master.")
        values = workbook[sheet_name].iter_rows(values_only=True)
        headers = next(values, ())
        if not headers or headers[0] == "No rows":
            return []
        names = [str(value or "") for value in headers]
        rows: list[dict[str, object]] = []
        for values_row in values:
            if not any(value is not None for value in values_row):
                continue
            row = {name: value for name, value in zip(names, values_row) if name}
            for key in ("protein_distribution_in_spine", "centerline_base_zyx", "centerline_tip_zyx"):
                value = row.get(key)
                if isinstance(value, str) and value[:1] in {"[", "{"}:
                    try:
                        row[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            rows.append(row)
        return rows
    finally:
        workbook.close()


def run_morphology_clustering_from_workbook(
    source_path: str | Path,
    output_path: str | Path,
    settings: MorphologyClusteringSettings,
    *,
    run_name: str = "Standalone morphology analysis",
) -> dict[str, object]:
    rows = load_morphology_rows_from_workbook(source_path)
    result = run_morphology_clustering(rows, settings)
    result.update(
        {
            "run_id": uuid.uuid4().hex,
            "name": run_name,
            "source_workbook": str(Path(source_path).resolve()),
            "plot_style": dict(DEFAULT_PLOT_STYLE),
        }
    )
    exported = export_morphology_analysis(result, output_path)
    return {"result": result, "export": exported}
