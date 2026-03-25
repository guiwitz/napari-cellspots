"""
Post-processing helpers for population-level analysis.

Functions here are designed to be called directly from Jupyter notebooks:
  - load_all_spots       – recursively collect *_spots.csv files into one DataFrame
  - apply_angle_correction – subtract major_axis_angle per nucleus from theta
  - filter_by_spot_count – keep only nuclei with enough spots
  - load_all_statistics  – recursively collect *_statistics.csv files
  - plot_polar_and_anisotropy – combined polar-rose + anisotropy-histogram figure
  - run_statistics_for_folder – convenience wrapper around compute_statistics_per_image
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_spots(root: Path | str) -> tuple[pd.DataFrame, list[Path]]:
    """Recursively load all ``*_spots.csv`` files under *root*.

    Returns
    -------
    all_spots : pd.DataFrame
        Combined DataFrame with extra columns ``source_file`` and ``source_path``.
    spots_files : list[Path]
        Sorted list of discovered CSV paths (needed for angle correction).
    """
    root = Path(root)
    spots_files = sorted(root.rglob("*_spots.csv"))
    print(f"Found {len(spots_files)} spots file(s)")

    frames = []
    for sf in spots_files:
        df = pd.read_csv(sf)
        #df["source_file"] = sf.stem.replace("_spots", "")
        #df["source_path"] = str(sf.parent)
        frames.append(df)

    all_spots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"Total spots loaded: {len(all_spots)}")
    return all_spots, spots_files


def load_all_statistics(root: Path | str) -> pd.DataFrame:
    """Recursively load all ``*_statistics.csv`` files under *root*.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with an extra ``source_file`` column.
    """
    root = Path(root)
    stats_files = sorted(root.rglob("*_statistics.csv"))
    print(f"Found {len(stats_files)} statistics file(s)")

    frames = []
    for sf in stats_files:
        df = pd.read_csv(sf)
        df["source_file"] = sf.stem.replace("_statistics", "")
        frames.append(df)

    all_stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_stats


# ---------------------------------------------------------------------------
# Angle correction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_by_spot_count(
    all_spots: pd.DataFrame,
    min_spots: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only nuclei that have at least *min_spots* spots.

    Parameters
    ----------
    all_spots : pd.DataFrame
        DataFrame with ``theta_corrected``, ``source_file``, and ``nuclei_index``.
    min_spots : int
        Minimum number of spots required to retain a nucleus.

    Returns
    -------
    filtered_spots : pd.DataFrame
    valid_nuclei : pd.DataFrame
        One row per accepted nucleus with ``n_spots``.
    """
    group_cols = ["source_file", "nuclei_index"]
    nucleus_counts = (
        all_spots.groupby(group_cols)
        .agg(n_spots=("theta_corrected", "count"))
        .reset_index()
    )
    valid_nuclei = nucleus_counts[nucleus_counts["n_spots"] >= min_spots]
    print(f"Nuclei with \u2265 {min_spots} spots: {len(valid_nuclei)}")

    filtered_spots = all_spots.merge(valid_nuclei[group_cols], on=group_cols, how="inner")
    print(f"Spots in filtered set: {len(filtered_spots)}")
    return filtered_spots, valid_nuclei


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_polar_and_anisotropy(
    spots_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    n_bins: int = 36,
    save_path: Path | str | None = None,
):
    """Two-panel figure: polar rose of θ-corrected angles + anisotropy histogram.

    Parameters
    ----------
    spots_df : pd.DataFrame
        spots dataframe with ``theta_corrected``, ``source_file``, and ``nuclei_index``.
    stats_df : pd.DataFrame
        per nucleus statistics dataframe with ``source_file``, ``nucleus_id``, and ``anisotropy``.
    n_bins : int
        Number of angular bins in the polar rose.
    save_path : Path or str, optional
        If given, the figure is saved there (PDF/PNG/SVG auto-detected by extension).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    theta_all = spots_df["theta_corrected"].values
    anisotropy_vals = (
        stats_df["anisotropy"].dropna().values
        if "anisotropy" in stats_df.columns
        else np.array([])
    )

    num_nuclei = len(stats_df.dropna(subset=['anisotropy']))

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Polar rose -----------------------------------------------------------
    ax_pol = fig.add_subplot(gs[0], projection="polar")
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    counts, _ = np.histogram(theta_all, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = 2 * np.pi / n_bins

    ax_pol.bar(
        bin_centers, counts,
        width=width * 0.9,
        color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.4,
    )
    ax_pol.set_theta_zero_location("E")
    ax_pol.set_theta_direction(1)
    ax_pol.set_title(
        f"\u03b8-corrected spot distribution\n"
        f"(n\u202f=\u202f{len(theta_all)} spots, {num_nuclei} nuclei)",
        pad=14, fontsize=10,
    )

    # --- Anisotropy histogram ------------------------------------------------
    ax_hist = fig.add_subplot(gs[1])
    if len(anisotropy_vals) > 0:
        ax_hist.hist(
            anisotropy_vals, bins=20, range=(0, 1),
            color="darkorange", alpha=0.8, edgecolor="white", linewidth=0.5,
        )
        median_a = np.nanmedian(anisotropy_vals)
        ax_hist.axvline(
            median_a, color="black", linestyle="--", linewidth=1.2,
            label=f"median = {median_a:.2f}",
        )
        ax_hist.legend(fontsize=9)
    else:
        ax_hist.text(
            0.5, 0.5, "No anisotropy data", ha="center", va="center",
            transform=ax_hist.transAxes, fontsize=11, color="gray",
        )

    ax_hist.set_xlabel("Anisotropy", fontsize=11)
    ax_hist.set_ylabel("Number of nuclei", fontsize=11)
    ax_hist.set_title(
        f"Anisotropy distribution\n(n\u202f=\u202f{len(anisotropy_vals)} nuclei)",
        fontsize=10,
    )
    ax_hist.set_xlim(0, 1)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Statistics convenience wrapper
# ---------------------------------------------------------------------------

def run_statistics_for_folder(
    input_folder: Path | str,
    output_folder: Path | str,
) -> list[pd.DataFrame]:
    """Run :func:`compute_statistics_per_image` for every image in *input_folder*.

    Parameters
    ----------
    input_folder : Path or str
        Folder containing the raw images (``.ics``, ``.tiff``, ``.tif``).
    output_folder : Path or str
        Folder where processed outputs (nuclei TIFFs, spots CSVs) were saved.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per successfully processed image.
    """
    from napari_cellspots._processing import compute_statistics_per_image

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    images = (
        sorted(input_folder.glob("*.ics"))
        + sorted(input_folder.glob("*.tiff"))
        + sorted(input_folder.glob("*.tif"))
    )
    results = []
    for idx, img in enumerate(images):
        try:
            df = compute_statistics_per_image(img, output_folder)
            print(f"[{idx}] {img.name}: {len(df)} nuclei")
            results.append(df)
        except Exception as exc:
            print(f"[{idx}] {img.name}: skipped – {exc}")
    return results
