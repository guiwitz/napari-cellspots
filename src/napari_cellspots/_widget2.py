from __future__ import annotations

from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class CellspotsPolarWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._output_folder: Path | None = None
        # Cache: list of dicts with keys r, theta, intens, prob
        self._cached_spots: list[dict] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Output folder (contains processed CSVs and TIFFs)
        btn_out = QPushButton("Select output folder")
        btn_out.clicked.connect(self._on_select_output)
        self._lbl_out = QLabel("(no folder selected)")
        self._lbl_out.setWordWrap(True)
        layout.addWidget(btn_out)
        layout.addWidget(self._lbl_out)

        # Spot filtering
        grp_filter = QGroupBox("Filter spots")
        filter_layout = QVBoxLayout()

        filter_layout.addWidget(QLabel("Intensity range:"))
        self._slider_intens = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._slider_intens.setRange(0.0, 1.0)
        self._slider_intens.setValue((0.0, 1.0))
        self._slider_intens.valueChanged.connect(self._redraw)
        filter_layout.addWidget(self._slider_intens)

        filter_layout.addWidget(QLabel("Probability range:"))
        self._slider_prob = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._slider_prob.setRange(0.0, 1.0)
        self._slider_prob.setValue((0.0, 1.0))
        self._slider_prob.valueChanged.connect(self._redraw)
        filter_layout.addWidget(self._slider_prob)

        grp_filter.setLayout(filter_layout)
        layout.addWidget(grp_filter)

        btn_scan = QPushButton("Scan ranges from data")
        btn_scan.clicked.connect(self._scan_ranges)
        layout.addWidget(btn_scan)

        # Plot button
        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self._on_plot)
        layout.addWidget(btn_plot)

        # Embedded polar plot
        self._fig = Figure(figsize=(4, 4), tight_layout=True)
        self._ax = self._fig.add_subplot(111, projection="polar")
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------

    def _on_select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        self._output_folder = Path(folder)
        self._lbl_out.setText(str(self._output_folder))

    def _scan_ranges(self):
        """Read all CSVs in the output folder and auto-range the filter sliders."""
        if self._output_folder is None:
            show_error("No output folder selected.")
            return
        csv_files = list(self._output_folder.rglob("*_spots.csv"))
        if not csv_files:
            show_error("No *_spots.csv files found.")
            return

        all_intens: list[float] = []
        all_prob: list[float] = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if "intens" in df.columns:
                    all_intens.extend(df["intens"].dropna().tolist())
                if "prob" in df.columns:
                    all_prob.extend(df["prob"].dropna().tolist())
            except Exception:
                continue

        if all_intens:
            lo, hi = float(min(all_intens)), float(max(all_intens))
            self._slider_intens.blockSignals(True)
            self._slider_intens.setRange(lo, hi)
            self._slider_intens.setValue((lo, hi))
            self._slider_intens.blockSignals(False)

        if all_prob:
            lo, hi = float(min(all_prob)), float(max(all_prob))
            self._slider_prob.blockSignals(True)
            self._slider_prob.setRange(lo, hi)
            self._slider_prob.setValue((lo, hi))
            self._slider_prob.blockSignals(False)

        show_info(f"Ranges updated from {len(csv_files)} CSV file(s).")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _on_plot(self):
        if self._output_folder is None:
            show_error("No output folder selected.")
            return

        csv_files = list(self._output_folder.rglob("*_spots.csv"))
        if not csv_files:
            show_error("No *_spots.csv files found in the output folder.")
            return

        self._cached_spots.clear()

        for csv_file in csv_files:
            spots_df = pd.read_csv(csv_file)
            if not {"nuclei_index"}.issubset(spots_df.columns):
                continue

            # Load per-nucleus major_axis_angle for alignment if statistics exist
            stem = csv_file.stem.replace("_spots", "")
            stats_csv = csv_file.parent / f"{stem}_statistics.csv"
            major_axis_map: dict[int, float] = {}
            if stats_csv.exists():
                try:
                    stats_df = pd.read_csv(stats_csv)
                    if {"nucleus_id", "major_axis_angle"}.issubset(stats_df.columns):
                        major_axis_map = dict(
                            zip(stats_df["nucleus_id"].astype(int),
                                stats_df["major_axis_angle"].astype(float))
                        )
                except Exception:
                    pass

            has_polar = {"r", "theta"}.issubset(spots_df.columns)

            if has_polar:
                # Use pre-computed polar coordinates — no nuclei TIFF needed
                assigned = spots_df[spots_df["nuclei_index"].fillna(0).astype(int) != 0].copy()
                if major_axis_map:
                    assigned["theta_align"] = assigned.apply(
                        lambda row: (row["theta"] - major_axis_map.get(int(row["nuclei_index"]), 0.0)) % (2 * np.pi),
                        axis=1,
                    )
                for _, row in assigned.iterrows():
                    self._cached_spots.append({
                        "r": float(row["r"]),
                        "theta": float(row["theta"]),
                        "theta_align": float(row["theta_align"]) if "theta_align" in row else float("nan"),
                        "intens": float(row["intens"]) if "intens" in row else float("nan"),
                        "prob": float(row["prob"]) if "prob" in row else float("nan"),
                    })
            else:
                # Fallback: compute from nuclei TIFF
                if not {"x", "y"}.issubset(spots_df.columns):
                    continue
                nuclei_tiff = csv_file.parent / f"{stem}_nuclei.tiff"
                if not nuclei_tiff.exists():
                    continue
                nuclei = tifffile.imread(str(nuclei_tiff))
                nuclei_ids = spots_df["nuclei_index"].dropna().astype(int).unique()
                from napari_cellspots._processing import cartesian_to_polar
                for nid in nuclei_ids:
                    if nid == 0:
                        continue
                    nucleus_mask = nuclei == nid
                    if not nucleus_mask.any():
                        continue
                    centroid = np.argwhere(nucleus_mask).mean(axis=0)
                    mask = spots_df["nuclei_index"].fillna(0).astype(int) == nid
                    sub = spots_df[mask]
                    points = sub[["x", "y"]].values
                    if len(points) == 0:
                        continue
                    r, theta = cartesian_to_polar(points, centroid)
                    correction = major_axis_map.get(int(nid), 0.0)
                    theta_align = (theta - correction) % (2 * np.pi)
                    intens_vals = sub["intens"].values if "intens" in sub.columns else np.full(len(r), np.nan)
                    prob_vals = sub["prob"].values if "prob" in sub.columns else np.full(len(r), np.nan)
                    for ri, ti, tai, iv, pv in zip(r, theta, theta_align, intens_vals, prob_vals):
                        self._cached_spots.append({"r": ri, "theta": ti, "theta_align": tai, "intens": iv, "prob": pv})

        if not self._cached_spots:
            show_error("No spots with assigned nucleus IDs found.")
            return

        self._redraw()

    def _redraw(self):
        """Filter cached spots by current slider values and refresh the plot."""
        if not self._cached_spots:
            return

        i_lo, i_hi = self._slider_intens.value()
        p_lo, p_hi = self._slider_prob.value()

        filtered_r = []
        filtered_theta = []
        for s in self._cached_spots:
            iv = s["intens"]
            pv = s["prob"]
            if not np.isnan(iv) and not (i_lo <= iv <= i_hi):
                continue
            if not np.isnan(pv) and not (p_lo <= pv <= p_hi):
                continue
            filtered_r.append(s["r"])
            #filtered_theta.append(s["theta"])
            filtered_theta.append(s["theta_align"])

        self._ax.clear()
        self._ax.scatter(filtered_theta, filtered_r, s=2, alpha=0.4, c="steelblue")
        self._ax.set_title(f"Polar plot ({len(filtered_r)} spots)")
        self._canvas.draw()
