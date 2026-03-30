from __future__ import annotations

from pathlib import Path

import napari
import napari.layers
import numpy as np
import pandas as pd
import tifffile
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from magicgui.widgets import create_widget
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider


# ---------------------------------------------------------------------------
# Module-level thread workers (defined once, not re-decorated on every call)
# ---------------------------------------------------------------------------

@thread_worker
def _worker_segment_cells(image_data, cell_proba, cell_channel, nucl_channel, diameter_nucl=30, diameter_cell=50):
    from napari_cellspots._processing import segment_cells2D
    return segment_cells2D(image_data, cell_proba, cell_channel=cell_channel, nucl_channel=nucl_channel, diameter_nucl=diameter_nucl, diameter_cell=diameter_cell)


@thread_worker
def _worker_segment_spots(image_data):
    from napari_cellspots._processing import segment_spots2D
    return segment_spots2D(image_data)


@thread_worker
def _worker_process_image(image_path, output_folder, cell_proba, cell_channel, nucl_channel, spot_channel, diameter_nucl=30, diameter_cell=50):
    from napari_cellspots._processing import process_image2D
    process_image2D(image_path, output_folder, cell_proba, cell_channel, nucl_channel, spot_channel, diameter_nucl=diameter_nucl, diameter_cell=diameter_cell)


@thread_worker
def _worker_process_folder(input_folder, output_folder, cell_proba, cell_channel, nucl_channel, spot_channel, diameter_nucl=30, diameter_cell=50):
    from napari_cellspots._processing import process_folder2D
    process_folder2D(input_folder, output_folder, cell_proba, cell_channel, nucl_channel, spot_channel, diameter_nucl=diameter_nucl, diameter_cell=diameter_cell)


@thread_worker
def _worker_compute_distances(output_folder, image_path):
    from napari_cellspots._processing import match_spots_to_nuclei
    
    spots_df = match_spots_to_nuclei(output_folder, image_path)
    return spots_df


@thread_worker
def _worker_compute_statistics(spots_df, nuclei_labels):
    import scipy.ndimage
    from napari_cellspots._quantification import compact_asymmetry_metrics

    if "nuclei_index" not in spots_df.columns:
        raise ValueError("Spots DataFrame has no 'nuclei_index' column. Run 'Compute distances' first.")

    nucleus_ids = sorted(spots_df["nuclei_index"].dropna().unique())
    nucleus_ids = [int(i) for i in nucleus_ids if int(i) > 0]

    rows = []
    for nid in nucleus_ids:
        mask = spots_df["nuclei_index"] == nid
        group = spots_df[mask]
        points = group[["x", "y"]].values
        # Use pre-computed polar coordinates if available
        if "r" in spots_df.columns and "theta" in spots_df.columns:
            r = group["r"].values.astype(float)
            theta = group["theta"].values.astype(float)
        else:
            cy, cx = scipy.ndimage.center_of_mass(nuclei_labels == nid)
            center = np.array([cy, cx])
            dy = points[:, 0] - center[0]
            dx = points[:, 1] - center[1]
            r = np.hypot(dy, dx)
            theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        metrics = compact_asymmetry_metrics(points, r, theta)
        rows.append({"nucleus_id": nid, "n_spots": len(group), **metrics})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class CellspotsProcessingWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._input_folder: Path | None = None
        self._output_folder: Path | None = None
        self._current_image_path: Path | None = None
        self._current_image_data: np.ndarray | None = None
        self._current_stem: str | None = None
        self._current_spots_df: pd.DataFrame | None = None
        self._build_ui()
        # Keep cell-layer combobox in sync with the viewer
        self._viewer.layers.events.inserted.connect(self._combo_cell_layer.reset_choices)
        self._viewer.layers.events.removed.connect(self._combo_cell_layer.reset_choices)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab1(), "Processing")
        tabs.addTab(self._build_tab2(), "Distances")
        outer.addWidget(tabs)
        self.setLayout(outer)

    def _build_tab1(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Input folder
        btn_input = QPushButton("Select input folder")
        btn_input.clicked.connect(self._on_select_input)
        self._lbl_input = QLabel("(no folder selected)")
        self._lbl_input.setWordWrap(True)
        layout.addWidget(btn_input)
        layout.addWidget(self._lbl_input)

        # File list
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._file_list.itemClicked.connect(self._on_file_selected)
        layout.addWidget(self._file_list)

        # Output folder
        btn_output = QPushButton("Select output folder")
        btn_output.clicked.connect(self._on_select_output)
        self._lbl_output = QLabel("(no folder selected)")
        self._lbl_output.setWordWrap(True)
        layout.addWidget(btn_output)
        layout.addWidget(self._lbl_output)

        # Cell probability threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("Cell probability threshold:"))
        self._spinbox_proba = QDoubleSpinBox()
        self._spinbox_proba.setRange(-6.0, 6.0)
        self._spinbox_proba.setSingleStep(0.5)
        self._spinbox_proba.setValue(0.0)
        row.addWidget(self._spinbox_proba)
        layout.addLayout(row)

        # Channel / diameter settings
        grp_ch = QGroupBox("Segmentation channels")
        ch_layout = QVBoxLayout()

        row_nucl = QHBoxLayout()
        row_nucl.addWidget(QLabel("Nucleus channel:"))
        self._spinbox_nucl_ch = QSpinBox()
        self._spinbox_nucl_ch.setRange(0, 31)
        self._spinbox_nucl_ch.setValue(1)
        row_nucl.addWidget(self._spinbox_nucl_ch)
        ch_layout.addLayout(row_nucl)

        row_cell = QHBoxLayout()
        row_cell.addWidget(QLabel("Cell channel:"))
        self._spinbox_cell_ch = QSpinBox()
        self._spinbox_cell_ch.setRange(0, 31)
        self._spinbox_cell_ch.setValue(0)
        row_cell.addWidget(self._spinbox_cell_ch)
        ch_layout.addLayout(row_cell)

        row_spot = QHBoxLayout()
        row_spot.addWidget(QLabel("Spot channel:"))
        self._spinbox_spot_ch = QSpinBox()
        self._spinbox_spot_ch.setRange(0, 31)
        self._spinbox_spot_ch.setValue(2)
        row_spot.addWidget(self._spinbox_spot_ch)
        ch_layout.addLayout(row_spot)

        self._chk_segment_cells = QCheckBox("Segment cells")
        self._chk_segment_cells.setChecked(True)
        self._chk_segment_cells.toggled.connect(self._on_segment_cells_toggled)
        ch_layout.addWidget(self._chk_segment_cells)

        self._chk_segment_spots = QCheckBox("Segment spots")
        self._chk_segment_spots.setChecked(True)
        self._chk_segment_spots.toggled.connect(self._on_segment_spots_toggled)
        ch_layout.addWidget(self._chk_segment_spots)

        row_diam = QHBoxLayout()
        row_diam.addWidget(QLabel("Cell diameter (px):"))
        self._spinbox_diameter_cell = QSpinBox()
        self._spinbox_diameter_cell.setRange(1, 2000)
        self._spinbox_diameter_cell.setValue(50)
        row_diam.addWidget(self._spinbox_diameter_cell)
        ch_layout.addLayout(row_diam)

        row_diam_nucl = QHBoxLayout()
        row_diam_nucl.addWidget(QLabel("Nucleus diameter (px):"))
        self._spinbox_diameter_nucl = QSpinBox()
        self._spinbox_diameter_nucl.setRange(1, 2000)
        self._spinbox_diameter_nucl.setValue(30)
        row_diam_nucl.addWidget(self._spinbox_diameter_nucl)
        ch_layout.addLayout(row_diam_nucl)

        grp_ch.setLayout(ch_layout)
        layout.addWidget(grp_ch)

        # Processing buttons
        grp_proc = QGroupBox("Processing")
        proc_layout = QVBoxLayout()
        for label, slot in [
            ("Segment cells (2D)", self._run_segment_cells),
            ("Segment spots (2D)", self._run_segment_spots),
            ("Process image (2D)", self._run_process_image),
            ("Process folder (2D)", self._run_process_folder),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            proc_layout.addWidget(btn)
        grp_proc.setLayout(proc_layout)
        layout.addWidget(grp_proc)

        # Spot coloring (exclusive radio buttons)
        grp_color = QGroupBox("Color spots by")
        color_layout = QVBoxLayout()
        self._radio_nucleus = QRadioButton("Nucleus ID")
        self._radio_dist = QRadioButton("Distance to nucleus")
        self._radio_nucleus.setChecked(True)
        self._btn_grp = QButtonGroup(self)
        self._btn_grp.addButton(self._radio_nucleus)
        self._btn_grp.addButton(self._radio_dist)
        self._btn_grp.buttonClicked.connect(self._on_color_changed)
        color_layout.addWidget(self._radio_nucleus)
        color_layout.addWidget(self._radio_dist)
        grp_color.setLayout(color_layout)
        layout.addWidget(grp_color)

        # Spot filtering by intensity and probability
        grp_filter = QGroupBox("Filter spots")
        filter_layout = QVBoxLayout()

        filter_layout.addWidget(QLabel("Intensity range:"))
        self._slider_intens = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._slider_intens.setRange(0.0, 1.0)
        self._slider_intens.setValue((0.0, 1.0))
        self._slider_intens.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._slider_intens)

        filter_layout.addWidget(QLabel("Probability range:"))
        self._slider_prob = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._slider_prob.setRange(0.0, 1.0)
        self._slider_prob.setValue((0.0, 1.0))
        self._slider_prob.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._slider_prob)

        grp_filter.setLayout(filter_layout)
        layout.addWidget(grp_filter)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _on_segment_cells_toggled(self, checked: bool):
        self._spinbox_cell_ch.setEnabled(checked)

    def _on_segment_spots_toggled(self, checked: bool):
        self._spinbox_spot_ch.setEnabled(checked)

    def _cell_channel_value(self) -> int | None:
        """Return cell channel int, or None if cell segmentation is disabled."""
        return self._spinbox_cell_ch.value() if self._chk_segment_cells.isChecked() else None
    
    def _spots_channel_value(self) -> int | None:
        """Return the channel index to use for spot detection."""
        return self._spinbox_spot_ch.value() if self._chk_segment_spots.isChecked() else None

    def _build_tab2(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._combo_cell_layer = create_widget(annotation=napari.layers.Labels, label='Cell mask layer')
        self._combo_cell_layer.reset_choices()
        layout.addWidget(self._combo_cell_layer.native)

        btn_add_cells = QPushButton("Add empty cells layer")
        btn_add_cells.clicked.connect(self._add_empty_cells_layer)
        layout.addWidget(btn_add_cells)

        btn_save_cells = QPushButton("Save cells layer")
        btn_save_cells.clicked.connect(self._save_cells_layer)
        layout.addWidget(btn_save_cells)

        btn_dist = QPushButton("Compute distances")
        btn_dist.clicked.connect(self._run_compute_distances)
        layout.addWidget(btn_dist)

        btn_stats = QPushButton("Compute per-cell statistics")
        btn_stats.clicked.connect(self._run_compute_statistics)
        layout.addWidget(btn_stats)

        _STAT_COLS = ["nucleus_id", "n_spots", "mean_resultant_length", "circular_mean",
                      "radial_mean", "anisotropy", "major_axis_angle"]
        self._stats_table = QTableWidget(0, len(_STAT_COLS))
        self._stats_table.setHorizontalHeaderLabels(_STAT_COLS)
        self._stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self._stats_table)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    # ------------------------------------------------------------------
    # Folder / file selection
    # ------------------------------------------------------------------

    def _on_select_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if not folder:
            return
        self._input_folder = Path(folder)
        self._lbl_input.setText(str(self._input_folder))
        self._populate_file_list()

    def _on_select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        self._output_folder = Path(folder)
        self._lbl_output.setText(str(self._output_folder))
        # Reload outputs for the currently displayed image if one is loaded
        if self._current_image_path is not None:
            self._check_and_load_outputs()

    def _populate_file_list(self):
        self._file_list.clear()
        if self._input_folder is None:
            return
        #files = sorted(self._input_folder.glob("*.ids")) + sorted(
        #    self._input_folder.glob("*.ics")
        #)
        files = sorted(self._input_folder.glob("*.ics")) + sorted(
            self._input_folder.glob("*.tiff")
        ) + sorted(
            self._input_folder.glob("*.tif")
        )
        for f in files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            self._file_list.addItem(item)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _on_file_selected(self, item: QListWidgetItem):
        self._load_image(Path(item.data(Qt.UserRole)))

    def _load_image(self, path: Path):
        # Remove all layers from the viewer before loading the new image
        self._viewer.layers.clear()

        self._current_image_path = path
        self._current_stem = path.stem
        self._current_spots_df = None

        try:
            if path.suffix.lower() in (".tiff", ".tif"):
                image_data = tifffile.imread(str(path))
            else:
                import pyics
                image_data, _meta = pyics.imread(path.as_posix())
        except Exception as exc:
            show_error(f"Failed to load {path.name}: {exc}")
            return

        # Take the middle Z-slice for 2D display: (C, Z, H, W) → (C, H, W)
        if image_data.ndim == 4:
            mid_z = image_data.shape[1] // 2
            display_data = image_data[:, mid_z, :, :]
        else:
            display_data = image_data

        self._current_image_data = display_data
        self._viewer.add_image(display_data, name=path.stem, channel_axis=0)
        self._check_and_load_outputs()

    def _check_and_load_outputs(self):
        """Load nuclei TIFF and spots CSV from the output folder if they exist."""
        if self._output_folder is None or self._current_image_path is None:
            return
        stem = self._current_stem
        parent_name = self._current_image_path.parent.name

        # Check two candidate locations: output_folder/parent_name/ and output_folder/
        candidates = [
            self._output_folder / parent_name,
            self._output_folder,
        ]

        nuclei_path: Path | None = None
        cells_path: Path | None = None
        spots_path: Path | None = None
        for folder in candidates:
            t = folder / f"{stem}_nuclei.tiff"
            cl = folder / f"{stem}_cells.tiff"
            c = folder / f"{stem}_spots.csv"
            if nuclei_path is None and t.exists():
                nuclei_path = t
            if cells_path is None and cl.exists():
                cells_path = cl
            if spots_path is None and c.exists():
                spots_path = c

        if nuclei_path is not None:
            nuclei = tifffile.imread(str(nuclei_path))
            self._viewer.add_labels(nuclei.astype(int), name=f"{stem}_nuclei")

        if cells_path is not None:
            cells = tifffile.imread(str(cells_path))
            self._viewer.add_labels(cells.astype(int), name=f"{stem}_cells")

        if spots_path is not None:
            spots_df = pd.read_csv(spots_path)
            self._add_spots_from_df(spots_df, f"{stem}_spots")

    # ------------------------------------------------------------------
    # Spots layer helpers
    # ------------------------------------------------------------------

    def _add_spots_from_df(self, spots_df: pd.DataFrame, layer_name: str):
        # Store the full (unfiltered) DataFrame and update slider ranges
        self._current_spots_df = spots_df.copy()
        self._update_filter_sliders(spots_df)
        self._draw_spots_layer(self._apply_filter(spots_df), layer_name)

    def _update_filter_sliders(self, spots_df: pd.DataFrame):
        """Set slider ranges to the actual min/max of the loaded spots data."""
        if "intens" in spots_df.columns:
            vals = spots_df["intens"].dropna()
            lo, hi = float(vals.min()), float(vals.max())
            # Block signals to avoid triggering filter during range update
            self._slider_intens.blockSignals(True)
            self._slider_intens.setRange(lo, hi)
            self._slider_intens.setValue((lo, hi))
            self._slider_intens.blockSignals(False)

        if "prob" in spots_df.columns:
            vals = spots_df["prob"].dropna()
            lo, hi = float(vals.min()), float(vals.max())
            self._slider_prob.blockSignals(True)
            self._slider_prob.setRange(lo, hi)
            self._slider_prob.setValue((lo, hi))
            self._slider_prob.blockSignals(False)

    def _apply_filter(self, spots_df: pd.DataFrame) -> pd.DataFrame:
        """Return a filtered copy of spots_df based on current slider values."""
        mask = pd.Series(True, index=spots_df.index)
        if "intens" in spots_df.columns:
            lo, hi = self._slider_intens.value()
            mask &= spots_df["intens"].between(lo, hi)
        if "prob" in spots_df.columns:
            lo, hi = self._slider_prob.value()
            mask &= spots_df["prob"].between(lo, hi)
        return spots_df[mask]

    def _draw_spots_layer(self, spots_df: pd.DataFrame, layer_name: str):
        """Add or replace the named Points layer with data from spots_df."""
        coords = spots_df[["x", "y"]].values
        props: dict = {}
        if "nuclei_index" in spots_df.columns:
            props["nuclei_index"] = (
                spots_df["nuclei_index"].fillna(0).values.astype(float)
            )
        if "dists" in spots_df.columns:
            props["dists"] = spots_df["dists"].fillna(0).values.astype(float)

        # Update in-place if layer already exists, otherwise create it
        existing = next(
            (l for l in self._viewer.layers if l.name == layer_name
             and isinstance(l, napari.layers.Points)),
            None,
        )
        if existing is not None:
            existing.data = coords
            existing.properties = props
            self._apply_spot_coloring(existing)
        else:
            layer = self._viewer.add_points(
                coords,
                properties=props,
                name=layer_name,
                size=5,
                face_color="white",
            )
            self._apply_spot_coloring(layer)

    def _on_filter_changed(self):
        if self._current_spots_df is None or self._current_stem is None:
            return
        layer_name = f"{self._current_stem}_spots"
        filtered = self._apply_filter(self._current_spots_df)
        self._draw_spots_layer(filtered, layer_name)

    def _find_spots_layer(self) -> napari.layers.Points | None:
        if self._current_stem is None:
            return None
        target = f"{self._current_stem}_spots"
        for layer in self._viewer.layers:
            if layer.name == target and isinstance(layer, napari.layers.Points):
                return layer
        return None

    def _apply_spot_coloring(self, layer: napari.layers.Points):
        try:
            if self._radio_nucleus.isChecked():
                if "nuclei_index" in layer.properties and len(layer.data) > 0:
                    vals = np.asarray(layer.properties["nuclei_index"], dtype=float)
                    layer.face_colormap = "tab20"
                    layer.face_color = "nuclei_index"
                    layer.face_contrast_limits = (float(vals.min()), float(vals.max()))
            else:
                if "dists" in layer.properties and len(layer.data) > 0:
                    vals = np.asarray(layer.properties["dists"], dtype=float)
                    finite = vals[np.isfinite(vals)]
                    if len(finite) > 0:
                        layer.face_colormap = "viridis"
                        layer.face_color = "dists"
                        layer.face_contrast_limits = (
                            float(finite.min()),
                            float(finite.max()),
                        )
        except Exception:
            pass  # Coloring is non-critical

    def _on_color_changed(self):
        layer = self._find_spots_layer()
        if layer is not None:
            self._apply_spot_coloring(layer)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _remove_layers_for_stem(self, stem: str):
        names = {stem, f"{stem}_nuclei", f"{stem}_cells", f"{stem}_spots"}
        for layer in list(self._viewer.layers):
            if layer.name in names:
                self._viewer.layers.remove(layer)

    def _remove_output_layers_for_stem(self, stem: str):
        """Remove only the nuclei/spots layers, keeping the raw image."""
        names = {f"{stem}_nuclei", f"{stem}_spots"}
        for layer in list(self._viewer.layers):
            if layer.name in names:
                self._viewer.layers.remove(layer)

    # ------------------------------------------------------------------
    # Processing buttons
    # ------------------------------------------------------------------

    def _run_segment_cells(self):
        if self._current_image_data is None:
            show_error("No image loaded. Please select a file first.")
            return
        stem = self._current_stem
        worker = _worker_segment_cells(
            self._current_image_data,
            self._spinbox_proba.value(),
            self._cell_channel_value(),
            self._spinbox_nucl_ch.value(),
            self._spinbox_diameter_nucl.value(),
            self._spinbox_diameter_cell.value(),
        )
        worker.returned.connect(lambda result: self._on_cells_done(result, stem))
        worker.errored.connect(lambda exc: show_error(f"Cell segmentation failed: {exc}"))
        worker.start()

    def _on_cells_done(self, result: tuple, stem: str):
        if self._current_stem != stem:
            return  # Image changed while worker was running
        nuclei_labels, cell_labels = result
        nuclei_labels = nuclei_labels.astype(int)

        for name, data in [(f"{stem}_nuclei", nuclei_labels), (f"{stem}_cells", cell_labels)]:
            for layer in list(self._viewer.layers):
                if layer.name == name:
                    self._viewer.layers.remove(layer)
            if data is not None:
                self._viewer.add_labels(data.astype(int), name=name)

        if self._output_folder and self._current_image_path:
            out = self._output_folder / self._current_image_path.parent.name
            out.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(out / f"{stem}_nuclei.tiff"), nuclei_labels.astype(np.uint16))
            if cell_labels is not None:
                tifffile.imwrite(str(out / f"{stem}_cells.tiff"), cell_labels.astype(np.uint16))

        n_nuclei = nuclei_labels.max()
        msg = f"Cells segmented: {n_nuclei} nuclei found."
        if cell_labels is not None:
            msg += f" {cell_labels.max()} cell regions found."
        show_info(msg)

    def _run_segment_spots(self):
        if self._current_image_data is None:
            show_error("No image loaded. Please select a file first.")
            return
        stem = self._current_stem
        worker = _worker_segment_spots(self._current_image_data[self._spots_channel_value()])
        worker.returned.connect(lambda result: self._on_spots_done(result, stem))
        worker.errored.connect(lambda exc: show_error(f"Spot detection failed: {exc}"))
        worker.start()

    def _on_spots_done(self, spots_df: pd.DataFrame, stem: str):
        if self._current_stem != stem:
            return
        layer_name = f"{stem}_spots"
        for layer in list(self._viewer.layers):
            if layer.name == layer_name:
                self._viewer.layers.remove(layer)
        self._add_spots_from_df(spots_df, layer_name)

        if self._output_folder and self._current_image_path:
            out = self._output_folder / self._current_image_path.parent.name
            out.mkdir(parents=True, exist_ok=True)
            spots_df.to_csv(out / f"{stem}_spots.csv", index=False)

        show_info(f"Spots detected: {len(spots_df)} spots found.")

    def _run_process_image(self):
        if self._current_image_path is None:
            show_error("No image selected. Please select a file first.")
            return
        if self._output_folder is None:
            show_error("No output folder selected.")
            return
        stem = self._current_stem
        worker = _worker_process_image(
            self._current_image_path,
            self._output_folder,
            self._spinbox_proba.value(),
            self._cell_channel_value(),
            self._spinbox_nucl_ch.value(),
            self._spots_channel_value(),
            self._spinbox_diameter_nucl.value(),
            self._spinbox_diameter_cell.value(),
        )
        worker.returned.connect(lambda _: self._on_process_image_done(stem))
        worker.errored.connect(lambda exc: show_error(f"Processing failed: {exc}"))
        worker.start()

    def _on_process_image_done(self, stem: str):
        if self._current_stem != stem:
            return
        # Remove stale output layers before reloading from disk
        self._remove_output_layers_for_stem(stem)
        self._check_and_load_outputs()
        show_info(f"Processing complete for {stem}.")

    def _run_process_folder(self):
        if self._input_folder is None:
            show_error("No input folder selected.")
            return
        if self._output_folder is None:
            show_error("No output folder selected.")
            return
        worker = _worker_process_folder(
            self._input_folder,
            self._output_folder,
            self._spinbox_proba.value(),
            self._cell_channel_value(),
            self._spinbox_nucl_ch.value(),
            self._spots_channel_value(),
            self._spinbox_diameter_nucl.value(),
            self._spinbox_diameter_cell.value(),
        )
        worker.returned.connect(lambda _: show_info("Folder processing complete."))
        worker.errored.connect(lambda exc: show_error(f"Folder processing failed: {exc}"))
        worker.start()

    # ------------------------------------------------------------------
    # Distances tab
    # ------------------------------------------------------------------

    def _save_cells_layer(self):
        cell_layer = self._combo_cell_layer.value
        if cell_layer is None:
            show_error("No cell mask layer selected.")
            return
        if self._output_folder is None or self._current_image_path is None:
            show_error("No output folder selected.")
            return
        out_folder = self._output_folder / self._current_image_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)
        save_path = out_folder / f"{self._current_stem}_cells.tiff"
        tifffile.imwrite(str(save_path), cell_layer.data.astype(np.uint32))
        show_info(f"Cells layer saved to {save_path}.")

    def _add_empty_cells_layer(self):
        if self._current_image_data is None:
            show_error("No image loaded. Please select a file first.")
            return
        if self._current_stem is None:
            return
        layer_name = f"{self._current_stem}_cells"
        # Use the spatial shape of the current image (H, W), ignoring channel axis
        shape = self._current_image_data.shape[-2:]
        empty = np.zeros(shape, dtype=np.uint32)
        # Remove any pre-existing layer with that name
        for layer in list(self._viewer.layers):
            if layer.name == layer_name:
                self._viewer.layers.remove(layer)
        self._viewer.add_labels(empty, name=layer_name)
        show_info(f"Empty cells layer '{layer_name}' added ({shape[0]}×{shape[1]}).")

    def _refresh_cell_layers(self, _event=None):
        """Repopulate the cell layer combobox."""
        self._combo_cell_layer.reset_choices()

    def _run_compute_distances(self):
        if self._current_spots_df is None:
            show_error("No spots loaded. Load or detect spots first.")
            return
        if self._current_stem is None:
            show_error("No image selected.")
            return

        # Locate the nuclei labels layer
        nuclei_layer_name = f"{self._current_stem}_nuclei"
        nuclei_layer = next(
            (l for l in self._viewer.layers
             if l.name == nuclei_layer_name and isinstance(l, napari.layers.Labels)),
            None,
        )
        if nuclei_layer is None:
            show_error(f"No nuclei layer '{nuclei_layer_name}' found in viewer.")
            return

        # Locate the selected cell mask layer
        cell_layer = self._combo_cell_layer.value
        if cell_layer is None:
            show_error("No cell mask layer selected.")
            return

        if self._output_folder is None or self._current_image_path is None:
            show_error("No output folder selected.")
            return

        out_folder = self._output_folder / self._current_image_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)
        stem = self._current_stem

        worker = _worker_compute_distances(
            self._output_folder,
            self._current_image_path,
        )
        worker.returned.connect(lambda df: self._on_distances_done(df, stem))
        worker.errored.connect(lambda exc: show_error(f"Distance computation failed: {exc}"))
        worker.start()

    def _on_distances_done(self, spots_df: pd.DataFrame, stem: str):
        if self._current_stem != stem:
            return
        layer_name = f"{stem}_spots"
        self._add_spots_from_df(spots_df, layer_name)
        show_info(f"Distances computed and saved for {stem}.")

    # ------------------------------------------------------------------
    # Per-cell statistics
    # ------------------------------------------------------------------

    def _run_compute_statistics(self):
        if self._current_spots_df is None:
            show_error("No spots loaded. Load or detect spots first.")
            return
        if self._current_stem is None:
            return

        nuclei_layer_name = f"{self._current_stem}_nuclei"
        nuclei_layer = next(
            (l for l in self._viewer.layers
             if l.name == nuclei_layer_name and isinstance(l, napari.layers.Labels)),
            None,
        )
        if nuclei_layer is None:
            show_error(f"No nuclei layer '{nuclei_layer_name}' found in viewer.")
            return

        stem = self._current_stem
        worker = _worker_compute_statistics(
            self._current_spots_df,
            nuclei_layer.data,
        )
        worker.returned.connect(lambda df: self._on_statistics_done(df, stem))
        worker.errored.connect(lambda exc: show_error(f"Statistics computation failed: {exc}"))
        worker.start()

    def _on_statistics_done(self, stats_df: pd.DataFrame, stem: str):
        if self._current_stem != stem:
            return

        COLS = ["nucleus_id", "n_spots", "mean_resultant_length", "circular_mean",
                "radial_mean", "anisotropy", "major_axis_angle"]
        self._stats_table.setRowCount(0)
        for _, row in stats_df.iterrows():
            r = self._stats_table.rowCount()
            self._stats_table.insertRow(r)
            for c, col in enumerate(COLS):
                val = row.get(col, float("nan"))
                if col in ("nucleus_id", "n_spots"):
                    text = str(int(val))
                elif col == "major_axis_angle":
                    text = f"{np.degrees(val):.1f}°" if np.isfinite(val) else "nan"
                elif col == "circular_mean":
                    text = f"{np.degrees(val):.1f}°" if np.isfinite(val) else "nan"
                else:
                    text = f"{val:.4f}" if np.isfinite(val) else "nan"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self._stats_table.setItem(r, c, item)

        if self._output_folder is not None and self._current_image_path is not None:
            out_folder = self._output_folder / self._current_image_path.parent.name
            out_folder.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(out_folder / f"{stem}_statistics.csv", index=False)

        show_info(f"Statistics computed for {len(stats_df)} nuclei in {stem}.")
