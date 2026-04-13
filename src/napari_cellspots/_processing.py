"""
Processing functions copied from cellspots_script.py.
Heavy imports (cellpose, spotiflow) happen only when the individual functions are called.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform
import tifffile
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Folder / image processing
# ---------------------------------------------------------------------------

def process_folder2D(
    folder_path: Path | str,
    output_path: Path | str,
    cell_proba: float,
    cell_channel: int | None,
    nucl_channel: int,
    spot_channel: int | None,
    diameter_nucl: int = 30,
    diameter_cell: int = 50,
    plane: int = None,
    scaling_factor: int = 4,
    pixel_size_xy: float = 1.0,
    pixel_size_z: float = 1.0,
    do_3D: bool = False,
) -> None:
    """Process all images in a folder with :func:`process_image2D`.

    Parameters
    ----------
    folder_path : Path or str
        Input folder containing ``.ics``, ``.tiff``, or ``.tif`` images.
    output_path : Path or str
        Root output directory; per-image sub-folders are created automatically.
    cell_proba : float
        Cellpose cell-probability threshold.
    cell_channel : int or None
        Channel index for cell segmentation, or ``None`` to skip.
    nucl_channel : int
        Channel index for nucleus segmentation.
    spot_channel : int or None
        Channel index for spot detection, or ``None`` to skip.
    diameter_nucl : int
        Expected nucleus diameter in pixels.
    diameter_cell : int
        Expected cell diameter in pixels.
    plane : int or None
        Plane index for 3D image processing.  If specified, only this plane is processed; otherwise, the middle plane is used.
    scaling_factor : int
        Downscaling factor applied before Cellpose; labels are upscaled back.
    pixel_size_xy : float
        Physical pixel size in XY dimensions (e.g. microns per pixel).
    pixel_size_z : float
        Physical pixel size in Z dimension (e.g. microns per pixel).
    do_3D : bool
        If ``True``, process the entire 3D stack instead of a single plane.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    folder_path = Path(folder_path)
    image_files = (
        sorted(folder_path.glob("*.ics"))
        + sorted(folder_path.glob("*.tiff"))
        + sorted(folder_path.glob("*.tif"))
    )
    for image_file in image_files:
        process_image2D(image_file, output_path, cell_proba, cell_channel,
                        nucl_channel, spot_channel, diameter_nucl=diameter_nucl,
                        diameter_cell=diameter_cell, plane=plane,
                        scaling_factor=scaling_factor,
                        pixel_size_xy=pixel_size_xy, pixel_size_z=pixel_size_z,
                        do_3D=do_3D)


def process_image2D(
    image_path: Path | str,
    output_path: Path | str,
    cell_proba: float,
    cell_channel: int | None,
    nucl_channel: int,
    spot_channel: int | None,
    diameter_nucl: int = 30,
    diameter_cell: int = 50,
    scaling_factor: int = 4,
    pixel_size_xy: float = 1.0,
    pixel_size_z: float = 1.0,
    plane: int = None,
    do_3D: bool = False,
) -> None:
    """Segment cells and spots in a single image and save all outputs.

    Runs the full pipeline: cell/nucleus segmentation, spot detection, nucleus
    statistics, spot-to-nucleus assignment, polar coordinates, angle correction,
    and per-image asymmetry statistics.

    Parameters
    ----------
    image_path : Path or str
        Path to the input image (``.ics``, ``.tiff``, or ``.tif``). The imported image
        is expected to have shape (C, H, W) or (C, Z, H, W).
    output_path : Path or str
        Root output directory; a sub-folder named after the image's parent
        directory is created automatically.
    cell_proba : float
        Cellpose cell-probability threshold.
    cell_channel : int or None
        Channel index for cell segmentation, or ``None`` to skip.
    nucl_channel : int
        Channel index for nucleus segmentation.
    spot_channel : int or None
        Channel index for spot detection, or ``None`` to skip.
    diameter_nucl : int
        Expected nucleus diameter in pixels.
    diameter_cell : int
        Expected cell diameter in pixels.
    scaling_factor : int
        Downscaling factor applied before Cellpose; labels are upscaled back.
    pixel_size_xy : float
        Physical pixel size in XY dimensions (e.g. microns per pixel).
    pixel_size_z : float
        Physical pixel size in Z dimension (e.g. microns per pixel).
    plane : int or None
        Plane index for 3D image processing.  If specified, only this plane is processed; otherwise, the middle plane is used.
    do_3D : bool
        If ``True``, process the entire 3D stack instead of a single plane.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    print(f"Processing {image_path}")

    xy_z_factor = pixel_size_xy / pixel_size_z if do_3D else 1.0

    if image_path.suffix.lower() in (".tiff", ".tif"):
        image_data = tifffile.imread(str(image_path))
    else:
        import pyics
        image_data, _meta = pyics.imread(image_path.as_posix())

    if image_data.ndim == 4 and not do_3D:
        if plane is not None:
            image_data = image_data[:, plane, :, :]
        else:
            image_data = image_data[:, image_data.shape[1] // 2, :, :]
    print("Segmenting cells...")
    nuclei_label_origscale, cell_label_origscale = segment_cells2D(
        image_data, cell_proba, cell_channel, nucl_channel,
        diameter_nucl=diameter_nucl, diameter_cell=diameter_cell,
        scaling_factor=scaling_factor, xy_z_factor=xy_z_factor, do_3D=do_3D
    )
    print("Segmenting spots...")
    if spot_channel is not None:
        spots_df = segment_spots2D(
            image_data[spot_channel], do_3D=do_3D)
    else:
        spots_df = pd.DataFrame()
    print("Measuring distances from spots to cell surface...")

    output_folder = output_path / image_path.parent.name
    output_folder.mkdir(parents=True, exist_ok=True)

    spots_df.to_csv(output_folder / f"{image_path.stem}_spots.csv", index=False)

    nuclei_stats_df = compute_nuclei_stats(nuclei_label_origscale)
    nuclei_stats_df.to_csv(output_folder / f"{image_path.stem}_statistics.csv", index=False)

    tifffile.imwrite(
        str(output_folder / f"{image_path.stem}_nuclei.tiff"),
        nuclei_label_origscale.astype(np.uint16),
    )
    if cell_label_origscale is not None:
        tifffile.imwrite(
            str(output_folder / f"{image_path.stem}_cells.tiff"),
            cell_label_origscale.astype(np.uint16),
        )

    spots_df = match_spots_to_nuclei(
        output_path=output_path, 
        image_path=image_path, 
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z
    )
    
    nuclei_stats_df = compute_statistics_per_image(image_path, output_path)

    spots_df = apply_angle_correction(output_path, image_path)

    print(f"Results saved to {output_folder}")


def match_spots_to_nuclei(
        output_path: Path | str,
        image_path: Path | str,
        pixel_size_xy: float = 1.0,
        pixel_size_z: float = 1.0,
) -> pd.DataFrame:
    """Assign spots to nuclei and compute polar coordinates.

    Parameters
    ----------
    output_path : Path or str
        Root output directory containing segmentation and spot files.
    image_path : Path or str
        Path to the original image (used to locate the output sub-folder).
    pixel_size_xy : float
        Physical pixel size in XY dimensions (e.g. microns per pixel).
    pixel_size_z : float
        Physical pixel size in Z dimension (e.g. microns per pixel).

    Returns
    -------
    pd.DataFrame
        Spots DataFrame enriched with ``nuclei_index``, ``dists``, ``r``,
        ``theta``, ``source_file``, and ``source_path`` columns.
    """
    print("Computing distances from spots to nuclei...")
    
    nuclei_labels, cell_labels, spots_df, nuclei_df = data_loader(output_path, image_path)
    do_3D = True if nuclei_labels.ndim == 3 else False
    spots_df_temp = spots_df.copy()
    spots_df_temp = point_to_nucleus2D(
        spots_df=spots_df_temp,
        nuclei_label_origscale=nuclei_labels,
        cell_label=cell_labels,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        do_3D=do_3D)
    spots_df_temp = compute_polar_coordinates(spots_df_temp, nuclei_df)
    
    spots_df_temp["source_file"] = image_path.name
    spots_df_temp["source_path"] = str(image_path.parent)

    export_df_to_csv(spots_df_temp, output_path, image_path, "spots")

    return spots_df_temp

def data_loader(
    output_folder: Path | str,
    image_path: Path | str,
    only_df: bool = False,
) -> tuple:
    """Load segmentation masks and CSV data from the output folder.

    Parameters
    ----------
    output_folder : Path or str
        Root output directory.
    image_path : Path or str
        Path to the original image (used to locate the per-image sub-folder).
    only_df : bool
        If ``True``, skip loading TIFF masks and return only DataFrames.

    Returns
    -------
    tuple
        ``(spots_df, nuclei_df)`` when *only_df* is ``True``, otherwise
        ``(nuclei_labels, cell_labels, spots_df, nuclei_df)``.
    """
    output_folder = Path(output_folder)
    image_path = Path(image_path)
    stem = image_path.stem
    parent_name = image_path.parent.name
    nuclei_label_path = output_folder / parent_name / f"{stem}_nuclei.tiff"
    cell_label_path = output_folder / parent_name / f"{stem}_cells.tiff"
    spots_csv_path = output_folder / parent_name / f"{stem}_spots.csv"
    nuclei_csv_path = output_folder / parent_name / f"{stem}_statistics.csv"

    if not only_df:
        nuclei_labels = tifffile.imread(str(nuclei_label_path)).astype(int) if nuclei_label_path.exists() else None
        cell_labels = tifffile.imread(str(cell_label_path)).astype(int) if cell_label_path.exists() else None
    spots_df = pd.read_csv(spots_csv_path) if spots_csv_path.exists() else None
    nuclei_df = pd.read_csv(nuclei_csv_path) if nuclei_csv_path.exists() else None

    if only_df:
        return spots_df, nuclei_df
    else:
        return nuclei_labels, cell_labels, spots_df, nuclei_df
    
def export_df_to_csv(df: pd.DataFrame, output_path: Path, image_path: Path, file_suffix: str) -> None:
    """Helper function to save a DataFrame to CSV in the appropriate output sub-folder.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : Path
        Root output directory.
    image_path : Path
        Path to the original image.
    file_suffix : str
        Suffix for the output CSV file.
    """
    parent_name = image_path.parent.name
    stem = image_path.stem
    out_folder = output_path / parent_name
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"{stem}_{file_suffix}.csv"
    df.to_csv(out_path, index=False)

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_cells2D(
    image_data: np.ndarray,
    cell_proba: float,
    cell_channel: int | None = None,
    nucl_channel: int = 1,
    diameter_nucl: int = 30,
    diameter_cell: int = 50,
    scaling_factor: int = 4,
    xy_z_factor: float = 1.0,
    do_3D: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Segment nuclei (and optionally cells) using Cellpose.

    Parameters
    ----------
    image_data : np.ndarray, shape (C, H, W)
        Multi-channel 2-D image array.
    cell_proba : float
        Cellpose cell-probability threshold.
    cell_channel : int or None
        Channel index for cell segmentation, or ``None`` to skip.
    nucl_channel : int
        Channel index for nucleus segmentation.
    diameter_nucl : int
        Expected nucleus diameter in pixels (used for Cellpose v3 and below).
    diameter_cell : int
        Expected cell diameter in pixels (used for Cellpose v3 and below).
    scaling_factor : int
        Downscaling factor applied before Cellpose; labels are upscaled back.
    xy_z_factor : float
        Factor for converting XY coordinates to Z coordinates (or vice versa).
        xy_z_factor = voxel_size_xy / voxel_size_z.  Only used if *do_3D* is ``True``.
    do_3D : bool
        If ``True``, process the entire 3D stack instead of a single plane.

    Returns
    -------
    nuclei_label_origscale : np.ndarray
        Integer label image of nuclei at the original resolution.
    cell_label_origscale : np.ndarray or None
        Integer label image of cells at the original resolution, or ``None``
        if *cell_channel* was not provided.
    """
    from cellpose import models
    import cellpose

    diameter_nucl = diameter_nucl // scaling_factor
    diameter_cell = diameter_cell // scaling_factor

    cellpose_version = int(cellpose.version.split(".")[0])
    print(f"Using Cellpose version {cellpose.version} (major={cellpose_version})")

    if not do_3D:
        im_nucl = image_data[nucl_channel, ::scaling_factor, ::scaling_factor]
    else:
        z_scaling = int(np.floor(scaling_factor * xy_z_factor))
        if z_scaling == 0:
            z_scaling = 1
        im_nucl = image_data[nucl_channel, ::z_scaling, ::scaling_factor, ::scaling_factor]
    im_nucl_gauss = skimage.filters.gaussian(im_nucl, sigma=2, preserve_range=True)

    cell_label_origscale = None

    if cellpose_version > 3:
        
        model = models.CellposeModel(gpu=True)
        out = model.eval(im_nucl_gauss, cellprob_threshold=cell_proba, do_3D=do_3D, z_axis=0)
        nuclei_label = out[0]
        nuclei_label_origscale = skimage.transform.resize(
            nuclei_label, output_shape=image_data.shape[1:], order=0
        )

        if cell_channel is not None:
            if not do_3D:
                im_cell = image_data[cell_channel, ::scaling_factor, ::scaling_factor]
            else:
                im_cell = image_data[cell_channel, ::z_scaling, ::scaling_factor, ::scaling_factor]
            im_cell_gauss = skimage.filters.gaussian(im_cell, sigma=2, preserve_range=True)
            
            out_cell = model.eval(im_cell_gauss, cellprob_threshold=cell_proba, do_3D=do_3D, z_axis=0)
            cell_label = out_cell[0]
            cell_label_origscale = skimage.transform.resize(
                cell_label, output_shape=image_data.shape[1:], order=0
            )
    else:
        model = models.Cellpose(gpu=True, model_type="nuclei")
        nuclei_label = model.eval(im_nucl_gauss, cellprob_threshold=cell_proba, diameter=diameter_nucl, do_3D=do_3D, z_axis=0)[0]
        final_shape = image_data.shape[1:] # (H, W) or (Z, H, W)
        nuclei_label_origscale = skimage.transform.resize(
            nuclei_label, output_shape=final_shape, order=0
        ).astype(np.int16)
        if cell_channel is not None:
            if not do_3D:
                im_cell = image_data[[cell_channel, nucl_channel], ::scaling_factor, ::scaling_factor]
            else:
                im_cell = image_data[[cell_channel, nucl_channel], ::z_scaling, ::scaling_factor, ::scaling_factor]
            im_cell_gauss = skimage.filters.gaussian(im_cell, sigma=2, preserve_range=True, channel_axis=0)
            model_cell = models.Cellpose(gpu=True, model_type="cyto3")
            cell_label = model_cell.eval(im_cell_gauss, channels=[1,2], diameter=diameter_cell, cellprob_threshold=cell_proba, do_3D=do_3D, z_axis=1)[0]
            cell_label_origscale = skimage.transform.resize(
                cell_label, output_shape=final_shape, order=0
            ).astype(np.int16)
    
    return nuclei_label_origscale, cell_label_origscale

def compute_nuclei_stats(nuclei_label: np.ndarray) -> pd.DataFrame:
    """Compute region properties for each labelled nucleus.

    Parameters
    ----------
    nuclei_label : np.ndarray
        Integer label image of nuclei.

    Returns
    -------
    pd.DataFrame
        One row per nucleus with columns ``nucleus_id``, ``area``,
        ``cm-y``, and ``cm-x``.
    """
    nuclei_stats = skimage.measure.regionprops_table(nuclei_label, properties=["label", "area", "centroid"])
    nuclei_stats_df = pd.DataFrame(nuclei_stats)
    nuclei_stats_df.rename(
        columns={
            "label": "nucleus_id",
            "centroid-0": "cm-y",
            "centroid-1": "cm-x",
            "centroid-2": "cm-z"
            }, inplace=True)
    return nuclei_stats_df

def segment_spots2D(image_data: np.ndarray, use_cuda: bool = True, do_3D: bool = False) -> pd.DataFrame:
    """Detect spots in *image_data* using Spotiflow.

    Parameters
    ----------
    image_data : np.ndarray, shape (H, W)
        2-D image array.
    use_cuda : bool
        Use GPU inference if available; falls back to CPU automatically.
    do_3D : bool
        If ``True``, process the entire 3D stack instead of a single plane.

    Returns
    -------
    pd.DataFrame
        One row per detected spot with columns ``x``, ``y``, (``z``), ``prob``,
        and ``intens``.
    """
    from spotiflow.model import Spotiflow
    import torch

    if use_cuda and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        use_cuda = False

    im_spots = image_data
    if do_3D:
        model_spot = Spotiflow.from_pretrained(
            "smfish_3d", map_location="cuda" if use_cuda else "cpu"
        )
    else:
        model_spot = Spotiflow.from_pretrained(
            "general", map_location="cuda" if use_cuda else "cpu"
        )
    spots, details = model_spot.predict(im_spots, subpix=True, min_distance=3)
    spots_df = pd.DataFrame(spots)
    spots_df["prob"] = details.prob
    spots_df["intens"] = details.intens
    if do_3D:
        spots_df.rename(columns={0: "z", 1: "x", 2: "y"}, inplace=True)
    else:
        spots_df.rename(columns={0: "x", 1: "y"}, inplace=True)
    return spots_df


# ---------------------------------------------------------------------------
# Distance measurement
# ---------------------------------------------------------------------------

def point_to_nucleus2D(
    spots_df: pd.DataFrame,
    nuclei_label_origscale: np.ndarray,
    cell_label: np.ndarray | None = None,
    pixel_size_xy: float = 1.0,
    pixel_size_z: float = 1.0,
    do_3D: bool = False,
) -> pd.DataFrame:
    """Assign each spot to its nearest nucleus and measure the distance.

    Parameters
    ----------
    spots_df : pd.DataFrame
        Spots DataFrame with columns ``x`` and ``y``.
    nuclei_label_origscale : np.ndarray
        Integer label image of nuclei at the original resolution.
    cell_label : np.ndarray or None
        Integer label image of cells.  If ``None``, nuclei are expanded to
        act as cell proxies.
    pixel_size_xy : float
        Physical pixel size in XY dimensions (e.g. microns per pixel).
    pixel_size_z : float
        Physical pixel size in Z dimension (e.g. microns per pixel).
    do_3D : bool
        If ``True``, process the entire 3D stack instead of a single plane.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns ``cell_index``, ``nuclei_index``,
        and ``dists``.
    """
    spots_df_temp = spots_df.copy()
    if cell_label is None:
        expand_by = nuclei_label_origscale.shape[1] // 2
        cell_label = skimage.segmentation.expand_labels(nuclei_label_origscale, distance=expand_by)
        c2n = {i: i for i in np.unique(nuclei_label_origscale) if i != 0}
    else:
        c2n = assign_cell_to_nucleus(nuclei_label_origscale, cell_label)

    coords = ["z", "x", "y"] if do_3D else ["x", "y"]
    coords_spots = spots_df_temp[coords].values
    coords_spots_int = coords_spots.astype(np.int16)
    if do_3D:
        spots_cell_index = cell_label[coords_spots_int[:, 0], coords_spots_int[:, 1], coords_spots_int[:, 2]]
    else:
        spots_cell_index = cell_label[coords_spots_int[:, 0], coords_spots_int[:, 1]]

    spots_df_temp["cell_index"] = spots_cell_index
    spots_df_temp["nuclei_index"] = 0
    spots_df_temp["dists"] = np.nan

    # compute hollow masks for each nucleus to get accurate distance from border
    nucl_hollow = make_mask_hollow(nuclei_label_origscale)

    # pre-compute distance maps
    dist_maps = {}
    for n in np.unique(nuclei_label_origscale):
        if n!=0:
            dist_maps[n] = distance_map_to_label(nucl_hollow, n, pixel_size_xy, pixel_size_z)


    for idx, spot in spots_df_temp.iterrows():
        cell_id = spot["cell_index"]
        x, y = int(spot["x"]), int(spot["y"])
        if do_3D:
            x, y, z = int(spot["x"]), int(spot["y"]), int(spot["z"])
        if cell_id == 0:
            spots_df_temp.at[idx, "nuclei_index"] = 0
        else:
            nuclei_of_cell = c2n.get(cell_id, 0)
            if not isinstance(nuclei_of_cell, list):
                nuclei_of_cell = [nuclei_of_cell]
            # If multiple nuclei assigned to the same cell, pick the closest one
            min_dist = np.inf
            closest_nucleus = 0
            for nuc_id in nuclei_of_cell:
                if nuc_id == 0:
                    continue
                point = (z, x, y) if do_3D else (x, y)
                dist = distance_point_to_label(
                    point=point,
                    labels=nuclei_label_origscale,
                    label_id=nuc_id,
                    dist_map=dist_maps[nuc_id],
                    pixel_size_xy=pixel_size_xy,
                    pixel_size_z=pixel_size_z,
                    do_3D=do_3D
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_nucleus = nuc_id
                    if do_3D:
                        if nuclei_label_origscale[z, x, y] == nuc_id:
                            min_dist = -min_dist  # inside the nucleus, distance is negative
                    else:
                        if nuclei_label_origscale[x, y] == nuc_id:
                            min_dist = -min_dist  # inside the nucleus, distance is negative
            spots_df_temp.at[idx, "nuclei_index"] = closest_nucleus
            spots_df_temp.at[idx, "dists"] = min_dist if closest_nucleus != 0 else np.nan
    
    return spots_df_temp

def make_mask_hollow(labelled_mask: np.ndarray) -> np.ndarray:
    """Make a labelled mask hollow by eroding and subtracting.

    Parameters
    ----------
    mask : np.ndarray
        Labelled mask to hollow.

    Returns
    -------
    np.ndarray
        Hollowed labelled mask.
    """
    shrink = skimage.morphology.erosion(labelled_mask)
    nucl_hollow = np.logical_xor(labelled_mask, shrink)
    nucl_hollow = nucl_hollow * labelled_mask
    return nucl_hollow

def compute_polar_coordinates(spots_df: pd.DataFrame, nuclei_df: pd.DataFrame) -> pd.DataFrame:
    """Add polar coordinates relative to nucleus centroids.

    Parameters
    ----------
    spots_df : pd.DataFrame
        Spots DataFrame with ``nuclei_index``, ``x``, and ``y`` columns.
    nuclei_df : pd.DataFrame
        Nuclei statistics DataFrame with ``nucleus_id``, ``cm-y``, and
        ``cm-x`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *spots_df* with added ``r`` and ``theta`` columns.
    """
    spots_df_new = spots_df.copy()
    spots_df_new["r"] = np.nan
    spots_df_new["theta"] = np.nan
    assigned = spots_df_new[spots_df_new["nuclei_index"] != 0]
    for nid, group in assigned.groupby("nuclei_index"):
        nucleus_row = nuclei_df[nuclei_df["nucleus_id"] == nid].iloc[0]
        cy, cx = nucleus_row["cm-y"], nucleus_row["cm-x"]
        center = np.array([cy, cx])
        r, theta = cartesian_to_polar(group[["x", "y"]].values, center)
        spots_df_new.loc[group.index, "r"] = r
        spots_df_new.loc[group.index, "theta"] = theta
    return spots_df_new


def apply_angle_correction(
    output_folder: Path | str,
    image_path: Path | str,
) -> pd.DataFrame:
    """Add a ``theta_corrected`` column to the spots DataFrame for one image.

    For each nucleus the ``major_axis_angle`` from the statistics CSV is
    subtracted from ``theta`` so that corrected angles are relative to the
    nucleus orientation.

    Parameters
    ----------
    output_folder : Path or str
        Root output directory containing the per-image sub-folder.
    image_path : Path or str
        Path to the original image (used to locate output files).

    Returns
    -------
    pd.DataFrame
        Spots DataFrame with a new ``theta_corrected`` column (modulo 2π).
    """
    
    spots_df, nuclei_df = data_loader(output_folder, image_path, only_df=True)

    nid = spots_df["nuclei_index"].dropna().unique()
    nid = [int(i) for i in nid if int(i) > 0]
    for n in nid:
        if n not in nuclei_df["nucleus_id"].values:
            print(f"Warning: nucleus_id {n} not found in nuclei_df")
        spots_df.loc[spots_df["nuclei_index"] == n, "theta_corrected"] = (
            spots_df.loc[spots_df["nuclei_index"] == n, "theta"].values
            - nuclei_df.loc[nuclei_df["nucleus_id"] == n, "major_axis_angle"].values[0]
        ) % (2 * np.pi)

    export_df_to_csv(spots_df, output_folder, image_path, "spots")

    return spots_df

def compute_statistics_per_image(image_path: Path, output_folder: Path) -> pd.DataFrame:
    """Compute asymmetry statistics for all nuclei in a processed image.

    Parameters
    ----------
    image_path : Path
        Path to the original image (used to locate the output sub-folder).
    output_folder : Path
        Root output directory containing ``*_spots.csv`` and
        ``*_statistics.csv`` files.

    Returns
    -------
    pd.DataFrame
        Nuclei statistics DataFrame merged with per-nucleus asymmetry metrics.
    """
    from napari_cellspots._quantification import compact_asymmetry_metrics

    spots_df, nuclei_df = data_loader(output_folder, image_path, only_df=True)

    rows = []
    nucleus_ids = sorted(spots_df["nuclei_index"].dropna().unique())
    nucleus_ids = [int(i) for i in nucleus_ids if int(i) > 0]
    for nid in nucleus_ids:
        group = spots_df[spots_df["nuclei_index"] == nid]
        points = group[["x", "y"]].values
        r = group["r"].values
        theta = group["theta"].values
        metrics = compact_asymmetry_metrics(points, r, theta)
        rows.append({"nucleus_id": nid, "n_spots": len(group), **metrics})

    stats_df = pd.DataFrame(rows)
    nuclei_df = nuclei_df.merge(stats_df, on="nucleus_id", how="left")

    export_df_to_csv(nuclei_df, output_folder, image_path, "statistics")

    return nuclei_df

def distance_point_to_label(
    point: tuple[int, int],
    labels: np.ndarray,
    label_id: int,
    dist_map: np.ndarray | None = None,
    pixel_size_xy: float = 1.0,
    pixel_size_z: float = 1.0,
    do_3D: bool = False,
) -> float:
    """Return the distance from *point* to the nearest pixel of *label_id*.

    Parameters
    ----------
    point : tuple of int
        ``(row, col)`` or ``(z, row, col)`` coordinates of the query point.
    labels : np.ndarray
        Integer label image.
    label_id : int
        Target label.
    dist_map : np.ndarray or None
        Pre-computed distance map for *label_id*; computed on-the-fly if
        ``None``.
    pixel_size_xy : float
        Physical pixel size in XY dimensions (e.g. microns per pixel).
    pixel_size_z : float
        Physical pixel size in Z dimension (e.g. microns per pixel).
    do_3D : bool
        If ``True``, process the entire 3D stack instead of a single plane.

    Returns
    -------
    float
        Euclidean distance in pixels, or ``nan`` if *point* is out of bounds.
    """
    if dist_map is None:
        dist_map = distance_map_to_label(labels, label_id, pixel_size_xy, pixel_size_z)
    if do_3D:
        z, x, y = int(point[0]), int(point[1]), int(point[2])
        if 0 <= x < dist_map.shape[1] and 0 <= y < dist_map.shape[2] and 0 <= z < dist_map.shape[0]:
            dist = dist_map[z, x, y]
        else:
            dist = np.nan
    else:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < dist_map.shape[0] and 0 <= y < dist_map.shape[1]:
            dist = dist_map[x, y]
        else:
            dist = np.nan
    return dist


def distance_map_to_label(
        labels: np.ndarray,
        label_id: int,
        pixel_size_xy: float = 1.0,
        pixel_size_z: float = 1.0
) -> np.ndarray:
    """Compute the Euclidean distance transform to *label_id*.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image.
    label_id : int
        Target label.
    pixel_size_xy : float
        Pixel size in XY dimensions.
    pixel_size_z : float
        Pixel size in Z dimension.

    Returns
    -------
    np.ndarray
        Array of the same shape as *labels* where each pixel holds its
        distance to the nearest pixel of *label_id*.
    """
    mask = labels == label_id
    if mask.ndim == 3:
        spacing = (pixel_size_z, pixel_size_xy, pixel_size_xy)
    else:
        spacing = (pixel_size_xy, pixel_size_xy)
    return ndi.distance_transform_edt(~mask, sampling=spacing)


# ---------------------------------------------------------------------------
# Assign nuclei and spots to cell
# ---------------------------------------------------------------------------

def assign_cell_to_nucleus(
    nuclei_labels: np.ndarray,
    cell_labels: np.ndarray,
) -> dict[int, int | list[int]]:
    """Build a mapping from cell ID to nucleus ID(s).

    Parameters
    ----------
    nuclei_labels : np.ndarray
        Integer label image of nuclei.
    cell_labels : np.ndarray
        Integer label image of cells.

    Returns
    -------
    dict
        ``{cell_id: nucleus_id}`` or ``{cell_id: [nucleus_id, ...]}`` when
        multiple nuclei share the same cell.
    """
    cellid_to_nucleid = {}
    for nuclei_id in np.unique(nuclei_labels):
        if nuclei_id == 0:
            continue
        cell_id = np.bincount(cell_labels[nuclei_labels == nuclei_id].flatten()).argmax()
        if cell_id in cellid_to_nucleid.keys():
            if isinstance(cellid_to_nucleid[cell_id], list):
                cellid_to_nucleid[cell_id].append(nuclei_id)
            else:
                cellid_to_nucleid[cell_id] = [cellid_to_nucleid[cell_id], nuclei_id]
        else:
            cellid_to_nucleid[cell_id] = nuclei_id
    return cellid_to_nucleid


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def cartesian_to_polar(
    points: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar relative to *center*.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Array of ``(row, col)`` coordinates.
    center : np.ndarray, shape (2,)
        ``(row, col)`` origin for the polar transform.

    Returns
    -------
    r : np.ndarray, shape (N,)
        Radii.
    theta : np.ndarray, shape (N,)
        Angles in [0, 2π).
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    dy = points[:, 0] - center[0]
    dx = points[:, 1] - center[1]
    r = np.hypot(dy, dx)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return r, theta
