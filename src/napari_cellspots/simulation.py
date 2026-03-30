import numpy as np


def simulate_image(
    image_size_um: float = 100.0,
    pixel_size_um: float = 0.2,
    nucleus_diameter_um: float = 8.0,
    cell_diameter_um: float = 20.0,
    spot_diameter_um: float = 0.4,
    symmetry_factor: float = 0.0,
    n_cells: int = 10,
    n_spots_per_cell: int = 20,
    rng_seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic two-channel fluorescence image.

    Channel 0 — nuclei: filled Gaussian blobs representing cell nuclei.
    Channel 1 — spots: a diffuse halo around each nucleus (above background)
        with bright point-like spots whose angular distribution is controlled
        by ``symmetry_factor``.

    Parameters
    ----------
    image_size_um:
        Side length of the square image in micrometres.
    pixel_size_um:
        Physical size of one pixel in micrometres.
    nucleus_diameter_um:
        Average diameter of a nucleus in micrometres.
    cell_diameter_um:
        Average diameter of the cell (controls the halo radius around the
        nucleus and the exclusion zone used to place nuclei).
    spot_diameter_um:
        Average diameter of an individual spot in micrometres.
    symmetry_factor:
        Controls the angular distribution of spots around the nucleus centroid.
        0.0  → fully isotropic (uniform in angle).
        1.0  → fully diametrically opposite (all spots at two antipodal poles).
        Values in between interpolate smoothly between both extremes.
    n_cells:
        Number of cells (nuclei) to place.
    n_spots_per_cell:
        Number of spots placed around each nucleus.
    rng_seed:
        Optional seed for the random-number generator (for reproducibility).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(2, H, W)`` where ``H = W = int(image_size_um /
        pixel_size_um)``.  Values lie roughly in ``[0, 1]`` before noise.
    """
    rng = np.random.default_rng(rng_seed)
    n_px = int(round(image_size_um / pixel_size_um))

    # ── pixel-space sizes ──────────────────────────────────────────────────
    nucleus_sigma = (nucleus_diameter_um / pixel_size_um) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    cell_radius_px = (cell_diameter_um / pixel_size_um) / 2.0
    nucleus_radius_px = (nucleus_diameter_um / pixel_size_um) / 2.0
    spot_sigma = (spot_diameter_um / pixel_size_um) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spot_sigma = max(spot_sigma, 0.5)  # never smaller than half a pixel

    ch0 = np.zeros((n_px, n_px), dtype=np.float32)  # nuclei
    ch1 = np.zeros((n_px, n_px), dtype=np.float32)  # spots / halo

    # ── place nuclei with a simple rejection sampler ───────────────────────
    min_sep = cell_radius_px  # minimum centre-to-centre separation
    margin = cell_radius_px
    centres: list[tuple[float, float]] = []

    for _ in range(n_cells * 200):         # max attempts
        if len(centres) == n_cells:
            break
        cy = rng.uniform(margin, n_px - margin)
        cx = rng.uniform(margin, n_px - margin)
        if all(np.hypot(cy - y, cx - x) >= min_sep for y, x in centres):
            centres.append((cy, cx))

    # ── coordinate grids ──────────────────────────────────────────────────
    yy, xx = np.mgrid[0:n_px, 0:n_px].astype(np.float32)

    for cy, cx in centres:
        dy = yy - cy
        dx = xx - cx
        dist = np.hypot(dy, dx)

        # --- channel 0: nucleus blob ----------------------------------------
        # tanh profile: flat top inside the nucleus, sharp boundary
        edge_sigma = max(nucleus_sigma * 0.25, 0.8)
        nucleus_blob = 0.5 * (1.0 - np.tanh((dist - nucleus_radius_px) / edge_sigma))
        ch0 += nucleus_blob

        # --- channel 1: diffuse halo -----------------------------------------
        # Gaussian halo that covers the cytoplasmic area between nucleus edge
        # and cell edge; sigma chosen so the halo peaks at the nucleus boundary.
        halo_sigma = (cell_radius_px - nucleus_radius_px) / 2.0
        halo_sigma = max(halo_sigma, 1.0)
        halo = 0.25 * np.exp(-0.5 * ((dist - nucleus_radius_px) / halo_sigma) ** 2)
        # keep only the ring outside the nucleus and inside the cell
        halo *= (dist <= cell_radius_px).astype(np.float32)
        ch1 += halo

        # --- channel 1: spots ------------------------------------------------
        # for each cell pick a single angle to be used for diametrically opposite spots
        base_angle = rng.uniform(0.0, 2.0 * np.pi)

        for _ in range(n_spots_per_cell):
            # sample a radial distance uniformly in the cytoplasmic annulus
            r = rng.uniform(nucleus_radius_px * 1.1, cell_radius_px * 0.95)

            # sample angle: mix between uniform and a bipolar (two-peaked) dist
            u = rng.uniform(0.0, 1.0)
            if symmetry_factor <= 0.0 or u >= symmetry_factor:
                # isotropic component
                angle = rng.uniform(0.0, 2.0 * np.pi)
            else:
                # diametrically opposite: concentrate around a random angle and it's opposite +π
                base = rng.choice([base_angle, (base_angle + np.pi) % (2.0 * np.pi)])
                # von Mises concentration scales with symmetry_factor
                kappa = 10.0 * symmetry_factor
                angle = rng.vonmises(base, kappa)

            sy = cy + r * np.sin(angle)
            sx = cx + r * np.cos(angle)

            # bounds check
            if not (0 <= sy < n_px and 0 <= sx < n_px):
                continue

            # add a Gaussian spot
            spot = np.exp(
                -0.5 * (((yy - sy) / spot_sigma) ** 2 + ((xx - sx) / spot_sigma) ** 2)
            )
            ch1 += spot

    # ── add Poisson-like noise ────────────────────────────────────────────
    # Scale to a photon-count-like range, add Poisson noise, rescale back
    SCALE = 200.0
    ch0_counts = np.clip(ch0 * SCALE, 0, None)
    ch1_counts = np.clip(ch1 * SCALE, 0, None)

    ch0_noisy = rng.poisson(ch0_counts).astype(np.float32) / SCALE
    ch1_noisy = rng.poisson(ch1_counts).astype(np.float32) / SCALE

    return np.stack([ch0_noisy, ch1_noisy], axis=0)  # (2, H, W)
