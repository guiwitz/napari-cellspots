import numpy as np


def compact_asymmetry_metrics(
    points: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
) -> dict:
    """Compute asymmetry metrics for a set of spots relative to a nucleus.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Cartesian coordinates of spots ``(row, col)``.
    r : np.ndarray, shape (N,)
        Pre-computed radii from the nucleus centroid.
    theta : np.ndarray, shape (N,)
        Pre-computed angles from the nucleus centroid, in [0, 2π).

    Returns
    -------
    dict
        Dictionary with keys:

        mean_resultant_length : float
            0 → evenly distributed around center, 1 → all in one direction.
        circular_mean : float
            Angle of the "average" direction, in [0, 2π).
        radial_mean : float
            Mean radius normalised by max radius; higher → points near the edge.
        anisotropy : float
            0 → roughly circular spread, 1 → strongly elongated along one axis.
        major_axis_angle : float
            Angle of the major axis (eigenvector of largest eigenvalue), in [0, π).
    """
    # Resolve polar coordinates
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    points = np.asarray(points, dtype=float)
    if (len(r) == 0) or (points.shape[0] == 0):
        return {
            "mean_resultant_length": np.nan,
            "circular_mean": np.nan,
            "radial_mean": np.nan,
            "anisotropy": np.nan,
            "major_axis_angle": np.nan,
        }

    # 1) Angular asymmetry
    z = np.mean(np.exp(1j * theta))
    mean_resultant_length = np.abs(z)
    circular_mean = np.angle(z) % (2 * np.pi)

    # 2) Edge-vs-center tendency
    radial_mean = 0.0 if r.max() == 0 else np.mean(r / r.max())

    # 3) Shape anisotropy — requires cartesian points relative to center
    if len(points) >= 2:
        center = np.mean(points, axis=0)
        X = points - center
        C = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        anisotropy = 0.0 if eigvals[0] <= 0 else 1 - eigvals[1] / eigvals[0]
        # Angle of the major axis (eigenvector of largest eigenvalue), in [0, pi)
        vx, vy = eigvecs[1, 0], eigvecs[0, 0]
        major_axis_angle = np.mod(np.arctan2(vy, vx), np.pi)
    else:
        anisotropy = np.nan
        major_axis_angle = np.nan

    return {
        "mean_resultant_length": mean_resultant_length,
        "circular_mean": circular_mean,
        "radial_mean": radial_mean,
        "anisotropy": anisotropy,
        "major_axis_angle": major_axis_angle,
    }
