from __future__ import annotations
import numpy as np

class RelaxationConvergenceError(RuntimeError):
    pass

def solve_model(rad, model, *, relax, precision=1e-4, max_iter=1000, method=4):
    """
    RADIA relaxation:
      RlxPre(object) -> RlxAuto(interaction_matrix, precision, max_iter, method)

    RADIA returns [AvPrec, Mmax, Hmax, Niter]. A relaxation is accepted only
    when AvPrec < requested precision AND Niter < max_iter, following RADIA's
    documented convergence criteria.
    """
    info = None
    if relax:
        if model.get("material_mode") != "Linear NdFeB + relaxation":
            raise ValueError(
                "Relaxation requested, but the model is using fixed remanent magnetization."
            )
        intr = rad.RlxPre(model["obj"])
        info = rad.RlxAuto(intr, float(precision), int(max_iter), int(method))
        if not isinstance(info, (list, tuple)) or len(info) < 4:
            raise RelaxationConvergenceError(
                f"RADIA RlxAuto returned an unexpected result: {info!r}"
            )
        av_prec = float(info[0])
        n_iter = int(info[3])
        if not np.isfinite(av_prec):
            raise RelaxationConvergenceError(
                f"RADIA relaxation returned non-finite AvPrec={av_prec}."
            )
        if av_prec >= float(precision) or n_iter >= int(max_iter):
            raise RelaxationConvergenceError(
                "RADIA relaxation did not converge: "
                f"AvPrec={av_prec:.6g} T (required < {float(precision):.6g} T), "
                f"Niter={n_iter} (required < {int(max_iter)})."
            )
    return info

def sample_points(rad, obj, points_mm, *, chunk_size=10000):
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_mm must have shape (N, 3); got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Sampling coordinates must be finite.")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be at least 1.")
    out = np.empty((len(points), 3), dtype=float)
    for start in range(0, len(points), int(chunk_size)):
        stop = min(start + int(chunk_size), len(points))
        for i in range(start, stop):
            p = points[i]
            b = rad.Fld(obj, "B", [float(p[0]), float(p[1]), float(p[2])])
            if not isinstance(b, (list, tuple, np.ndarray)) or len(b) < 3:
                raise RuntimeError(f"RADIA Fld returned an invalid vector at point {p.tolist()}: {b!r}")
            out[i] = [float(b[0]), float(b[1]), float(b[2])]
    if not np.all(np.isfinite(out)):
        raise RuntimeError("RADIA returned a non-finite magnetic-field value.")
    return out

def sample_on_axis(rad, obj, z_mm):
    return sample_points(rad, obj, [[0.0, 0.0, float(z)] for z in z_mm])

def sample_slice_xz(rad, obj, x_mm, z_mm, y_mm=0.0):
    pts = [[float(x), float(y_mm), float(z)] for z in z_mm for x in x_mm]
    B = sample_points(rad, obj, pts)
    return B.reshape(len(z_mm), len(x_mm), 3)

def sample_slice_yz(rad, obj, y_mm, z_mm, x_mm=0.0):
    pts = [[float(x_mm), float(y), float(z)] for z in z_mm for y in y_mm]
    B = sample_points(rad, obj, pts)
    return B.reshape(len(z_mm), len(y_mm), 3)

def sample_3d(rad, obj, x_mm, y_mm, z_mm):
    pts = [[float(x), float(y), float(z)] for z in z_mm for y in y_mm for x in x_mm]
    B = sample_points(rad, obj, pts)
    return B.reshape((len(z_mm), len(y_mm), len(x_mm), 3))
