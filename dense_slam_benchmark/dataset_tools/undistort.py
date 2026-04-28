import numpy as np

try:
    import pycolmap
except ImportError:  # pragma: no cover - optional at import time
    pycolmap = None


def _as_points(x, y):
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _intrinsics_to_K(intrinsics):
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics
    intrinsics = intrinsics.reshape(-1)
    if intrinsics.size != 4:
        raise ValueError(
            f"Expected intrinsics to be either a 3x3 matrix or [fx, fy, cx, cy], got {intrinsics}"
        )
    fx, fy, cx, cy = intrinsics.tolist()
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def distort_thin_prism_fisheye_normalized(x, y, distortion_coeffs):
    """
    ETH3D / COLMAP THIN_PRISM_FISHEYE model.

    Args:
        x, y:
            Coordinates on the virtual pinhole image plane.
        distortion_coeffs:
            [k1, k2, p1, p2, k3, k4, sx1, sy1]

    Returns:
        Distorted normalized image coordinates (u_n, v_n).
    """
    x, y = _as_points(x, y)
    k1, k2, p1, p2, k3, k4, sx1, sy1 = np.asarray(
        distortion_coeffs, dtype=np.float32
    ).reshape(-1)

    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)

    scale = np.ones_like(r, dtype=np.float32)
    nonzero = r > 1e-8
    scale[nonzero] = theta[nonzero] / r[nonzero]

    u_d = x * scale
    v_d = y * scale

    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4

    t_r = 1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8

    u_n = (
        u_d * t_r
        + 2.0 * p1 * u_d * v_d
        + p2 * (theta2 + 2.0 * u_d * u_d)
        + sx1 * theta2
    )
    v_n = (
        v_d * t_r
        + 2.0 * p2 * u_d * v_d
        + p1 * (theta2 + 2.0 * v_d * v_d)
        + sy1 * theta2
    )

    return u_n.astype(np.float32), v_n.astype(np.float32)


def undistort_thin_prism_fisheye_normalized(
    u_n, v_n, distortion_coeffs, max_iterations=25, tolerance=1e-10
):
    """
    Numerically invert the ETH3D / COLMAP THIN_PRISM_FISHEYE model.

    Returns:
        x, y, valid
        where x, y are coordinates on the virtual pinhole image plane.
    """
    u_n, v_n = _as_points(u_n, v_n)
    x = u_n.astype(np.float64).copy()
    y = v_n.astype(np.float64).copy()
    target_u = u_n.astype(np.float64)
    target_v = v_n.astype(np.float64)

    eps = 1e-6
    valid = np.ones_like(target_u, dtype=bool)

    for _ in range(max_iterations):
        pred_u, pred_v = distort_thin_prism_fisheye_normalized(x, y, distortion_coeffs)
        pred_u = pred_u.astype(np.float64)
        pred_v = pred_v.astype(np.float64)

        err_u = pred_u - target_u
        err_v = pred_v - target_v
        if np.max(np.abs(err_u)) < tolerance and np.max(np.abs(err_v)) < tolerance:
            break

        pred_u_dx, pred_v_dx = distort_thin_prism_fisheye_normalized(x + eps, y, distortion_coeffs)
        pred_u_dy, pred_v_dy = distort_thin_prism_fisheye_normalized(x, y + eps, distortion_coeffs)

        j11 = (pred_u_dx.astype(np.float64) - pred_u) / eps
        j12 = (pred_u_dy.astype(np.float64) - pred_u) / eps
        j21 = (pred_v_dx.astype(np.float64) - pred_v) / eps
        j22 = (pred_v_dy.astype(np.float64) - pred_v) / eps

        det = j11 * j22 - j12 * j21
        good = np.isfinite(det) & (np.abs(det) > 1e-12)
        valid &= good
        if not np.any(good):
            break

        step_x = np.zeros_like(x)
        step_y = np.zeros_like(y)
        step_x[good] = (-j22[good] * err_u[good] + j12[good] * err_v[good]) / det[good]
        step_y[good] = (j21[good] * err_u[good] - j11[good] * err_v[good]) / det[good]

        step_norm = np.sqrt(step_x * step_x + step_y * step_y)
        large = step_norm > 1.0
        if np.any(large):
            scale = 1.0 / step_norm[large]
            step_x[large] *= scale
            step_y[large] *= scale

        x += step_x
        y += step_y

    pred_u, pred_v = distort_thin_prism_fisheye_normalized(x, y, distortion_coeffs)
    residual = np.sqrt(
        (pred_u.astype(np.float64) - target_u) ** 2 + (pred_v.astype(np.float64) - target_v) ** 2
    )
    valid &= np.isfinite(residual) & (residual < 1e-5)

    return x.astype(np.float32), y.astype(np.float32), valid


def estimate_thin_prism_fisheye_new_intrinsics(
    resolution, intrinsics, distortion_coeffs
):
    image_w, image_h = resolution
    fx, fy, cx, cy = intrinsics

    top_u = np.arange(image_w, dtype=np.float32)
    top_v = np.zeros(image_w, dtype=np.float32)

    bottom_u = np.arange(image_w, dtype=np.float32)
    bottom_v = np.full(image_w, image_h - 1, dtype=np.float32)

    side_v = np.arange(1, image_h - 1, dtype=np.float32)
    left_u = np.zeros_like(side_v)
    right_u = np.full_like(side_v, image_w - 1)

    border_u = np.concatenate([top_u, bottom_u, left_u, right_u])
    border_v = np.concatenate([top_v, bottom_v, side_v, side_v])

    distorted_u_n = (border_u - cx) / fx
    distorted_v_n = (border_v - cy) / fy
    x, y, valid = undistort_thin_prism_fisheye_normalized(
        distorted_u_n, distorted_v_n, distortion_coeffs
    )
    if not np.any(valid):
        raise ValueError("Failed to invert THIN_PRISM_FISHEYE border pixels")

    x_valid = x[valid]
    y_valid = y[valid]
    max_r = float(np.max(np.sqrt(x_valid * x_valid + y_valid * y_valid)))

    x_min = float(np.min(x_valid))
    x_max = float(np.max(x_valid))
    y_min = float(np.min(y_valid))
    y_max = float(np.max(y_valid))

    span_x = max(x_max - x_min, 1e-8)
    span_y = max(y_max - y_min, 1e-8)

    new_fx = (image_w - 1) / span_x
    new_fy = (image_h - 1) / span_y
    new_cx = -x_min * new_fx
    new_cy = -y_min * new_fy

    newK = np.array(
        [[new_fx, 0.0, new_cx], [0.0, new_fy, new_cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return newK, max_r


def create_thin_prism_fisheye_undistort_map(
    resolution,
    intrinsics,
    distortion_coeffs,
    new_intrinsics=None,
    target_resolution=None,
):
    """
    Build remap tables that convert a distorted THIN_PRISM_FISHEYE image into an
    undistorted pinhole image, following the ETH3D/COLMAP camera model.
    """
    image_w, image_h = resolution
    fx, fy, cx, cy = intrinsics

    if new_intrinsics is None:
        newK, max_r = estimate_thin_prism_fisheye_new_intrinsics(
            resolution, intrinsics, distortion_coeffs
        )
        target_w, target_h = image_w, image_h
    else:
        newK = _intrinsics_to_K(new_intrinsics)
        if target_resolution is None:
            target_w, target_h = image_w, image_h
        else:
            target_w, target_h = target_resolution
        border_u = np.array([0, image_w - 1, 0, image_w - 1], dtype=np.float32)
        border_v = np.array([0, 0, image_h - 1, image_h - 1], dtype=np.float32)
        distorted_u_n = (border_u - cx) / fx
        distorted_v_n = (border_v - cy) / fy
        x, y, valid = undistort_thin_prism_fisheye_normalized(
            distorted_u_n, distorted_v_n, distortion_coeffs
        )
        if not np.any(valid):
            raise ValueError("Failed to invert THIN_PRISM_FISHEYE border pixels")
        max_r = float(np.max(np.sqrt(x[valid] * x[valid] + y[valid] * y[valid])))

    new_fx = float(newK[0, 0])
    new_fy = float(newK[1, 1])
    new_cx = float(newK[0, 2])
    new_cy = float(newK[1, 2])

    grid_u, grid_v = np.meshgrid(
        np.arange(target_w, dtype=np.float32),
        np.arange(target_h, dtype=np.float32),
    )
    x = (grid_u - new_cx) / new_fx
    y = (grid_v - new_cy) / new_fy
    r = np.sqrt(x * x + y * y)

    distorted_u_n, distorted_v_n = distort_thin_prism_fisheye_normalized(
        x, y, distortion_coeffs
    )
    map_x = fx * distorted_u_n + cx
    map_y = fy * distorted_v_n + cy

    valid = (
        (r <= max_r)
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= image_w - 1)
        & (map_y >= 0.0)
        & (map_y <= image_h - 1)
    )

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return newK, map_x, map_y


def create_colmap_thin_prism_fisheye_undistort_map(
    resolution,
    intrinsics,
    distortion_coeffs,
    new_intrinsics=None,
    target_resolution=None,
):
    """
    Build remap tables using the exact THIN_PRISM_FISHEYE camera model from pycolmap.
    """
    if pycolmap is None:
        raise ImportError("pycolmap is required for exact THIN_PRISM_FISHEYE undistortion")

    image_w, image_h = resolution
    fx, fy, cx, cy = intrinsics
    distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)

    distorted_camera = pycolmap.Camera(
        model="THIN_PRISM_FISHEYE",
        width=int(image_w),
        height=int(image_h),
        params=np.array([fx, fy, cx, cy, *distortion_coeffs.tolist()], dtype=np.float64),
    )

    top_u = np.arange(image_w, dtype=np.float64)
    top_v = np.zeros(image_w, dtype=np.float64)
    bottom_u = np.arange(image_w, dtype=np.float64)
    bottom_v = np.full(image_w, image_h - 1, dtype=np.float64)
    side_v = np.arange(1, image_h - 1, dtype=np.float64)
    left_u = np.zeros_like(side_v)
    right_u = np.full_like(side_v, image_w - 1, dtype=np.float64)

    border_u = np.concatenate([top_u, bottom_u, left_u, right_u])
    border_v = np.concatenate([top_v, bottom_v, side_v, side_v])
    border_img_pts = np.stack([border_u, border_v], axis=1)
    border_cam_pts = distorted_camera.cam_from_img(border_img_pts)
    valid_border = np.all(np.isfinite(border_cam_pts), axis=1)
    if not np.any(valid_border):
        raise ValueError("Failed to invert THIN_PRISM_FISHEYE border pixels with pycolmap")

    border_cam_pts = border_cam_pts[valid_border]
    x_valid = border_cam_pts[:, 0]
    y_valid = border_cam_pts[:, 1]
    max_r = float(np.max(np.linalg.norm(border_cam_pts, axis=1)))

    if new_intrinsics is None:
        x_min = float(np.min(x_valid))
        x_max = float(np.max(x_valid))
        y_min = float(np.min(y_valid))
        y_max = float(np.max(y_valid))
        span_x = max(x_max - x_min, 1e-8)
        span_y = max(y_max - y_min, 1e-8)
        new_fx = (image_w - 1) / span_x
        new_fy = (image_h - 1) / span_y
        new_cx = -x_min * new_fx
        new_cy = -y_min * new_fy
        newK = np.array(
            [[new_fx, 0.0, new_cx], [0.0, new_fy, new_cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        target_w, target_h = image_w, image_h
    else:
        newK = _intrinsics_to_K(new_intrinsics)
        if target_resolution is None:
            target_w, target_h = image_w, image_h
        else:
            target_w, target_h = target_resolution

    new_fx = float(newK[0, 0])
    new_fy = float(newK[1, 1])
    new_cx = float(newK[0, 2])
    new_cy = float(newK[1, 2])

    grid_u, grid_v = np.meshgrid(
        np.arange(target_w, dtype=np.float64),
        np.arange(target_h, dtype=np.float64),
    )
    pinhole_cam_pts_xy = np.stack(
        [
            (grid_u - new_cx) / new_fx,
            (grid_v - new_cy) / new_fy,
        ],
        axis=-1,
    ).reshape(-1, 2)
    pinhole_cam_pts = np.concatenate(
        [pinhole_cam_pts_xy, np.ones((pinhole_cam_pts_xy.shape[0], 1), dtype=np.float64)],
        axis=1,
    )

    distorted_img_pts = distorted_camera.img_from_cam(pinhole_cam_pts)
    distorted_img_pts = distorted_img_pts.reshape(target_h, target_w, 2)

    pinhole_r = np.linalg.norm(pinhole_cam_pts_xy, axis=1).reshape(target_h, target_w)
    map_x = distorted_img_pts[..., 0].astype(np.float32)
    map_y = distorted_img_pts[..., 1].astype(np.float32)

    valid = (
        np.all(np.isfinite(distorted_img_pts), axis=-1)
        & (pinhole_r <= max_r)
        & (map_x >= 0.0)
        & (map_x <= image_w - 1)
        & (map_y >= 0.0)
        & (map_y <= image_h - 1)
    )
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return newK, map_x, map_y
