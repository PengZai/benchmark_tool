import argparse
import copy
import os

import numpy as np
import open3d as o3d


def draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
) -> None:
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    src.paint_uniform_color([1.0, 0.706, 0.0])
    tgt.paint_uniform_color([0.0, 0.651, 0.929])
    src.transform(transformation)
    o3d.visualization.draw_geometries([src, tgt])


def decompose_similarity(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Decompose a 4x4 similarity transform into uniform scale, rotation, and translation.
    The top-left 3x3 block is assumed to be close to scale * rotation.
    """
    linear = transform[:3, :3]
    translation = transform[:3, 3].copy()

    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular_values))
    if abs(scale) <= 1e-12:
        raise ValueError("Estimated scale is too close to zero.")

    rotation = linear / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = u @ vt

    return scale, rotation, translation


def make_scale_transform(scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    return transform


def estimate_initial_scale(source_pts: np.ndarray, target_pts: np.ndarray) -> float:
    source_norms = np.linalg.norm(source_pts, axis=1)
    target_norms = np.linalg.norm(target_pts, axis=1)

    source_norms = source_norms[source_norms > 1e-12]
    target_norms = target_norms[target_norms > 1e-12]
    if len(source_norms) == 0 or len(target_norms) == 0:
        raise ValueError("Point clouds are too close to the origin to estimate scale.")

    return float(np.median(target_norms) / np.median(source_norms))


def scale_alignment_error(
    scale: float,
    source_pts: np.ndarray,
    target_pcd: o3d.geometry.PointCloud,
) -> tuple[float, float]:
    scaled_source = o3d.geometry.PointCloud()
    scaled_source.points = o3d.utility.Vector3dVector(source_pts * scale)

    distances = np.asarray(scaled_source.compute_point_cloud_distance(target_pcd), dtype=np.float64)
    if len(distances) == 0:
        return float("inf"), float("inf")

    rmse = float(np.sqrt(np.mean(np.square(distances))))
    median = float(np.median(distances))
    return rmse, median


def estimate_scale_only(
    source_pts: np.ndarray,
    target_pcd: o3d.geometry.PointCloud,
    initial_scale: float,
    search_ratio: float,
    num_steps: int,
    refinement_rounds: int,
) -> tuple[float, float, float]:
    if initial_scale <= 0.0:
        raise ValueError("Initial scale must be positive.")
    if search_ratio <= 1.0:
        raise ValueError("search_ratio must be greater than 1.")
    if num_steps < 3:
        raise ValueError("num_steps must be at least 3.")

    best_scale = initial_scale
    best_rmse, best_median = scale_alignment_error(best_scale, source_pts, target_pcd)

    low = initial_scale / search_ratio
    high = initial_scale * search_ratio

    print(
        "Running scale-only alignment: "
        f"initial_scale={initial_scale:.6f}, search_ratio={search_ratio}, "
        f"num_steps={num_steps}, refinement_rounds={refinement_rounds}"
    )

    for round_idx in range(refinement_rounds):
        candidate_scales = np.linspace(low, high, num_steps, dtype=np.float64)
        print(
            f"  round {round_idx + 1}/{refinement_rounds}: "
            f"search_range=[{low:.6f}, {high:.6f}], "
            f"current_best_scale={best_scale:.6f}, current_best_rmse={best_rmse:.6f}"
        )
        for scale in candidate_scales:
            rmse, median = scale_alignment_error(float(scale), source_pts, target_pcd)
            if rmse < best_rmse:
                best_scale = float(scale)
                best_rmse = rmse
                best_median = median

        span = (high - low) / max(num_steps - 1, 1)
        low = max(best_scale - 2.0 * span, 1e-12)
        high = best_scale + 2.0 * span

    return best_scale, best_rmse, best_median


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float | None,
) -> o3d.geometry.PointCloud:
    out = copy.deepcopy(pcd)
    if voxel_size is not None and voxel_size > 0.0:
        out = out.voxel_down_sample(voxel_size)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate scale and transformation between two similar point clouds using ICP."
    )
    parser.add_argument(
        "--alignment-mode",
        choices=["scale_only", "similarity"],
        default="scale_only",
        help="Choose scale-only alignment or full scale+rotation+translation alignment.",
    )
    parser.add_argument(
        "--source",
        default="/mnt/lboro_nas/personal/Zhipeng/wai_data/BotanicGarden_1018_00_32_test_3/logs/dense_32_view/depth_enhancement_1/pc_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31.pcd",
        help="Path to source point cloud.",
    )
    parser.add_argument(
        "--target",
        default="/mnt/lboro_nas/personal/Zhipeng/wai_data/BotanicGarden_1018_00_32_test_3/livox_lidar_c10.pcd",
        help="Path to target point cloud.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/lboro_nas/personal/Zhipeng/wai_data/BotanicGarden_1018_00_32_test_3/colmap_sparse_point_in_left_rgb_coordinate_c0_aligned_with_livox_lidar_c10.pcd",
        help="Path to save aligned source point cloud.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.001,
        help="Optional downsample voxel size before ICP. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=0.1,
        help="Maximum correspondence distance for ICP in similarity mode.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of ICP iterations.",
    )
    parser.add_argument(
        "--relative-fitness",
        type=float,
        default=1e-6,
        help="Relative fitness convergence threshold.",
    )
    parser.add_argument(
        "--relative-rmse",
        type=float,
        default=1e-6,
        help="Relative RMSE convergence threshold.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the downsampled registration result.",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=1.0,
        help="Optional initial scale for scale-only mode. If omitted, it is estimated automatically.",
    )
    parser.add_argument(
        "--scale-search-ratio",
        type=float,
        default=1.3,
        help="Scale-only mode search range multiplier.",
    )
    parser.add_argument(
        "--scale-search-steps",
        type=int,
        default=30,
        help="Number of candidate scales evaluated per round in scale-only mode.",
    )
    parser.add_argument(
        "--scale-refinement-rounds",
        type=int,
        default=6,
        help="Number of coarse-to-fine rounds in scale-only mode.",
    )
    args = parser.parse_args()

    print(f"Loading source point cloud: {args.source}")
    print(f"Loading target point cloud: {args.target}")
    source = o3d.io.read_point_cloud(args.source)
    target = o3d.io.read_point_cloud(args.target)

    source_pts = np.asarray(source.points)
    target_pts = np.asarray(target.points)
    if len(source_pts) == 0 or len(target_pts) == 0:
        raise ValueError("Source or target point cloud is empty.")

    voxel_size = None if args.voxel_size <= 0.0 else args.voxel_size
    source_ds = preprocess_point_cloud(source, voxel_size)
    target_ds = preprocess_point_cloud(target, voxel_size)

    print(
        "Point cloud sizes: "
        f"source={len(source_pts)}, target={len(target_pts)}, "
        f"source_down={len(source_ds.points)}, target_down={len(target_ds.points)}, "
        f"voxel_size={voxel_size if voxel_size is not None else 0.0}"
    )

    print(f"Alignment mode: {args.alignment_mode}")

    if args.alignment_mode == "scale_only":
        source_ds_pts = np.asarray(source_ds.points)
        target_ds_pts = np.asarray(target_ds.points)

        initial_scale = args.init_scale
        if initial_scale is None:
            initial_scale = estimate_initial_scale(source_ds_pts, target_ds_pts)
            print(f"Estimated initial scale: {initial_scale:.6f}")
        else:
            print(f"Using provided initial scale: {initial_scale:.6f}")

        scale, rmse, median_distance = estimate_scale_only(
            source_ds_pts,
            target_ds,
            initial_scale=initial_scale,
            search_ratio=args.scale_search_ratio,
            num_steps=args.scale_search_steps,
            refinement_rounds=args.scale_refinement_rounds,
        )
        transform = make_scale_transform(scale)
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        fitness = float("nan")
        inlier_rmse = rmse
        extra_metric_label = "Median distance"
        extra_metric_value = median_distance
    else:
        init = np.eye(4, dtype=np.float64)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        )

        print(
            "Running ICP: "
            f"max_correspondence_distance={args.max_correspondence_distance}, "
            f"max_iterations={args.max_iterations}, "
            f"relative_fitness={args.relative_fitness}, "
            f"relative_rmse={args.relative_rmse}"
        )

        result = o3d.pipelines.registration.registration_icp(
            source_ds,
            target_ds,
            max_correspondence_distance=args.max_correspondence_distance,
            init=init,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=args.relative_fitness,
                relative_rmse=args.relative_rmse,
                max_iteration=args.max_iterations,
            ),
        )

        transform = result.transformation
        scale, rotation, translation = decompose_similarity(transform)
        fitness = result.fitness
        inlier_rmse = result.inlier_rmse
        extra_metric_label = "Fitness"
        extra_metric_value = fitness

    aligned_source = copy.deepcopy(source)
    aligned_source.transform(transform)

    output_path = args.output
    if output_path is None:
        source_root, source_ext = os.path.splitext(args.source)
        output_path = f"{source_root}_aligned_to_{os.path.basename(args.target).split('.')[0]}{source_ext}"

    o3d.io.write_point_cloud(output_path, aligned_source, write_ascii=False)

    np.set_printoptions(suppress=True, precision=6)
    print("Estimated 4x4 transform T_target_source:")
    print(transform)
    print(f"\nEstimated scale: {scale}")
    print("\nEstimated rotation:")
    print(rotation)
    print(f"\nEstimated translation: {translation}")
    print(f"\n{extra_metric_label}: {extra_metric_value}")
    print(f"Inlier RMSE: {inlier_rmse}")
    print(f"Saved aligned point cloud to: {output_path}")

    if args.visualize:
        print("Opening registration visualization for downsampled point clouds.")
        draw_registration_result(source_ds, target_ds, transform)


if __name__ == "__main__":
    main()
