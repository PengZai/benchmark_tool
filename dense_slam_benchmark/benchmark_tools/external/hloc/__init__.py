import os
import inspect
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dense_slam_benchmark.dataset_tools import utils as dataset_utils


class HLocWrapper(torch.nn.Module):
    SPARSE_MODULES = {"superpoint", "sift", "loma"}
    DENSE_MODULE_ALIASES = {
        "loftr": "loftr",
        "efficient_loftr": "efficient_loftr",
        "efficientloftr": "efficient_loftr",
        "roma": "roma",
        "romav2": "romav2",
    }
    DEFAULT_MODULE_CONFIGS = {
        "superpoint": {"feature_conf": "superpoint_max", "matcher_conf": "superglue"},
        "sift": {"feature_conf": "sift", "matcher_conf": "NN-mutual"},
        "loma": {"feature_conf": "loma_aachen", "matcher_conf": "loma"},
        "loftr": {"dense_matcher_conf": "loftr"},
        "efficient_loftr": {"dense_matcher_conf": "efficient_loftr"},
        "efficientloftr": {"dense_matcher_conf": "efficient_loftr"},
        "roma": {"dense_matcher_conf": "roma"},
        "romav2": {"dense_matcher_conf": "romav2"},
    }

    def __init__(
        self,
        name,
        cache_dir,
        module="superpoint",
        feature_conf=None,
        matcher_conf=None,
        dense_matcher_conf=None,
        pair_mode="exhaustive",
        use_input_intrinsics=True,
        use_input_poses_as_init_guess=False,
        no_refine_intrinsics=True,
        skip_geometric_verification=False,
        min_model_size=2,
        n_threads=16,
        max_kps=8192,
        verbose=False,
        triangulation_options=None,
        bundle_adjustment_options=None,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.cache_dir = cache_dir
        self.module = str(module).lower()
        self.feature_conf = feature_conf
        self.matcher_conf = matcher_conf
        self.dense_matcher_conf = dense_matcher_conf
        self.pair_mode = pair_mode
        self.use_input_intrinsics = use_input_intrinsics
        self.use_input_poses_as_init_guess = use_input_poses_as_init_guess
        self.no_refine_intrinsics = no_refine_intrinsics
        self.skip_geometric_verification = skip_geometric_verification
        self.min_model_size = min_model_size
        self.n_threads = n_threads
        self.max_kps = max_kps
        self.verbose = verbose
        self.triangulation_options = triangulation_options or {}
        self.bundle_adjustment_options = bundle_adjustment_options or {}

        os.makedirs(self.cache_dir, exist_ok=True)

        (
            self.extract_features,
            self.match_features,
            self.match_dense,
            self.pairs_from_exhaustive,
            self.reconstruction,
            self.reconstruction_with_poses,
            self.hloc_io,
            self.pycolmap,
        ) = self._load_hloc_modules()

        self.pipeline_kind = self._resolve_pipeline_kind()
        self._apply_default_configs()
        self._validate_configs()

    def _resolve_pipeline_kind(self):
        self.module = self.DENSE_MODULE_ALIASES.get(self.module, self.module)
        if self.module in self.SPARSE_MODULES:
            return "sparse"
        if self.module in self.DENSE_MODULE_ALIASES:
            return "dense"
        raise ValueError(
            f"Unsupported hloc module '{self.module}'. "
            f"Supported modules: {sorted(set(self.SPARSE_MODULES) | set(self.DENSE_MODULE_ALIASES))}"
        )

    def _apply_default_configs(self):
        defaults = self.DEFAULT_MODULE_CONFIGS.get(self.module, {})
        if self.pipeline_kind == "sparse":
            if self.feature_conf is None:
                self.feature_conf = defaults.get("feature_conf")
            if self.matcher_conf is None:
                self.matcher_conf = defaults.get("matcher_conf")
        else:
            if self.dense_matcher_conf is None:
                self.dense_matcher_conf = defaults.get("dense_matcher_conf")

    def _validate_configs(self):
        if self.pair_mode != "exhaustive":
            raise ValueError(
                f"Unsupported pair_mode '{self.pair_mode}'. Only 'exhaustive' is implemented."
            )

        if self.pipeline_kind == "sparse":
            if self.feature_conf not in self.extract_features.confs:
                raise ValueError(
                    f"Unknown HLoc feature config '{self.feature_conf}'. "
                    f"Available configs: {sorted(self.extract_features.confs)}"
                )
            if self.matcher_conf not in self.match_features.confs:
                raise ValueError(
                    f"Unknown HLoc matcher config '{self.matcher_conf}'. "
                    f"Available configs: {sorted(self.match_features.confs)}"
                )
        else:
            if self.dense_matcher_conf not in self.match_dense.confs:
                raise ValueError(
                    f"Requested dense module '{self.module}' via dense config "
                    f"'{self.dense_matcher_conf}', but this HLoc installation does not expose it. "
                    f"Available dense configs: {sorted(self.match_dense.confs)}"
                )

    def _load_hloc_modules(self):
        try:
            import pycolmap
            from hloc import (
                extract_features,
                match_dense,
                match_features,
                pairs_from_exhaustive,
                reconstruction,
                reconstruction_with_poses,
                utils,
            )

            return (
                extract_features,
                match_features,
                match_dense,
                pairs_from_exhaustive,
                reconstruction,
                reconstruction_with_poses,
                utils.io,
                pycolmap,
            )
        except ImportError:
            repo_root = Path(__file__).resolve().parents[4]
            detectorfree_root = repo_root / "thirdparty" / "DetectorFreeSfM"
            vendored_hloc_root = detectorfree_root / "third_party" / "Hierarchical_Localization"

            for path in (detectorfree_root, vendored_hloc_root):
                path_str = str(path)
                if path.exists() and path_str not in sys.path:
                    sys.path.insert(0, path_str)

            try:
                import pycolmap
                from hloc import (
                    extract_features,
                    match_dense,
                    match_features,
                    pairs_from_exhaustive,
                    reconstruction,
                    reconstruction_with_poses,
                    utils,
                )

                return (
                    extract_features,
                    match_features,
                    match_dense,
                    pairs_from_exhaustive,
                    reconstruction,
                    reconstruction_with_poses,
                    utils.io,
                    pycolmap,
                )
            except ImportError as exc:
                raise ImportError(
                    "HLocWrapper requires the optional 'hloc' and 'pycolmap' dependencies. "
                    "Install the 'hloc' extra from this project, or make the vendored "
                    "Hierarchical Localization dependencies available."
                ) from exc

    @staticmethod
    def _build_image_name(view):
        return os.path.join(
            str(view["scene_name"][0]),
            str(view["camera_name"][0]),
            "undistorted_images",
            str(view["name"][0]),
        )

    @staticmethod
    def _to_camera_center_pose(image):
        if hasattr(image, "cam_from_world"):
            cam_from_world = image.cam_from_world
            if callable(cam_from_world):
                cam_from_world = cam_from_world()
            if hasattr(cam_from_world, "matrix"):
                matrix = cam_from_world.matrix()
                return dataset_utils.invert_transform(np.asarray(matrix, dtype=np.float32))

        rotation_w2c = image.rotmat()
        translation_w2c = np.asarray(image.tvec, dtype=np.float32).reshape(3)

        T_w_c = np.eye(4, dtype=np.float32)
        T_w_c[:3, :3] = rotation_w2c.T.astype(np.float32)
        T_w_c[:3, 3] = (-rotation_w2c.T @ translation_w2c).astype(np.float32)
        return T_w_c

    @staticmethod
    def _write_image(image_path, image_bgr):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), image_bgr):
            raise RuntimeError(f"Failed to write temporary image: {image_path}")

    @staticmethod
    def _write_intrinsics_txt(intrinsics_path, intrinsics):
        intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(intrinsics_path, intrinsics.astype(np.float32))

    @staticmethod
    def _to_numpy_image(raw_image):
        if torch.is_tensor(raw_image):
            raw_image = raw_image.detach().cpu().numpy()
        raw_image = np.asarray(raw_image)
        if raw_image.dtype != np.uint8:
            raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
        return raw_image

    @staticmethod
    def _write_pose_priors(pose_path, views, image_names):
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pose_path, "w", encoding="utf-8") as f:
            for view, image_name in zip(views, image_names):
                T_w_c = view["T_w_c"].squeeze(0).cpu().numpy()
                cam_from_world = dataset_utils.invert_transform(T_w_c)
                tvec, quat_xyzw = dataset_utils.T_to_pose(cam_from_world)
                qx, qy, qz, qw = quat_xyzw.tolist()
                tx, ty, tz = tvec.tolist()
                f.write(f"{image_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}\n")

    def _prepare_inputs(self, frames):
        num_frame = len(frames)
        target_views = [frame[0] for frame in frames]
        views = [view for frame in frames for view in frame]

        batch_size_per_view = views[0]["undistorted_image"].shape[0]
        if batch_size_per_view != 1:
            raise AssertionError(
                f"Batch size of input views should be 1, but got {batch_size_per_view}."
            )

        work_dir = Path(tempfile.mkdtemp(dir=self.cache_dir, prefix="hloc_"))
        image_dir = work_dir / "images"
        outputs_dir = work_dir / "outputs"
        intrinsics_dir = work_dir / "intrinsics"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        image_names = []
        target_names = []

        for view in views:
            image_name = self._build_image_name(view)
            image_path = image_dir / image_name
            raw_image = self._to_numpy_image(view["undistorted_raw_image"][0])
            self._write_image(image_path, raw_image)
            image_names.append(image_name)

            if self.use_input_intrinsics and "intrinsics" in view:
                intrinsics_path = intrinsics_dir / (Path(image_name).stem + ".txt")
                intrinsics = view["intrinsics"].squeeze(0).cpu().numpy()
                self._write_intrinsics_txt(intrinsics_path, intrinsics)

        for view in target_views:
            target_names.append(self._build_image_name(view))

        return {
            "num_frame": num_frame,
            "target_views": target_views,
            "views": views,
            "work_dir": work_dir,
            "image_dir": image_dir,
            "outputs_dir": outputs_dir,
            "intrinsics_dir": intrinsics_dir,
            "image_names": image_names,
            "target_names": target_names,
        }

    def _run_matching_pipeline(self, prepared):
        pairs_path = prepared["outputs_dir"] / "pairs-exhaustive.txt"
        self.pairs_from_exhaustive.main(pairs_path, image_list=prepared["image_names"])

        if self.pipeline_kind == "sparse":
            feature_path = self.extract_features.main(
                self.extract_features.confs[self.feature_conf],
                prepared["image_dir"],
                export_dir=prepared["outputs_dir"],
                image_list=prepared["image_names"],
                overwrite=True,
            )
            matches_path = self.match_features.main(
                self.match_features.confs[self.matcher_conf],
                pairs_path,
                feature_path,
                matches=prepared["outputs_dir"] / "matches.h5",
                overwrite=True,
            )
            return feature_path, matches_path, pairs_path

        feature_path, matches_path = self.match_dense.main(
            self.match_dense.confs[self.dense_matcher_conf],
            pairs_path,
            prepared["image_dir"],
            export_dir=prepared["outputs_dir"],
            features=prepared["outputs_dir"] / "dense_features.h5",
            matches=prepared["outputs_dir"] / "dense_matches.h5",
            max_kps=self.max_kps,
            overwrite=True,
        )
        return feature_path, matches_path, pairs_path

    def _run_reconstruction(self, prepared, feature_path, matches_path, pairs_path):
        colmap_configs = {
            "use_pba": False,
            "ImageReader_camera_mode": "per_image",
            "no_refine_intrinsics": self.no_refine_intrinsics,
            "colmap_mapper_cfgs": None,
            "min_model_size": self.min_model_size,
            "n_threads": self.n_threads,
        }

        prior_intrin = str(prepared["intrinsics_dir"]) if self.use_input_intrinsics else None

        start = time.time()
        if self.use_input_poses_as_init_guess:
            pose_path = prepared["outputs_dir"] / "pose_priors.txt"
            self._write_pose_priors(
                pose_path,
                prepared["views"],
                prepared["image_names"],
            )
            reconstruction = self.reconstruction_with_poses.main(
                sfm_dir=prepared["outputs_dir"] / "sfm",
                image_dir=prepared["image_dir"],
                pairs=pairs_path,
                features=feature_path,
                matches=matches_path,
                poses=pose_path,
                camera_mode=self.pycolmap.CameraMode.PER_IMAGE,
                verbose=self.verbose,
                skip_geometric_verification=self.skip_geometric_verification,
                min_match_score=None,
                image_list=prepared["image_names"],
                triangulation_options=self.triangulation_options,
                bundle_adjustment_options=self.bundle_adjustment_options,
            )
        else:
            reconstruction_kwargs = dict(
                sfm_dir=prepared["outputs_dir"] / "sfm",
                image_dir=prepared["image_dir"],
                pairs=pairs_path,
                features=feature_path,
                matches=matches_path,
                camera_mode=self.pycolmap.CameraMode.PER_IMAGE,
                verbose=self.verbose,
                skip_geometric_verification=self.skip_geometric_verification,
                min_match_score=None,
                image_list=prepared["image_names"],
            )
            reconstruction_signature = inspect.signature(self.reconstruction.main)
            if "prior_intrin" in reconstruction_signature.parameters:
                reconstruction_kwargs["prior_intrin"] = prior_intrin
            if "colmap_configs" in reconstruction_signature.parameters:
                reconstruction_kwargs["colmap_configs"] = colmap_configs
            if "mapper_options" in reconstruction_signature.parameters:
                reconstruction_kwargs["mapper_options"] = {"num_threads": self.n_threads}
            reconstruction = self.reconstruction.main(**reconstruction_kwargs)
        runtime = time.time() - start
        return reconstruction, runtime

    def forward(self, frames):
        prepared = self._prepare_inputs(frames)
        feature_path, matches_path, pairs_path = self._run_matching_pipeline(prepared)
        reconstruction, runtime = self._run_reconstruction(
            prepared, feature_path, matches_path, pairs_path
        )

        reconstructed_images = {}
        sparse_points = {}
        if reconstruction is not None:
            reconstructed_images = {image.name: image for image in reconstruction.images.values()}
            sparse_points = self.hloc_io.get_sparse_points_per_image(
                reconstruction,
                with_depth=True,
            )

        res = []
        for frame_idx, target_view in enumerate(prepared["target_views"]):
            height, width = target_view["undistorted_raw_image"].shape[1:3]
            target_name = prepared["target_names"][frame_idx]

            pred_depth = np.zeros((1, height, width), dtype=np.float32)
            pred_depth_mask = np.zeros((1, height, width), dtype=bool)
            pred_T_w_c = target_view["T_w_c"].cpu().numpy()

            if target_name in reconstructed_images:
                reconstruction_image = reconstructed_images[target_name]
                sparse_depth = sparse_points[target_name]["depth"]
                pred_depth[0] = sparse_depth
                pred_depth_mask[0] = sparse_depth > 0.0
                pred_T_w_c = self._to_camera_center_pose(reconstruction_image)[None]

            res.append(
                {
                    "pred_depth": pred_depth,
                    "pred_depth_mask": pred_depth_mask,
                    "pred_T_w_c": pred_T_w_c,
                    "runtime": runtime / float(prepared["num_frame"]),
                }
            )

        return res
