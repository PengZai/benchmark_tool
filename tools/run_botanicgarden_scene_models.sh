#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/spiderman/Shuoyuan/miniconda3/bin/conda}"

SCENE_NAME="BotanicGarden_1018_00_32_test_4"
DATASET_NAME="BotanicGarden"
BATCH_SIZE=1
NUM_VIEW=32
MACHINE_CONFIG="${MACHINE_CONFIG:-default}"

MODELS=(
  depth_anything_v2
  depth_enhancement
  mapanything
  mast3r
  priorda
)

get_env_path() {
  case "$1" in
    depth_enhancement)
      echo "/home/spiderman/Shuoyuan/miniconda3/envs/benchmark-depth-enhancement"
      ;;
    depth_anything_v2)
      echo "/home/spiderman/Shuoyuan/miniconda3/envs/benchmark-depthanything2"
      ;;
    mapanything)
      echo "/home/spiderman/Shuoyuan/miniconda3/envs/benchmark-mapanything"
      ;;
    mast3r)
      echo "/home/spiderman/Shuoyuan/miniconda3/envs/benchmark-mast3r"
      ;;
    priorda)
      echo "/home/spiderman/Shuoyuan/miniconda3/envs/benchmark-prior-depth-anything"
      ;;
    *)
      echo "Unknown model: $1" >&2
      return 1
      ;;
  esac
}

cd "$ROOT_DIR"

for model in "${MODELS[@]}"; do
  env_path="$(get_env_path "$model")"

  echo "============================================================"
  echo "Running model=${model} dataset=${DATASET_NAME} scene=${SCENE_NAME}"
  echo "batch_size=${BATCH_SIZE} num_view=${NUM_VIEW} machine=${MACHINE_CONFIG}"
  echo "conda_env=${env_path}"
  echo "============================================================"

  "$CONDA_BIN" run --prefix "$env_path" python dense_slam_benchmark/benchmark_tools/scripts/dense_n_view_benchmark.py \
    machine="${MACHINE_CONFIG}" \
    model="${model}" \
    dataset_test="${DATASET_NAME}" \
    dataset="${DATASET_NAME}" \
    scene_name="${SCENE_NAME}" \
    batch_size="${BATCH_SIZE}" \
    num_view="${NUM_VIEW}"
done
