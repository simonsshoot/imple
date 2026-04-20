#!/usr/bin/env bash
set -euo pipefail

# One-click run pipeline:
# 1) use existing dataset json
# 2) run batch execution
# 3) save metrics summary (including TSR)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASET_JSON="${DATASET_JSON:-imple/outputs/humanoid_tasks.json}"
OUTPUT_DIR="${OUTPUT_DIR:-imple/outputs/humanoid}"
MODEL="${MODEL:-qwen-vl-max}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
PIPELINE="${PIPELINE:-auto}"
CONTROL_MODE="${CONTROL_MODE:-}"

echo "^^^ [pipeline] starting ^^^"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$DATASET_JSON" ]]; then
  echo "[pipeline] dataset json not found: $DATASET_JSON"
  echo "[pipeline] please generate it first, then rerun this script."
  exit 1
fi

echo "[pipeline] using existing dataset: $DATASET_JSON"

echo "[pipeline] running batch benchmark..."
python imple/run.py \
  --dataset-json "$DATASET_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --pipeline "$PIPELINE" \
  --obs-mode state \
  --control-mode "$CONTROL_MODE" \
  --model "$MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --sim-backend cpu \
  --render-backend cpu \
  --render-mode rgb_array \
  --max_episode_steps 240 \
  --save-video \
  --video-fps 20 \
  --save-metrics \
  --metrics-file metrics_summary.json

echo "[pipeline] run results: $OUTPUT_DIR/run_results.json"
echo "[pipeline] metrics saved at: $OUTPUT_DIR/metrics_summary.json"

echo "[pipeline] done."
