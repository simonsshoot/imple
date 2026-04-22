#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline for imple_new:
# 1) Auto-generate dataset if missing
# 2) Run batch benchmark with VLM visual feedback
# 3) Save metrics summary (GR, HR, PSR, PRR, TSR)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Python interpreter: prefer system python3 which has sapien/mani_skill pre-installed.
# sapien only provides wheels for Python 3.10; conda envs with 3.11+ will fail.
# Override with PYTHON env var if needed (e.g. PYTHON=/path/to/python bash pipeline.sh)
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
echo "[pipeline] python:        $PYTHON ($($PYTHON --version 2>&1))"

# --- Configurable environment variables ---
DATASET_JSON="${DATASET_JSON:-imple_new/outputs/tabletop_tasks.json}"
OUTPUT_DIR="${OUTPUT_DIR:-imple_new/outputs/run}"
VLM_MODEL="${VLM_MODEL:-qwen-vl-max}"
PLANNER_MODEL="${PLANNER_MODEL:-deepseek-chat}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
GENERATE_MODEL="${GENERATE_MODEL:-deepseek-chat}"
ENABLE_REPLAN="${ENABLE_REPLAN:-true}"
MAX_REPLAN="${MAX_REPLAN:-3}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-2}"
MAX_SCENES="${MAX_SCENES:-0}"
SEED="${SEED:-0}"
OBS_MODE="${OBS_MODE:-state}"
SIM_BACKEND="${SIM_BACKEND:-cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-cpu}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-240}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
CPU_ONLY="${CPU_ONLY:-}"

echo "=== [pipeline] imple_new benchmark pipeline ==="
echo "[pipeline] dataset:       $DATASET_JSON"
echo "[pipeline] output dir:    $OUTPUT_DIR"
echo "[pipeline] VLM model:     $VLM_MODEL"
echo "[pipeline] planner model: $PLANNER_MODEL"
echo "[pipeline] judge model:   $JUDGE_MODEL"
echo "[pipeline] replan:        $ENABLE_REPLAN (max $MAX_REPLAN)"

mkdir -p "$OUTPUT_DIR"

# --- Step 1: Auto-generate dataset if missing ---
if [[ ! -f "$DATASET_JSON" ]]; then
  echo "[pipeline] dataset not found, generating..."
  GENERATE_ARGS=(
    --output-json "$DATASET_JSON"
    --image-dir "imple_new/outputs/generated_frames"
    --samples-per-scene "$SAMPLES_PER_SCENE"
    --seed "$SEED"
    --model "$GENERATE_MODEL"
  )
  if [[ "$MAX_SCENES" -gt 0 ]]; then
    GENERATE_ARGS+=(--max-scenes "$MAX_SCENES")
  fi
  $PYTHON imple_new/generate.py "${GENERATE_ARGS[@]}"
  echo "[pipeline] dataset generated: $DATASET_JSON"
else
  echo "[pipeline] using existing dataset: $DATASET_JSON"
fi

# --- Step 2: Run batch benchmark ---
echo "[pipeline] running batch benchmark..."
RUN_ARGS=(
  --dataset-json "$DATASET_JSON"
  --output-dir "$OUTPUT_DIR"
  --vlm-model "$VLM_MODEL"
  --planner-model "$PLANNER_MODEL"
  --judge-model "$JUDGE_MODEL"
  --enable-replan "$ENABLE_REPLAN"
  --max-replan-attempts "$MAX_REPLAN"
  --obs-mode "$OBS_MODE"
  --sim-backend "$SIM_BACKEND"
  --render-backend "$RENDER_BACKEND"
  --render-mode rgb_array
  --max-episode-steps "$MAX_EPISODE_STEPS"
  --seed "$SEED"
  --save-metrics
  --metrics-file metrics_summary.json
)

if [[ -n "$SAVE_VIDEO" ]]; then
  RUN_ARGS+=(--save-video --video-fps 20)
fi

if [[ -n "$CPU_ONLY" ]]; then
  RUN_ARGS+=(--cpu-only)
fi

$PYTHON imple_new/run.py "${RUN_ARGS[@]}"

# --- Step 3: Report ---
echo ""
echo "=== [pipeline] done ==="
echo "[pipeline] run results:    $OUTPUT_DIR/run_results.json"
echo "[pipeline] metrics:        $OUTPUT_DIR/metrics_summary.json"
