#!/usr/bin/env bash
set -euo pipefail

# Run the imple pipeline entry to verify VLM->plan->controller integration.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "^^^ [pipeline] starting ^^^"
# python imple/run.py \
#   --scene PlaceSphere-v1 \
#   --obs-mode state \
#   --tasks "Move the ball from left to right" \
#   --model qwen-vl-max \
#   --output-dir imple/outputs \
#   --sim-backend cpu \
#   --render-backend cpu \
#   --render-mode rgb_array \
#   --max_episode_steps 240 \
#   --save-video \
#   --video-fps 20 \
#   --seed 0

# python imple/run.py \
#   --scene PickCube-v1 \
#   --obs-mode state \
#   --tasks "Pickup the cube, move to right, and put it down" \
#   --model qwen-vl-max \
#   --output-dir imple/outputs \
#   --sim-backend cpu \
#   --render-backend cpu \
#   --render-mode rgb_array \
#   --max_episode_steps 240 \
#   --save-video \
#   --video-fps 20 \
#   --seed 0

python imple/run.py \
	--meta-json /home/yx/yx_search/agentsafe/VLATest/automation/output/meta.json \
	--meta-index 0 \
	--meta-count 3 \
	--obs-mode rgb \
	--model qwen-vl-max \
	--output-dir imple/outputs/meta_batch \
	--sim-backend cpu \
	--render-backend cpu \
	--render-mode rgb_array \
	--max_episode_steps 240 \
	--save-video \
	--video-fps 20 \
	--min-video-seconds 5

echo "[pipeline] done. Outputs should be under: imple/outputs/meta_batch/meta_case_*"
