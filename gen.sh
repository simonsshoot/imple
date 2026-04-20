#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash gen.sh tabletop
#   bash gen.sh humanoid
#   bash gen.sh mobile_manipulation
#   bash gen.sh control
#   bash gen.sh dexterity
# Optional:
#   MODEL=deepseek-chat MAX_SCENES=10 SAMPLES_PER_SCENE=2 bash gen.sh tabletop

CATEGORY="${1:-tabletop}"
MODEL="${MODEL:-deepseek-chat}"
MAX_SCENES="${MAX_SCENES:-0}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-2}"
SEED="${SEED:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_ROOT="${SCRIPT_DIR}/../ManiSkill/mani_skill/envs/tasks"
IMAGE_DIR="${SCRIPT_DIR}/outputs/generated_first_frames"

case "${CATEGORY}" in
  tabletop|humanoid|mobile_manipulation|control|dexterity)
    ;;
  *)
    echo "[gen.sh] Invalid category: ${CATEGORY}"
    echo "[gen.sh] Use: tabletop | humanoid | mobile_manipulation | control | dexterity"
    exit 1
    ;;
esac

OUTPUT_JSON="${SCRIPT_DIR}/outputs/${CATEGORY}_tasks.json"

echo "[gen.sh] category=${CATEGORY}"
echo "[gen.sh] output=${OUTPUT_JSON}"

action_cmd=(
  python "${SCRIPT_DIR}/generate.py"
  --tasks-root-dir "${TASKS_ROOT}"
  --task-categories "${CATEGORY}"
  --model "${MODEL}"
  --max-scenes "${MAX_SCENES}"
  --samples-per-scene "${SAMPLES_PER_SCENE}"
  --seed "${SEED}"
  --output-json "${OUTPUT_JSON}"
  --image-dir "${IMAGE_DIR}"
  --obs-mode rgb
  --control-mode pd_ee_delta_pose
  --max-episode-steps 120
  --sim-backend cpu
  --render-backend cpu
  --render-mode rgb_array
  --shader default
)

"${action_cmd[@]}"

echo "[gen.sh] Done. JSON saved to: ${OUTPUT_JSON}"
