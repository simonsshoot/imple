import argparse
from pathlib import Path
import gymnasium as gym
import mani_skill.envs
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
from controller import Controller
from utils import execute_low_level_plan

def run_test(args: argparse.Namespace) -> None:
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  env = gym.make(
    args.scene,
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    render_mode=args.render_mode,
    render_backend=args.render_backend,
    sim_backend=args.sim_backend,
  )

  env = RecordEpisode(
    env,
    output_dir=str(output_dir),
    save_trajectory=False,
    save_video=True,
    video_fps=args.video_fps,
  )

  env.reset(seed=args.seed)

  low_level_plan = [
    "find cube",
    "pick cube",
    # "move_back cube",
    # "move_right cube",
    # "put table",
  ]

  planner = Controller(env)
  result = execute_low_level_plan(planner, low_level_plan)
  print("[test] execute result:", result)
  if hasattr(env, "render_images") and len(env.render_images) < 2:
    noop = np.zeros_like(env.action_space.sample(), dtype=np.float32)
    env.step(noop)
  if hasattr(env, "flush_video"):
    env.flush_video(name="test")
  video_path = output_dir / "test.mp4"
  print(f"[test] saved video to: {video_path}")
  env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description="Standalone low-level planner test runner")
  parser.add_argument("--scene", type=str, default="PickCube-v1", help="ManiSkill scene id")
  parser.add_argument("--output-dir", type=str, default="/home/yx/yx_search/agentsafe/imple/outputs", help="Directory for output video")
  parser.add_argument("--sim-backend", type=str, default="cpu", choices=["auto", "cpu", "gpu"], help="Simulation backend")
  parser.add_argument("--render-backend", type=str, default="cpu", choices=["cpu", "gpu", "none"], help="Render backend")
  parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "sensors", "all", "none"], help="Render mode")
  parser.add_argument("--video-fps", type=int, default=20, help="Saved video FPS")
  parser.add_argument("--seed", type=int, default=0, help="Environment seed")
  args = parser.parse_args()
  run_test(args)

if __name__ == "__main__":
  main()
