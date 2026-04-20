import argparse
import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode

# Ensure imports work when running this file directly from imple/infos.
CURRENT_DIR = Path(__file__).resolve().parent
IMPLE_DIR = CURRENT_DIR.parent
if str(IMPLE_DIR) not in sys.path:
	sys.path.insert(0, str(IMPLE_DIR))

from controller import Controller
from utils import execute_low_level_plan


def _to_np(x):
	if hasattr(x, "detach"):
		x = x.detach().cpu().numpy()
	return np.asarray(x)


def _quat_diff_deg(q1: np.ndarray, q2: np.ndarray) -> float:
	q1 = np.asarray(q1, dtype=np.float64)
	q2 = np.asarray(q2, dtype=np.float64)
	q1 = q1 / (np.linalg.norm(q1) + 1e-12)
	q2 = q2 / (np.linalg.norm(q2) + 1e-12)
	dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
	return float(np.degrees(2.0 * np.arccos(dot)))


def run_test(args: argparse.Namespace) -> None:
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	env = gym.make(
		args.scene,
		obs_mode="state",
		control_mode="pd_ee_delta_pose",
		max_episode_steps=args.max_episode_steps,
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

	planner = Controller(env)
	tcp_q_before = _to_np(env.unwrapped.agent.tcp.pose.q)[0]

	# Dedicated rotation-atomic test: pick small sphere, rotate wrist in 4 directions.
	low_level_plan = [
		"find sphere",
		"pick sphere",
		"rotate_left",
		"rotate_right",
		"rotate_up",
		"rotate_down",
		"drop",
	]

	result = execute_low_level_plan(planner, low_level_plan)
	tcp_q_after = _to_np(env.unwrapped.agent.tcp.pose.q)[0]
	rot_delta_deg = _quat_diff_deg(tcp_q_before, tcp_q_after)

	print("[test2] low-level plan:", low_level_plan)
	print("[test2] execute result:", result)
	print(f"[test2] tcp rotation delta (deg): {rot_delta_deg:.3f}")

	if hasattr(env, "render_images") and len(env.render_images) < 2:
		noop = np.zeros_like(env.action_space.sample(), dtype=np.float32)
		env.step(noop)

	if hasattr(env, "flush_video"):
		env.flush_video(name=args.video_name)

	print(f"[test2] saved video to: {output_dir / (args.video_name + '.mp4')}")
	env.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Atomic rotate test: pick sphere then rotate end-effector.")
	parser.add_argument("--scene", type=str, default="PlaceSphere-v1", help="ManiSkill scene id with a small sphere")
	parser.add_argument("--output-dir", type=str, default="imple/outputs/test_rotate_atomic", help="Directory for output video")
	parser.add_argument("--video-name", type=str, default="test2_rotate_sphere", help="Saved video name (without extension)")
	parser.add_argument("--sim-backend", type=str, default="cpu", choices=["auto", "cpu", "gpu"], help="Simulation backend")
	parser.add_argument("--render-backend", type=str, default="cpu", choices=["cpu", "gpu", "none"], help="Render backend")
	parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "sensors", "all", "none"], help="Render mode")
	parser.add_argument("--video-fps", type=int, default=20, help="Saved video FPS")
	parser.add_argument("--max-episode-steps", type=int, default=240, help="Episode step budget")
	parser.add_argument("--seed", type=int, default=0, help="Environment seed")
	args = parser.parse_args()
	run_test(args)


if __name__ == "__main__":
	main()
