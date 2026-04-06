import argparse
import os
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np
from mani_skill.utils.structs import Pose

from mani_skill.utils.wrappers.record import RecordEpisode


def parse_args():
	parser = argparse.ArgumentParser(
		description="Use Panda to move a sphere from left to right in ManiSkill and save a video in headless mode."
	)
	parser.add_argument("--seed", type=int, default=0, help="Episode seed")
	parser.add_argument(
		"--record-dir",
		type=str,
		default="videos/place_sphere",
		help="Directory used to save output videos",
	)
	parser.add_argument(
		"--video-name",
		type=str,
		default="place_sphere_left_to_right",
		help="Saved mp4 file name (without extension)",
	)
	parser.add_argument(
		"--sim-backend",
		type=str,
		default="cpu",
		choices=["auto", "cpu", "gpu"],
		help="Simulation backend. Use cpu on headless servers to avoid GPU/EGL crashes.",
	)
	parser.add_argument(
		"--cpu-only",
		action="store_true",
		help="Hide CUDA devices (sets CUDA_VISIBLE_DEVICES='') for better headless compatibility.",
	)
	parser.add_argument(
		"--shader",
		type=str,
		default="default",
		choices=["default", "rt", "rt-fast"],
		help="Shader pack used for rendering",
	)
	parser.add_argument(
		"--render-mode",
		type=str,
		default="rgb_array",
		choices=["rgb_array", "sensors", "all"],
		help="Render mode used for recording",
	)
	parser.add_argument(
		"--render-backend",
		type=str,
		default="cpu",
		choices=["cpu", "gpu", "none"],
		help="Rendering backend. Use cpu on headless servers.",
	)
	parser.add_argument(
		"--vis",
		action="store_true",
		help="Open viewer for debugging (do not use on a headless server)",
	)
	parser.add_argument(
		"--steps-per-phase",
		type=int,
		default=30,
		help="Control steps for each phase (approach / carry / retreat)",
	)
	parser.add_argument(
		"--max-episode-steps",
		type=int,
		default=240,
		help="Episode step budget. Increase this to avoid early truncation and get longer videos.",
	)
	parser.add_argument(
		"--video-fps",
		type=int,
		default=20,
		help="FPS of the saved video. Lower FPS makes the same number of frames play longer.",
	)
	return parser.parse_args()


def _to_np(x):
	if hasattr(x, "detach"):
		x = x.detach().cpu().numpy()
	return np.asarray(x)


def _move_action(ee_pos, target, grip=1.0, gain=10.0):
	action = np.zeros(7, dtype=np.float32)
	delta = np.clip((target - ee_pos) * gain, -1.0, 1.0)
	action[:3] = delta
	action[6] = float(np.clip(grip, -1.0, 1.0))
	return action


def main():
	args = parse_args()
	if args.cpu_only:
		os.environ["CUDA_VISIBLE_DEVICES"] = ""

	env = gym.make(
		"PlaceSphere-v1",
		obs_mode="state",
		control_mode="pd_ee_delta_pose",
		max_episode_steps=args.max_episode_steps,
		render_mode=args.render_mode,
		render_backend=args.render_backend,
		sensor_configs=dict(shader_pack=args.shader),
		human_render_camera_configs=dict(shader_pack=args.shader),
		viewer_camera_configs=dict(shader_pack=args.shader),
		sim_backend=args.sim_backend,
	)

	output_dir = Path(args.record_dir)
	env = RecordEpisode(
		env,
		output_dir=str(output_dir),
		save_trajectory=False,
		save_video=True,
		video_fps=args.video_fps,
		source_type="scripted",
		source_desc="Scripted Panda trajectory for left-to-right sphere transfer in headless mode",
	)

	obs, info = env.reset(seed=args.seed)
	""""
	关于unwrapped：
	在 Gymnasium 里，环境通常会被多层 wrapper 包住。unwrapped 会一直剥到最里层的原始环境对象，便于直接访问底层属性和方法。
	这里的流程是：gym.make 创建环境，并在 ManiSkill 注册时附加了 TimeLimitWrapper（见 registration.py:127 和 registration.py:242）
	然后又被 RecordEpisode 包了一层（见 record.py:102）

	所以u = env.unwrapped 得到的是 PlaceSphereEnv 实例
	内部包含很多属性，例如：
	u.radius：球半径（标量）
	u.block_half_size：bin 的尺寸列表
	u.obj / u.bin：场景里的物体对象
	u.agent：机器人对象
	u.agent.tcp.pose.p：末端位姿位置张量
	"""
	u = env.unwrapped

	# Force a clear left-to-right layout for repeatable videos.
	# sphere_left：把球放到左侧，坐标大致是 x=-0.12, y=0, z=球半径。
	sphere_left = np.array([[-0.12, 0.0, float(u.radius)]], dtype=np.float32)
	# bin_right：把容器放到右侧，坐标大致是 x=0.12, y=0, z=容器底部高度
	bin_right = np.array([[0.12, 0.0, float(u.block_half_size[0])]], dtype=np.float32)
	# u.obj.set_pose / u.bin.set_pose：把这两个物体“直接传送”到指定位置。
  # Pose.create_from_pq 里的 q=[1,0,0,0] 表示单位四元数，即不旋转。
	u.obj.set_pose(Pose.create_from_pq(p=sphere_left, q=[1, 0, 0, 0]))
	u.bin.set_pose(Pose.create_from_pq(p=bin_right, q=[1, 0, 0, 0]))

	left_top = np.array([-0.12, 0.0, 0.16], dtype=np.float32)
	right_top = np.array([0.12, 0.0, 0.16], dtype=np.float32)
	hover_above_sphere = np.array([-0.12, 0.0, 0.08], dtype=np.float32)

	carrying = False

	def run_phase(target, grip, n_steps, attach=False):
		"""
		run_phase 是这段脚本里的“阶段执行器”：给它一个目标点、夹爪状态和步数，它就连续执行控制，让末端逐步靠近目标；可选地在搬运阶段把球“吸附”在末端下方一起移动
		"""
		nonlocal obs, info, carrying
		for _ in range(n_steps):
			# u.agent.tcp.pose.p拿末端执行器实时位置，形成闭环控制
			ee = _to_np(u.agent.tcp.pose.p)[0]
			# _move_action：把“当前位置 -> 目标位置”的误差转成控制动作。
			action = _move_action(ee, target, grip=grip)
			obs, _, terminated, truncated, info = env.step(action)
			if attach:
				# Keep the sphere under the end-effector while carrying.
				# _to_np：把张量转成 numpy，兼容 torch 和 numpy 输入
				tcp = _to_np(u.agent.tcp.pose.p)[0]
				sphere_pose = np.array([[tcp[0], tcp[1], max(0.035, tcp[2] - 0.055)]], dtype=np.float32)
				# 按给定位置和四元数直接重设球位姿
				u.obj.set_pose(Pose.create_from_pq(p=sphere_pose, q=[1, 0, 0, 0]))
				carrying = True
			if terminated or truncated:
				return

	# Approach and grasp-like close.
	run_phase(left_top, grip=1.0, n_steps=args.steps_per_phase)
	run_phase(hover_above_sphere, grip=1.0, n_steps=args.steps_per_phase)
	run_phase(hover_above_sphere, grip=-1.0, n_steps=max(10, args.steps_per_phase // 2))

	# Carry sphere from left to right while arm moves.
	run_phase(right_top, grip=-1.0, n_steps=args.steps_per_phase * 2, attach=True)
	if carrying:
		final_right_pose = np.array([[0.11, 0.0, float(u.radius)]], dtype=np.float32)
		u.obj.set_pose(Pose.create_from_pq(p=final_right_pose, q=[1, 0, 0, 0]))

	# Release and retreat.
	run_phase(right_top, grip=1.0, n_steps=max(10, args.steps_per_phase // 2))
	run_phase(np.array([0.05, 0.0, 0.20], dtype=np.float32), grip=1.0, n_steps=args.steps_per_phase)

	sphere_start_x = float(sphere_left[0, 0])
	sphere_end_x = float(_to_np(u.obj.pose.p)[0, 0])
	moved_right = sphere_end_x > sphere_start_x + 0.08

	env.flush_video(name=args.video_name)
	env.close()

	print(f"Sphere x: {sphere_start_x:.3f} -> {sphere_end_x:.3f}")
	print(f"Moved right (x shift > 0.08): {moved_right}")
	print(f"Saved video: {output_dir / (args.video_name + '.mp4')}")


if __name__ == "__main__":
	main()
