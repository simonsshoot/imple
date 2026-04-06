import argparse
import os
from pathlib import Path
import gymnasium as gym
import mani_skill.envs
import numpy as np
from utils import gen_low_level_plan, execute_low_level_plan, save_pics
from agents import Agents
from controller import Controller
from mani_skill.utils.wrappers.record import RecordEpisode

def run(args: argparse.Namespace)-> None:
  task= args.tasks
  scene = args.scene
  output_dir = Path(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

  if args.cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  if args.obs_mode == "state":
    # Keep this path strictly aligned with infos/vlm.py, which is known to be
    # more stable on this machine.
    env = gym.make(
      scene,
      obs_mode="state",
      control_mode="pd_ee_delta_pose",
      render_mode=args.render_mode,
      render_backend=args.render_backend,
      sim_backend=args.sim_backend,
    )
  else:
    env = gym.make(
      scene,
      obs_mode=args.obs_mode,
      control_mode="pd_ee_delta_pose",
      max_episode_steps=args.max_episode_steps,
      render_mode=args.render_mode,
      render_backend=args.render_backend,
      sensor_configs=dict(shader_pack=args.shader),
      human_render_camera_configs=dict(shader_pack=args.shader),
      viewer_camera_configs=dict(shader_pack=args.shader),
      sim_backend=args.sim_backend,
    )

  if args.save_video:
    env = RecordEpisode(
      env,
      output_dir=str(output_dir),
      save_trajectory=False,
      save_video=True,
      video_fps=args.video_fps,
    )
  obs, info = env.reset(seed=args.seed) 

  # Prefer sensor image when available; fallback to env.render() for better
  # compatibility on headless systems where rgb+segmentation may crash.
  img = None
  if isinstance(obs, dict) and "sensor_data" in obs and len(obs["sensor_data"]) > 0:
    cam_name = next(iter(obs["sensor_data"]))
    img = obs["sensor_data"][cam_name]["rgb"][0].cpu().numpy()
  elif args.obs_mode != "state":
    frame = env.render()
    if frame is None:
      raise RuntimeError("Failed to get image for VLM: sensor_data missing and env.render() returned None")
    img = frame
  else:
    # In state mode, many tasks do not provide sensor_data. Try rendering one
    # frame, and fallback to a blank placeholder if rendering is unavailable.
    frame = env.render()
    if frame is not None:
      img = np.asarray(frame)
      if img.ndim == 4:
        img = img[0]
      save_pics(img, output_dir / "initial_frame.png")
    else:
      img = np.zeros((224, 224, 3), dtype=np.uint8)

  objs_all = []
  # 遍历刚体对象集合 scene.actors
  for name, actor in env.unwrapped.scene.actors.items():
    objs_all.append({
    "name": name,
    "type": "actor",
    # "pose": actor.pose.raw_pose[0].cpu().numpy() # [x,y,z,qw,qx,qy,qz]
    })
  # 遍历关节对象集合 scene.articulations
  for name, art in env.unwrapped.scene.articulations.items():
    objs_all.append({
    "name": name,
    "type": "articulation",
    # "pose": art.pose.raw_pose[0].cpu().numpy()
    })

  agent = Agents(image=img, task_description=task, model=args.model)

  _, plan = agent.multi_agent_vision_planning(objs_all)

  low_level_plan = gen_low_level_plan(plan)

  planner = Controller(env)
  execute_low_level_plan(planner, low_level_plan)

  # Ensure at least one recorded transition so RecordEpisode can write mp4
  if args.save_video and hasattr(env, "render_images") and len(env.render_images) < 2:
    noop = np.zeros_like(env.action_space.sample(), dtype=np.float32)
    env.step(noop)

  if args.save_video and hasattr(env, "flush_video"):
    env.flush_video(name="final_video")
    print(f"[run] saved video to: {output_dir / 'final_video.mp4'}")
  env.close()




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run ManiSkill with VLM integration.")
  parser.add_argument("--scene",type=str, default="LiftCube-v0", help="ManiSkill environment to run")
  parser.add_argument("--tasks",type=str, default="Move the ball from left to right", help="task prompt")
  parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save results")
  parser.add_argument("--max_episode_steps", type=int, default=250, help="Maximum steps per episode")
  parser.add_argument("--obs-mode", type=str, default="state", help="Observation mode, e.g. state/rgb/rgb+segmentation")
  parser.add_argument("--model",type=str, default="gpt-4o", help="Model API to use")
  parser.add_argument("--save-video", action="store_true", help="Enable video recording. Disable in unstable headless setups.")
  parser.add_argument(
		"--video-fps",
		type=int,
		default=20,
		help="FPS of the saved video. Lower FPS makes the same number of frames play longer.",
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
    choices=["rgb_array", "sensors", "all", "none"],
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
  parser.add_argument("--seed", type=int, default=0, help="Episode seed")
  args = parser.parse_args()
  run(args)

