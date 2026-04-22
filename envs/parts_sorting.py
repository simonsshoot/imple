"""PartsSorting-v1: Industrial parts sorting environment for ManiSkill3.

A Panda robot sorts 3 colored parts (red cube, blue cylinder, green sphere)
into 3 matching-color bins (trays with walls) on a tabletop.
"""

from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# -- Colour palette --------------------------------------------------------
RED = [0.86, 0.05, 0.05, 1.0]
BLUE = [0.05, 0.17, 0.76, 1.0]
GREEN = [0.07, 0.75, 0.27, 1.0]

# Darker shades for bin walls
RED_DARK = [0.60, 0.04, 0.04, 1.0]
BLUE_DARK = [0.04, 0.12, 0.53, 1.0]
GREEN_DARK = [0.05, 0.52, 0.19, 1.0]

WALL_COLORS = {"red": RED_DARK, "blue": BLUE_DARK, "green": GREEN_DARK}
FLOOR_COLORS = {"red": RED, "blue": BLUE, "green": GREEN}


@register_env("PartsSorting-v1", max_episode_steps=200)
class PartsSortingEnv(BaseEnv):
    """
    **Task Description:**
    Sort 3 coloured parts into their matching-colour bins.

    **Randomisations:**
    - Part positions are randomised on the table in x∈[-0.15, -0.05], y∈[-0.15, 0.15].
    - Bin positions are fixed at y = -0.15, 0.0, 0.15 (x = 0.05).

    **Success Conditions:**
    - All 3 parts are inside their correct bins, robot is not grasping any part.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # Bin geometry
    bin_floor_half = np.array([0.04, 0.04, 0.003], dtype=np.float32)
    bin_wall_thickness = 0.004
    bin_wall_height = 0.025

    # Part geometry
    cube_half_size = 0.02
    sphere_radius = 0.02
    cylinder_radius = 0.015
    cylinder_half_length = 0.02

    # Fixed bin center positions (x, y) — z computed from floor half-height
    bin_xy_positions = [
        [0.05, -0.15],  # red bin
        [0.05, 0.00],   # blue bin
        [0.05, 0.15],   # green bin
    ]

    # Match colour index: parts[i] -> bins[i]
    COLOR_NAMES = ["red", "blue", "green"]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.05, 0, 0.05])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128,
                fov=np.pi / 2, near=0.01, far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512,
            fov=1, near=0.01, far=100,
        )

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _build_bin(self, name: str, color_name: str):
        """Build a tray-style bin: floor + 4 walls, kinematic."""
        builder = self.scene.create_actor_builder()

        fh = self.bin_floor_half
        wt = self.bin_wall_thickness
        wh = self.bin_wall_height

        floor_color = FLOOR_COLORS[color_name]
        wall_color = WALL_COLORS[color_name]

        floor_mat = sapien.render.RenderMaterial(base_color=floor_color)
        wall_mat = sapien.render.RenderMaterial(base_color=wall_color)

        # Floor
        builder.add_box_collision(sapien.Pose([0, 0, 0]), half_size=fh)
        builder.add_box_visual(sapien.Pose([0, 0, 0]), half_size=fh, material=floor_mat)

        # 4 walls: front (+x), back (-x), left (+y), right (-y)
        wall_defs = [
            # (pose_offset, half_size)
            ([fh[0] + wt / 2, 0, fh[2] + wh / 2], [wt / 2, fh[1], wh / 2]),  # front
            ([-fh[0] - wt / 2, 0, fh[2] + wh / 2], [wt / 2, fh[1], wh / 2]),  # back
            ([0, fh[1] + wt / 2, fh[2] + wh / 2], [fh[0] + wt, wt / 2, wh / 2]),  # left
            ([0, -fh[1] - wt / 2, fh[2] + wh / 2], [fh[0] + wt, wt / 2, wh / 2]),  # right
        ]
        for offset, hs in wall_defs:
            pose = sapien.Pose(offset)
            builder.add_box_collision(pose, half_size=hs)
            builder.add_box_visual(pose, half_size=hs, material=wall_mat)

        return builder.build_kinematic(name=name)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # --- Bins (kinematic) ---
        self.red_bin = self._build_bin("red_bin", "red")
        self.blue_bin = self._build_bin("blue_bin", "blue")
        self.green_bin = self._build_bin("green_bin", "green")
        self.bins = [self.red_bin, self.blue_bin, self.green_bin]

        # --- Parts (dynamic) ---
        self.red_cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size,
            color=RED, name="red_cube", body_type="dynamic",
        )
        self.blue_cylinder = actors.build_cylinder(
            self.scene, radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
            color=BLUE, name="blue_cylinder", body_type="dynamic",
        )
        self.green_sphere = actors.build_sphere(
            self.scene, radius=self.sphere_radius,
            color=GREEN, name="green_sphere", body_type="dynamic",
        )
        self.parts = [self.red_cube, self.blue_cylinder, self.green_sphere]

        # Height of each part above table surface when resting
        self._part_rest_z = [
            self.cube_half_size,       # cube rests on face
            self.cylinder_radius,      # cylinder on its curved side
            self.sphere_radius,        # sphere on surface
        ]

    # ------------------------------------------------------------------
    # Episode init
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- Set bin positions (fixed) ---
            q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(b, -1)
            for i, bin_actor in enumerate(self.bins):
                pos = torch.zeros((b, 3))
                pos[:, 0] = float(self.bin_xy_positions[i][0])
                pos[:, 1] = float(self.bin_xy_positions[i][1])
                pos[:, 2] = float(self.bin_floor_half[2])  # half floor thickness
                bin_actor.set_pose(Pose.create_from_pq(p=pos, q=q_identity))

            # --- Randomise part positions (no overlap) ---
            # Parts spawn in x∈[-0.15, -0.05], y∈[-0.15, 0.15]
            for i, part in enumerate(self.parts):
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = torch.rand(b) * 0.10 - 0.15  # [-0.15, -0.05]
                xyz[:, 1] = torch.rand(b) * 0.30 - 0.15  # [-0.15, 0.15]
                xyz[:, 2] = float(self._part_rest_z[i])
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True,
                )
                part.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _part_in_bin(self, part_idx: int) -> torch.Tensor:
        """Check if part[i] is inside bin[i]. Returns bool tensor (B,)."""
        part_pos = self.parts[part_idx].pose.p
        bin_pos = self.bins[part_idx].pose.p
        offset = part_pos - bin_pos
        xy_ok = torch.linalg.norm(offset[..., :2], axis=1) <= (
            self.bin_floor_half[0] - 0.005
        )
        # Part should be above bin floor and below top of walls
        z_above_floor = offset[..., 2] > -0.005
        z_below_top = offset[..., 2] < (
            self.bin_floor_half[2] + self.bin_wall_height + 0.02
        )
        return xy_ok & z_above_floor & z_below_top

    def evaluate(self):
        per_part = [self._part_in_bin(i) for i in range(3)]
        all_sorted = per_part[0] & per_part[1] & per_part[2]

        any_grasped = torch.zeros(all_sorted.shape, dtype=torch.bool, device=self.device)
        for part in self.parts:
            any_grasped = any_grasped | self.agent.is_grasping(part)

        success = all_sorted & (~any_grasped)
        return {
            "success": success,
            "all_sorted": all_sorted,
            "any_grasped": any_grasped,
            "red_sorted": per_part[0],
            "blue_sorted": per_part[1],
            "green_sorted": per_part[2],
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            for i, (part, cname) in enumerate(zip(self.parts, self.COLOR_NAMES)):
                obs[f"{cname}_part_pose"] = part.pose.raw_pose
            for i, (bin_a, cname) in enumerate(zip(self.bins, self.COLOR_NAMES)):
                obs[f"{cname}_bin_pos"] = bin_a.pose.p
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        b = tcp_pos.shape[0]
        reward = torch.zeros(b, device=self.device)

        # Count already-sorted parts (0-3), each worth 4 points
        sorted_count = torch.zeros(b, device=self.device)
        for i in range(3):
            sorted_count += info[f"{self.COLOR_NAMES[i]}_sorted"].float()
        reward += sorted_count * 4.0

        # For the first unsorted part: reaching + grasping + placing reward
        for i in range(3):
            not_sorted = ~info[f"{self.COLOR_NAMES[i]}_sorted"]
            if not not_sorted.any():
                continue

            part_pos = self.parts[i].pose.p
            bin_pos = self.bins[i].pose.p

            # Reach reward
            tcp_to_part = torch.linalg.norm(tcp_pos - part_pos, axis=1)
            reach = 1.0 - torch.tanh(5.0 * tcp_to_part)

            # Grasp reward
            is_grasping = self.agent.is_grasping(self.parts[i])

            # Place reward (distance from part to bin center)
            part_to_bin = torch.linalg.norm(
                part_pos[..., :2] - bin_pos[..., :2], axis=1
            )
            place = 1.0 - torch.tanh(5.0 * part_to_bin)

            sub_reward = reach + is_grasping.float() * (1.0 + place)
            reward[not_sorted] = (reward + sub_reward)[not_sorted]
            break  # only reward the first unsorted part

        # Success bonus
        reward[info["success"]] = 15.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 15.0
