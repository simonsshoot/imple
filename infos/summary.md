# 任务学习笔记
## 基础task任务
在ManiSkill/mani_skill/envs/tasks
顶层
- empty_env.py: 空环境占位，用于最小化流程测试
- rotate_cube.py: 旋转方块到目标姿态

control
- ant.py: 经典 Ant 运动控制任务
- cartpole.py: 小车倒立摆平衡
- hopper.py: Hopper 跳跃控制
- humanoid.py: 人形运动控制

dexterity
- insert_flower.py: 花形插销插入孔槽
- rotate_single_object_in_hand.py: 手内物体旋转
- rotate_valve.py: 旋转阀门到目标角度

digital_twins
- base_env.py: 数字孪生任务基类

drawing
- draw.py: 通用绘制任务
- draw_svg.py: 按 SVG 路径绘制
- draw_triangle.py: 绘制三角形

fmb
- fmb.py: 基础模型基准任务入口

humanoid
- humanoid_pick_place.py: 人形拾取与放置
- humanoid_stand.py: 人形站立平衡
- transport_box.py: 搬运箱子到目标位置

mobile_manipulation
- open_cabinet_drawer.py: 打开柜子抽屉

quadruped
- quadruped_reach.py: 四足到达目标
- quadruped_spin.py: 四足原地旋转

tabletop
- assembling_kits.py: 组装零件套件
- lift_peg_upright.py: 提起并竖直放置插销
- peg_insertion_side.py: 侧向插入插销
- pick_clutter_ycb.py: 在杂乱 YCB 物体中抓取
- pick_cube.py: 抓取方块
- pick_cube_cfgs.py: 抓取方块任务配置
- pick_single_ycb.py: 抓取单个 YCB 物体
- place_sphere.py: 放置球体到目标位置
- plug_charger.py: 插入充电器
- poke_cube.py: 推碰方块到目标区域
- pull_cube.py: 拉回方块
- pull_cube_tool.py: 使用工具拉方块
- push_cube.py: 推动方块到目标
- push_t.py: 推动 T 形物体
- roll_ball.py: 滚动球体到目标
- stack_cube.py: 堆叠方块
- stack_pyramid.py: 堆叠成金字塔
- turn_faucet.py: 旋转水龙头把手
- two_robot_pick_cube.py: 双机器人抓取方块
- two_robot_stack_cube.py: 双机器人堆叠方块

## 任务构建
官方文档的教程那一栏给出了自定义任务，可以使用mani_skill/envs/tasks/tabletop/push_cube.py做模版.
这里同样介绍了初始化：
“所有任务都由各自的类定义，并且必须继承自 BaseEnv ，这与许多其他机器人学习仿真框架的设计类似。您还必须使用装饰器注册该类，以便将来可以通过 ` gym.make(env_id=...) 命令轻松创建环境。”

```python
import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

@register_env("PushCube-v1", max_episode_steps=50)
class PushCubeEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

在任何任务开始时，必须将所有对象（机器人、资产等）加载到每个并行环境（也称为子场景）中。在_load_scene中完成（其目的仅仅是加载具有初始姿态的对象，以确保它们在第一步中不会发生碰撞）
对于机器人添加：系统会默认在原点添加
要定义机器人的初始姿态，需要重写 _load_agent 函数。以上 PushCube 示例就是这么做的。建议为所有物体设置初始姿态，以确保它们在生成时不会与其他物体发生碰撞。这里我们将机器人生成在离地 1 米的高度，这样就不会与其他物体发生冲突。如果您打算生成多个机器人，可以将姿态列表传递给 _load_agent 函数。

## 构建动态Actor
动态Actor是指可以参与物理模拟，受运动学方程控制，可交互（抓取、碰撞）和响应外力的物体。maniskill通过SAPIEN 物理引擎创建：
```python
# 创建一个 Actor 构建器，用于定义和组装 Actor 的各种组件
builder = self.scene.create_actor_builder()

# 添加盒形碰撞体组件
builder.add_box_collision(
    half_size=[0.02] * 3,  # 每个维度（x, y, z）的半长，这里创建边长 0.04m 的立方体
)

# 添加盒形视觉组件（渲染外观）
builder.add_box_visual(
    half_size=[0.02] * 3,  # 视觉尺寸与碰撞体一致
    material=sapien.render.RenderMaterial(
        base_color=[1, 0, 0, 1],  # RGBA颜色，红色不透明立方体
    ),
)

# 设置初始位姿（位置和旋转）
# p=[x, y, z]: 位置坐标，z=0.02 表示底面在 z=0 平面
# q=[w, x, y, z]: 四元数表示旋转，[1,0,0,0] 表示无旋转
builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])

# 构建并命名 Actor
self.obj = builder.build(name="cube")
```
```python
class PushCubeEnv(BaseEnv):
    # ...
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
        )
        self.table_scene.build()
        # ...
```
TableSceneBuilder 非常适合轻松构建桌面任务，它会为您创建桌子和地板，并将取物机器人和熊猫机器人放置在合理的位置。

## 关于视频
推荐的方法是使用我们的 RecordEpisode 封装器，它支持单一环境和矢量化环境，并将视频和/或轨迹数据（ ManiSkill 格式 ）保存到磁盘。
示例代码：（这部分在官方代码的wrappers里面）

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
env = gym.make("PickCube-v1", num_envs=1, render_mode="rgb_array")
env = RecordEpisode(env, output_dir="videos", save_trajectory=True, trajectory_name="trajectory", save_video=True, video_fps=30)
env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
```
env.flush_video(name=args.video_name)  详见ManiSkill/mani_skill/utils/wrappers/record.py，作用是把当前缓存的帧 render_images 写成 mp4


## run代码说明
```python
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
```
PlaceSphere-v1：环境 ID。这个在ManiSkill/mani_skill/envs/tasks/tabletop/place_sphere.py中被注册了：@register_env("PlaceSphere-v1", max_episode_steps=50)
关于注册，可以参见demo，也可以在ManiSkill/mani_skill/utils/registration.py中阅读，它提供了一个类似于 OpenAI Gym 的环境注册机制，专门为 ManiSkill 机器人仿真环境设计。
obs_mode="state"：观测模式。state 表示低维状态观测；其他常见值还有 rgbd、pointcloud 等
control_mode="pd_ee_delta_pose"：控制模式。pd_ee_delta_pose 表示给末端执行器的增量位姿控制（PD）。不同机器人/任务支持的控制模式会不同
max-episode-steps：任务的最长回合步数
render_mode=args.render_mode：渲染输出类型。这里参数约束在 rgb_array / sensors / all。
render_backend=args.render_backend：渲染后端（CPU/GPU/none），通常在无头环境用 cpu
ensor_configs=dict(shader_pack=args.shader)：传感器相机的 shader 包（如 default、rt、rt-fast）。
human_render_camera_configs=dict(shader_pack=args.shader)：人类可视化相机的 shader 包。
viewer_camera_configs=dict(shader_pack=args.shader)：GUI 里 viewer 相机的 shader 包。
sim_backend=args.sim_backend：物理模拟后端，cpu / gpu / auto。

在 ManiSkill 里，环境通过注册器挂到 Gym：
注册逻辑在 registration.py
环境基类构造参数在 sapien_env.py
PlaceSphere 注册与实现在 place_sphere.py
这意味着：
传给 gym.make 的大多数参数，最终会进入 BaseEnv.init 或具体任务类 PlaceSphereEnv.init。

## 经典的env.step
作用：把动作送进仿真，返回 obs, reward, terminated, truncated, info。
在你脚本里 reward 被忽略（用下划线占位）。
一个step就是一个步数

## vlm
control_mode参数，该参数很重要，决定了三件事情：
你发给 env.step 的 action 各维度分别代表什么
动作空间大小和上下界
底层控制器如何把动作转成关节驱动目标（位置、速度、末端位姿等）
具体表现为：在创建环境时传入 control_mode、环境基类接收该参数（sapien_env.py）、机器人代理根据 control_mode 选择控制器（base_agent.py）、之后每一步 action 都按该控制器的规则解释并执行
```python
	ACTION_SCHEMA = {
		"dx": "float in [-1,1]",
		"dy": "float in [-1,1]",
		"dz": "float in [-1,1]",
		"droll": "float in [-1,1]",
		"dpitch": "float in [-1,1]",
		"dyaw": "float in [-1,1]",
		"gripper": "float in [-1,1], negative=close, positive=open",
		"reason": "short explanation string",
	}
```
  dx
  末端执行器在 x 方向的增量指令。
  正负号表示前后方向（具体朝向取决于任务坐标系）。
  dy
  末端执行器在 y 方向的增量指令。
  dz
  末端执行器在 z 方向的增量指令。
  通常正值上抬，负值下探。
  droll
  末端绕 x 轴旋转的增量指令（滚转）。
  dpitch
  末端绕 y 轴旋转的增量指令（俯仰）。
  dyaw
  末端绕 z 轴旋转的增量指令（偏航）。
  gripper
  夹爪开合指令：
  负值表示闭合，正值表示张开，0 附近表示保持。
  reason
  一行简短理由，说明这一步为什么这么动。
  不直接参与控制计算，但对调试很有用

## obs
### 1 reset / step 的返回结构

- env.reset(...) -> (obs, info)
- env.step(action) -> (obs, reward, terminated, truncated, info)

出处：
- ManiSkill/mani_skill/examples/demo_random_action.py:107,117

### 2 obs 顶层字段由 obs_mode 决定

统一入口：BaseEnv.get_obs

出处：
- ManiSkill/mani_skill/envs/sapien_env.py:501

#### A. obs_mode = "none"

obs = {}

出处：
- ManiSkill/mani_skill/envs/sapien_env.py:519-521

#### B. obs_mode = "state_dict"

obs = {
    "agent": {...},
    "extra": {...}
}

字段含义：
- agent：机器人本体观测（典型是本体感觉，如 qpos/qvel 等），由 agent.get_proprioception() 提供
- extra：任务相关额外观测（各任务自行定义）

出处：
- ManiSkill/mani_skill/envs/sapien_env.py:546-551
- ManiSkill/mani_skill/envs/sapien_env.py:553-560

#### C. obs_mode = "state"

先得到 state_dict 结构（agent + extra），再扁平化为 1D 状态向量。

字段含义：
- 本质信息仍是 agent + extra，只是被 flatten 成一个向量，便于 MLP 策略直接输入。

出处：
- ManiSkill/mani_skill/envs/sapien_env.py:521-523
- ManiSkill/mani_skill/envs/sapien_env.py:537-545

#### D. 视觉类模式（rgb / depth / rgbd / rgb+segmentation / pointcloud 等）

典型顶层结构：

{
    "agent": {...},
    "extra": {...},
    "sensor_param": {
        "cam_name": {
            "extrinsic_cv": ...,
            "cam2world_gl": ...,
            "intrinsic_cv": ...
        }
    },
    "sensor_data": {
        "cam_name": {
            "rgb": [N, H, W, 3],
            "depth": [N, H, W, 1],
            "segmentation": [N, H, W, 1],
            "position": [N, H, W, 3]
        }
    }
}

字段含义：
- sensor_param：每个相机的内外参与位姿矩阵
- sensor_data：每个相机输出的各模态图像张量
- cam_name：相机名（由环境/机器人传感器配置决定）

出处：
- ManiSkill/mani_skill/envs/sapien_env.py:627-634
- ManiSkill/mani_skill/envs/sapien_env.py:564-577
- ManiSkill/mani_skill/sensors/camera.py:245-249

### 3 sensor_data 分层访问方式

标准访问：
- obs["sensor_data"] -> 所有相机字典
- obs["sensor_data"][cam_name] -> 某个相机所有模态
- obs["sensor_data"][cam_name]["rgb"] -> 该相机 RGB 批量
- obs["sensor_data"][cam_name]["rgb"][0] -> 第 0 个并行环境图像

示例：
- cam_name = next(iter(obs["sensor_data"]))
- img = obs["sensor_data"][cam_name]["rgb"][0].cpu().numpy()

其中 img 常见形状为 (H, W, 3)。

出处：
- ManiSkill/docs/source/user_guide/tutorials/domain_randomization.md:308-309
- ManiSkill/mani_skill/examples/demo_vis_segmentation.py:126-130

### 4 各视觉字段的数据类型与形状

官方标准模态定义：
- rgb: torch.uint8, [H, W, 3]
- depth: torch.int16, [H, W]
- segmentation: torch.int16, [H, W]
- position: torch.float32, [H, W, 3]

实际在环境返回时通常带 batch 维，并且 depth/segmentation 常见为 [N, H, W, 1]。

出处：
- ManiSkill/mani_skill/render/shaders.py:21-25
- ManiSkill/mani_skill/examples/demo_vis_segmentation.py:129-130

### 5 obs_mode 与可用字段的关系

obs_mode 解析逻辑支持：
- rgbd
- pointcloud
- sensor_data
- 以及任意组合字符串（如 rgb+depth+segmentation）

出处：
- ManiSkill/mani_skill/envs/utils/observations/__init__.py:37-90

## env

### 1 env 是什么

在 ManiSkill 中，env 通常是任务类（继承 BaseEnv）经 gym.make(...) 创建后的实例（可能再被 wrapper 包裹）。

出处：
- ManiSkill/mani_skill/envs/sapien_env.py:38 (BaseEnv)
- ManiSkill/mani_skill/utils/registration.py:160-172 (make)

### 2 env 构造参数含义

env = gym.make(scene, ...)

关键字段：
- scene：环境 ID，例如 PlaceSphere-v1
- obs_mode：观测模式，决定 obs 结构
- control_mode：控制器解释 action 的方式
- max_episode_steps：最大步数（由 TimeLimit/向量包装逻辑处理）
- render_mode：渲染模式（human/rgb_array/sensors/all）
- render_backend：渲染后端（cpu/gpu/none）
- sim_backend：物理仿真后端（auto/physx_cpu/physx_cuda）
- sensor_configs：观测相机配置（例如 shader_pack）
- human_render_camera_configs：人类可视化相机配置
- viewer_camera_configs：GUI 观察相机配置

出处：
- ManiSkill/mani_skill/envs/sapien_env.py:44-122 (BaseEnv 构造参数说明)
- ManiSkill/mani_skill/envs/sapien_env.py:127-134 (SUPPORTED_RENDER_MODES)
- ManiSkill/mani_skill/examples/demo_random_action.py:71-91 (实参示例)

### 3 env 常用属性与方法

- env.observation_space：当前 obs 的空间描述
- env.action_space：动作空间
- env.reset(seed=...)：重置并返回初始 obs/info
- env.step(action)：推进一步
- env.close()：释放资源

出处：
- ManiSkill/mani_skill/examples/demo_random_action.py:98-120

### 4 一句话总结

- env 是“任务+仿真+渲染+控制器+传感器”的统一接口。
- obs 是 env 在当前 obs_mode 下给出的观测快照，字段是否包含 sensor_data 完全由 obs_mode 决定。

## env.observation_space
_shape​ (Optional[tuple[int, ...]])
作用：用于定义空间中元素的形状。如果空间中的元素是NumPy数组（例如，图像观察或连续动作向量），此字段会以不可变元组的形式存储其维度。
示例：对于一个连续动作空间，其形状可能是 (3,)，表示一个三维向量。
dtype​ (Optional[np.dtype])
作用：定义空间中元素的数据类型。它被存储为NumPy的dtype对象，确保了类型检查的严格性和与NumPy生态的兼容性。
示例：常用的类型包括np.float32（用于连续值）、np.int64（用于离散值）或np.uint8（用于图像像素）。
_np_random​ (Optional[np.random.Generator])

## env.unwrapped

`self.u = env.unwrapped` 的本质：
- `env` 可能被多层 wrapper 包裹（如 `RecordEpisode`）。
- `env.unwrapped` 返回最底层环境对象（ManiSkill 的 `BaseEnv` 子类实例）。

在当前项目里（`imple/controller.py`），`u` 被用于直接访问底层场景和机器人：
- `self.u = env.unwrapped`（`imple/controller.py:30`）
- `self.u.scene`（`imple/controller.py:68`）
- `self.u.agent.tcp.pose.p`（`imple/controller.py:175`）
- `self.u.agent.is_grasping(...)`（`imple/controller.py:371-373`）

### 1) `u` 顶层字段（BaseEnv 关键字段）

| 字段 | 数据结构/类型 | 含义 | 出处 |
|---|---|---|---|
| `u.backend` | `sim-and-render backend 对象` | 仿真后端+渲染后端的组合配置入口 | `ManiSkill/mani_skill/envs/sapien_env.py:238` |
| `u.device` | `torch.device` | 环境主设备（CPU/CUDA） | `ManiSkill/mani_skill/envs/sapien_env.py:240` |
| `u.sim_config` | `SimConfig(dataclass)` | 仿真参数总配置（频率、内存、接触参数等） | `ManiSkill/mani_skill/envs/sapien_env.py:265` |
| `u._obs_mode` | `str` | 当前观测模式（如 `state`/`rgbd`） | `ManiSkill/mani_skill/envs/sapien_env.py:295` |
| `u.render_mode` | `Optional[str]` | 当前渲染模式（`human/rgb_array/sensors/all`） | `ManiSkill/mani_skill/envs/sapien_env.py:313` |
| `u.scene` | `ManiSkillScene` | 场景对象，包含 actor/articulation/渲染/物理状态接口 | `ManiSkill/mani_skill/envs/sapien_env.py:1220` |
| `u.agent` | `BaseAgent \| MultiAgent \| None` | 机器人代理（控制器、tcp、动作空间等） | `ManiSkill/mani_skill/envs/sapien_env.py:434` |
| `u.action_space` | `gym.Space` | 动作空间（通常来自 `agent.action_space`） | `ManiSkill/mani_skill/envs/sapien_env.py:335` |
| `u.single_action_space` | `gym.Space` | 非 batch 的单环境动作空间 | `ManiSkill/mani_skill/envs/sapien_env.py:335-339` |
| `u.single_observation_space` | `gym.Space` | 非 batch 的单环境观测空间（cached property） | `ManiSkill/mani_skill/envs/sapien_env.py:374` |
| `u.observation_space` | `gym.Space` | batch 后的观测空间（cached property） | `ManiSkill/mani_skill/envs/sapien_env.py:379` |
| `u._elapsed_steps` | `torch.Tensor[int32], shape=[num_envs]` | 每个并行环境当前步数计数 | `ManiSkill/mani_skill/envs/sapien_env.py:321-324` |
| `u._last_obs` | `Any` | 上一次返回的观测缓存 | `ManiSkill/mani_skill/envs/sapien_env.py:325-326` |

说明：字段带前缀 `_` 往往是“内部字段”，可读但不建议随意写。

### 2) `u` 常用方法与返回结构

| 方法 | 返回结构 | 含义 | 出处 |
|---|---|---|---|
| `u.reset(...)` | `(obs, info)` | 重置环境并返回初始观测和信息 | `ManiSkill/mani_skill/examples/demo_random_action.py:107` |
| `u.step(action)` | `(obs, reward, terminated, truncated, info)` | 推进一步仿真 | `ManiSkill/mani_skill/examples/demo_random_action.py:117` |
| `u.get_state_dict()` | `dict` | 获取可恢复的环境状态字典 | `ManiSkill/mani_skill/envs/sapien_env.py:1272` |
| `u.get_state()` | `torch.Tensor` | 将 state_dict 扁平化后的向量状态 | `ManiSkill/mani_skill/envs/sapien_env.py:1284` |
| `u.set_state_dict(state, env_idx=None)` | `None` | 用状态字典恢复环境 | `ManiSkill/mani_skill/envs/sapien_env.py:1291` |
| `u.render()` | `np.ndarray \| tiled image \| viewer` | 按 `render_mode` 输出渲染结果 | `ManiSkill/mani_skill/envs/sapien_env.py:1416` |
| `u.close()` | `None` | 关闭环境并释放资源 | `ManiSkill/mani_skill/envs/sapien_env.py:1240` |

### 3) `u.scene` 与 `u.agent`（controller 实际依赖字段）

#### 3.1 `u.scene`

| 字段/方法 | 数据结构 | 含义 | 出处 |
|---|---|---|---|
| `u.scene.actors` | `dict[str, Actor]` | 场景中所有 actor，按名字索引 | `imple/controller.py:69`（使用）, `ManiSkill/mani_skill/envs/sapien_env.py:1261`（遍历 actors） |
| `u.scene.articulations` | `dict[str, Articulation]` | 场景中所有 articulation，按名字索引 | `imple/controller.py:71`（使用）, `ManiSkill/mani_skill/envs/sapien_env.py:1263`（遍历 articulations） |
| `u.scene.get_sim_state()` | `dict` | 仿真状态字典（被 `get_state_dict` 直接返回） | `ManiSkill/mani_skill/envs/sapien_env.py:1276` |

#### 3.2 `u.agent`

| 字段/方法 | 数据结构 | 含义 | 出处 |
|---|---|---|---|
| `u.agent.tcp.pose.p` | `torch.Tensor`, 常见 `[N,3]` 或 `[3]` | 末端执行器位置（controller 用于追踪/逼近） | `imple/controller.py:175` |
| `u.agent.is_grasping(actor)` | `torch.Tensor[bool]`（常见 batch） | 判定当前是否抓住目标 actor | `imple/controller.py:371-373` |
| `u.agent.action_space` | `gym.Space` | 机器人动作空间（赋给 `u.action_space`） | `ManiSkill/mani_skill/envs/sapien_env.py:335` |
| `u.agent.get_controller_state()` | `dict` | 控制器内部状态（并入 `get_state_dict`） | `ManiSkill/mani_skill/envs/sapien_env.py:1277` |

### 4) `u` 的完整字段如何“全量查看”

因为 `BaseEnv` 里字段较多（且不同任务子类还会新增任务字段），最可靠方式是运行时 introspection：

```python
def dump_unwrapped_fields(env):
    u = env.unwrapped
    print("type:", type(u))

    # 仅展示非魔法字段
    attrs = [k for k in dir(u) if not k.startswith("__")]
    print("num attrs:", len(attrs))

    for k in attrs:
        try:
            v = getattr(u, k)
            print(f"{k:40s} | {type(v)}")
        except Exception as e:
            print(f"{k:40s} | <unreadable: {e}>")
```

补充：
- 如果你只想看“实例字段”（不含类方法），可看 `u.__dict__.keys()`。
- 如果要追溯某字段来源，优先在 `ManiSkill/mani_skill/envs/sapien_env.py` 搜该字段赋值位置。


## BaseAgent

BaseAgent 是 ManiSkill 里所有机器人代理的基类，负责：
- 加载机器人本体（URDF/MJCF -> Articulation）
- 管理控制器（control mode / action space / set_action）
- 提供本体观测与可恢复状态接口（get_proprioception / get_state / set_state）

定义位置：`ManiSkill/mani_skill/agents/base_agent.py:46`

### 1) 类级字段（通常由子类覆盖）

| 字段 | 数据结构/类型 | 含义 | 出处 |
|---|---|---|---|
| `uid` | `str` | 机器人唯一标识，用于注册与按名称实例化 | `ManiSkill/mani_skill/agents/base_agent.py:61` |
| `urdf_path` | `Union[str, None]` | URDF 文件路径（与 `mjcf_path` 二选一） | `ManiSkill/mani_skill/agents/base_agent.py:63` |
| `urdf_config` | `Union[str, dict, None]` | URDF 附加配置（材质、碰撞参数等） | `ManiSkill/mani_skill/agents/base_agent.py:65` |
| `mjcf_path` | `Union[str, None]` | MJCF 文件路径（与 `urdf_path` 二选一） | `ManiSkill/mani_skill/agents/base_agent.py:67` |
| `fix_root_link` | `bool` | 是否固定机器人根链接 | `ManiSkill/mani_skill/agents/base_agent.py:71` |
| `load_multiple_collisions` | `bool` | 是否加载多重凸碰撞体 | `ManiSkill/mani_skill/agents/base_agent.py:73` |
| `disable_self_collisions` | `bool` | 是否禁用自碰撞 | `ManiSkill/mani_skill/agents/base_agent.py:75` |
| `keyframes` | `dict[str, Keyframe]` | 预定义关键帧（用于 reset 到指定姿态） | `ManiSkill/mani_skill/agents/base_agent.py:80` |
| `robot` | `Articulation` | 机器人本体对象（pose/qpos/qvel 等入口） | `ManiSkill/mani_skill/agents/base_agent.py:83` |

### 2) Keyframe 数据结构

| 字段 | 数据结构/类型 | 含义 | 出处 |
|---|---|---|---|
| `pose` | `sapien.Pose` | 关键帧位姿 | `ManiSkill/mani_skill/agents/base_agent.py:38` |
| `qpos` | `Optional[Array]` | 关键帧关节位置 | `ManiSkill/mani_skill/agents/base_agent.py:40` |
| `qvel` | `Optional[Array]` | 关键帧关节速度 | `ManiSkill/mani_skill/agents/base_agent.py:42` |

### 3) 初始化后实例字段

| 字段 | 数据结构/类型 | 含义 | 出处 |
|---|---|---|---|
| `scene` | `ManiSkillScene` | 当前 agent 所在场景 | `ManiSkill/mani_skill/agents/base_agent.py:95` |
| `_control_freq` | `int` | 控制频率（Hz） | `ManiSkill/mani_skill/agents/base_agent.py:96` |
| `_agent_idx` | `Optional[str]` | 多机器人任务中的 agent 索引 | `ManiSkill/mani_skill/agents/base_agent.py:97` |
| `build_separate` | `bool` | 是否为每个并行环境分开构建机器人后再合并 | `ManiSkill/mani_skill/agents/base_agent.py:98` |
| `controllers` | `dict[str, BaseController]` | 已创建控制器缓存 | `ManiSkill/mani_skill/agents/base_agent.py:100` |
| `sensors` | `dict[str, BaseSensor]` | 机器人自带传感器集合 | `ManiSkill/mani_skill/agents/base_agent.py:102` |
| `supported_control_modes` | `list[str]` | 支持的控制模式名列表 | `ManiSkill/mani_skill/agents/base_agent.py:109` |
| `_default_control_mode` | `str` | 默认控制模式 | `ManiSkill/mani_skill/agents/base_agent.py:114` |
| `_control_mode` | `str` | 当前激活控制模式 | `ManiSkill/mani_skill/agents/base_agent.py:259` |
| `robot_link_names` | `list[str]` | 机器人 link 名称缓存 | `ManiSkill/mani_skill/agents/base_agent.py:231` |

### 4) 控制与动作接口（方法+返回结构）

| 接口 | 数据结构/返回类型 | 含义 | 出处 |
|---|---|---|---|
| `controller` | `BaseController` | 当前控制模式对应控制器 | `ManiSkill/mani_skill/agents/base_agent.py:291` |
| `action_space` | `gymnasium.spaces.Space` | 当前动作空间（可为 Dict 或 Box） | `ManiSkill/mani_skill/agents/base_agent.py:299` |
| `single_action_space` | `gymnasium.spaces.Space` | 单环境动作空间 | `ManiSkill/mani_skill/agents/base_agent.py:311` |
| `set_action(action)` | `None` | 设置下一步控制输入 | `ManiSkill/mani_skill/agents/base_agent.py:322` |
| `before_simulation_step()` | `None` | 仿真步前控制器预处理 | `ManiSkill/mani_skill/agents/base_agent.py:332` |

### 5) 观测与状态接口（方法+字段结构）

#### 5.1 `get_proprioception()` 返回结构

```python
{
    "qpos": Tensor/Array,
    "qvel": Tensor/Array,
    "controller": dict,  # 可选，控制器有状态时才出现
}
```

字段含义：
- `qpos`：关节位置。
- `qvel`：关节速度。
- `controller`：控制器内部状态。

出处：
- `ManiSkill/mani_skill/agents/base_agent.py:339-347`
- `ManiSkill/mani_skill/envs/sapien_env.py:556`（环境侧直接调用 `agent.get_proprioception()`）

#### 5.2 `get_state()` 返回结构

```python
{
    "robot_root_pose": Pose,
    "robot_root_vel": Tensor/Array,
    "robot_root_qvel": Tensor/Array,
    "robot_qpos": Tensor/Array,
    "robot_qvel": Tensor/Array,
    "controller": dict,
}
```

字段含义：
- `robot_root_pose`：机器人根链接位姿。
- `robot_root_vel`：根链接线速度。
- `robot_root_qvel`：根链接角速度。
- `robot_qpos`：全关节位置。
- `robot_qvel`：全关节速度。
- `controller`：控制器状态。

出处：
- `ManiSkill/mani_skill/agents/base_agent.py:361-375`

#### 5.3 `set_state(state, ignore_controller=False)`

输入 `state` 需要与 `get_state()` 同结构；若 `ignore_controller=False` 且存在 `controller` 字段，则会同步恢复控制器状态。

出处：
- `ManiSkill/mani_skill/agents/base_agent.py:378-396`

### 6) 抽象能力接口（由具体机器人实现）

| 接口 | 返回类型 | 含义 | 出处 |
|---|---|---|---|
| `is_grasping(object=None)` | `bool/Tensor[bool]` | 判断是否抓住目标或任意物体 | `ManiSkill/mani_skill/agents/base_agent.py:413` |
| `is_static(threshold)` | `bool/Tensor[bool]` | 判断是否静止 | `ManiSkill/mani_skill/agents/base_agent.py:427` |

说明：BaseAgent 里这两个方法是 `NotImplementedError`，需要子类实现。

### 7) BaseAgent 在环境中的来源链路

1. 子类通过 `@register_agent(...)` 注册到全局注册表 `REGISTERED_AGENTS`。  
出处：`ManiSkill/mani_skill/agents/registration.py:20,17,39`

2. BaseEnv 在 `_load_agent` 中根据 `robot_uids` 查注册表并实例化 agent。  
出处：`ManiSkill/mani_skill/envs/sapien_env.py:421-438`

3. 观测里 `obs["agent"]` 默认来自 `agent.get_proprioception()`。  
出处：`ManiSkill/mani_skill/envs/sapien_env.py:546-556`

### 8) 子类扩展示例（Panda）

`tcp` 不是 BaseAgent 保证字段，而是具体机器人子类在 `_after_init` 增加的字段。

| 字段/方法 | 数据结构/类型 | 含义 | 出处 |
|---|---|---|---|
| `self.tcp` | `Link` | 末端执行器 TCP link 引用 | `ManiSkill/mani_skill/agents/robots/panda/panda.py:233-235` |
| `tcp_pos` | `Tensor/Array` | TCP 位置 | `ManiSkill/mani_skill/agents/robots/panda/panda.py:274-276` |
| `tcp_pose` | `Pose` | TCP 位姿 | `ManiSkill/mani_skill/agents/robots/panda/panda.py:278-280` |
| `is_grasping(object)` | `Tensor[bool]` | Panda 具体抓取判定实现 | `ManiSkill/mani_skill/agents/robots/panda/panda.py:237-272` |

这也是为什么在你的控制器里直接访问 `u.agent.tcp.pose.p` 可行：当前环境机器人子类实现了 `tcp`。


## 目前的原子动作
注意，这些方法都是类的成员方法。
find(target_obj: str, obj_num: Optional[int]) -> str
含义：让机器人找到目标物体并移动到其附近。如果找不到物体则返回错误信息。
pick(obj_name: str, obj_num: Optional[int], manualInteract: bool = False) -> str
含义：让机器人抓取指定物体。如果物体不存在或不可抓取则返回错误信息。如果抓取失败，则使用软附着（soft attach）方式将物体与机器人绑定。
put(receptacle_name: str, obj_num: Optional[int]) -> str
含义：让机器人将当前抓取的物体放置到指定的容器（或位置）上。如果机器人没有抓取物体或找不到容器则返回错误信息。
slice(obj_name: str, obj_num: Optional[int]) -> str
含义：模拟对物体进行切片操作。实际上只是将物体的状态标记为已切片。
turn_on(obj_name: str, obj_num: Optional[int]) -> str
含义：打开物体（例如电器）。实际上只是将物体的状态标记为打开。
turn_off(obj_name: str, obj_num: Optional[int]) -> str
含义：关闭物体。实际上只是将物体的状态标记为关闭。
drop() -> str
含义：让机器人放下当前抓取的物体（松开夹爪）。如果机器人没有抓取物体则返回错误信息。
throw() -> str
含义：让机器人扔出当前抓取的物体。模拟一个向前扔的动作，然后松开夹爪。
break_(obj_name: str, obj_num: Optional[int]) -> str
含义：打破物体。实际上只是将物体的状态标记为已打破。
cook(obj_name: str, obj_num: Optional[int]) -> str
含义：烹饪物体。实际上只是将物体的状态标记为已烹饪。
dirty(obj_name: str, obj_num: Optional[int]) -> str
含义：弄脏物体。实际上只是将物体的状态标记为脏。
clean(obj_name: str, obj_num: Optional[int]) -> str
含义：清洁物体。实际上只是将物体的状态标记为干净。
fillLiquid(obj_name: str, obj_num: Optional[int], liquid_name: str) -> str
含义：将物体填充液体。实际上只是将物体的状态标记为填充了指定液体。
emptyLiquid(obj_name: str, obj_num: Optional[int]) -> str
含义：清空物体中的液体。实际上只是将物体的状态标记为未填充液体。
pour() -> str
含义：倾倒当前抓取的物体中的液体。模拟一个倾倒的动作，并将物体的液体状态清空。
close(obj_name: str, obj_num: Optional[int]) -> str
含义：关闭物体（例如门、容器）。实际上只是将物体的状态标记为关闭。
open(obj_name: str, obj_num: Optional[int]) -> str
含义：打开物体（例如门、容器）。实际上只是将物体的状态标记为打开。
注意：以上很多动作（如slice、turn_on、turn_off等）并没有真正的物理效果，只是改变了物体在对象状态字典中的状态。
还有一些辅助方法，如move_to、hold_position等，这些是内部使用的，不对外暴露为原子动作

## 原子动作的实现

1) 原子动作是如何从字符串指令一路落到 ManiSkill 的 `env.step(action)` 的；
2) 以 `move_left` 为例，`_move_by` / `_move_to` / `_build_action` 分别做了什么。

### 1) 端到端执行链路

| 阶段 | 代码位置 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| 低层计划生成 | `imple/run.py:94` | 高层计划文本 | `List[str]` | 由 `agent.generate_low_level_plan(...)` 生成低层动作字符串列表 |
| 执行器入口 | `imple/run.py:102` | `low_level_plan` | 执行日志 | 调用 `execute_low_level_plan(planner, low_level_plan)` |
| 指令分发 | `imple/utils.py:429,450` | 每条字符串指令 | `ret_dict` | `planner.llm_skill_interact(step)` 将文本映射到具体 Controller 方法 |
| 动作解析 | `imple/controller.py` 的 `llm_skill_interact` | 如 `move_right cube` | 方法调用 | 解析前缀，路由到 `move_right/find/pick/put/...` |
| 控制命令构造 | `imple/controller.py:150` | `delta_xyz` + `grip` | `np.ndarray action` | `_build_action` 按 `pd_ee_delta_pose` 约定填充动作向量 |
| 仿真步执行 | `imple/controller.py:165,167` | `action` | `obs,reward,terminated,truncated,info` | `_step` 调用 `self.env.step(action)` |
| ManiSkill 环境步进 | `ManiSkill/mani_skill/envs/sapien_env.py:1042` | `action` | 新状态 | BaseEnv `step` 将动作传给 agent/controller 并推进物理引擎 |

补充：
- `control_mode="pd_ee_delta_pose"` 在 `run.py` 里设置（`imple/run.py:25,34`），决定 action 各维度语义。  
- BaseEnv 在初始化时把 `action_space` 绑定到 agent/controller（`ManiSkill/mani_skill/envs/sapien_env.py:335`）。

### 2) `move_left` 的具体实现路径

#### 2.1 `move_left` 只是一个薄封装

定义：`imple/controller.py:423`

逻辑：
- 调用 `_move_by(delta_xyz=[0.0, +0.10, 0.0], steps=18, grip=...)`
- 如果当前持物（`self.held_object_name` 非空），`grip=-1.0`（保持夹紧）；否则 `grip=1.0`（张开/不夹持）

`move_right/move_forward/move_back` 同理，只是 `delta_xyz` 的方向不同：
- `move_right`: `imple/controller.py:428`
- `move_forward`: `imple/controller.py:432`
- `move_back`: `imple/controller.py:436`

#### 2.2 `_move_by` 把“相对位移”变成“目标位姿”

定义：`imple/controller.py:417`

逻辑：
- 读取当前末端位置 `start = _get_tcp_pos()`
- 计算目标 `target = start + delta_xyz`
- 对 z 做安全下限裁剪（`target[2] >= 0.05`）
- 调用 `_move_to(target, grip, steps)` 执行闭环逼近

所以 `_move_by` 是“相对移动原语”，不直接 step，而是委托 `_move_to`。

#### 2.3 `_move_to` 做闭环控制（逐步逼近）

定义：`imple/controller.py:185`

每个内部迭代：
1. 当前末端位置 `ee = _get_tcp_pos()`
2. 误差 `target - ee`
3. 乘增益 `gain` 并裁剪到 `[-1, 1]` 得到 `delta`
4. `_build_action(delta, grip)` 组动作向量
5. `_step(action)` 推进仿真一步
6. 若误差范数小于 `tol` 提前停止

这是典型笛卡尔空间的简化比例控制回路。

#### 2.4 `_build_action` 如何落到 ManiSkill action

定义：`imple/controller.py:150`

逻辑：
- 从 `env.action_space.sample()` 拿一个同形状模板并清零
- 将 `action[:3]`（或 batch 的 `...,:3`）写入 `delta_xyz`
- 若维度允许且给了 `grip`，写入 `action[6]`（夹爪开合）

这与 `pd_ee_delta_pose` 的动作语义对应：
- 前 3 维是末端位置增量；
- 夹爪通道用第 7 维（索引 6）。

#### 2.5 `_step` 如何触发 ManiSkill 物理更新

定义：`imple/controller.py:165`

逻辑：
- 调用 `self.env.step(action)`（`imple/controller.py:167`）
- 缓存 `obs/reward/terminated/truncated/info`
- 若启用 soft attach，执行 `_soft_follow_held_object()`
- 若回合终止则中断 repeat

在 ManiSkill 侧，`BaseEnv.step` 负责：
- 处理 action 格式/维度；
- 将 action 传给 agent/controller（见 `BaseAgent.set_action`: `ManiSkill/mani_skill/agents/base_agent.py:322`）；
- 每个仿真子步调用 controller 的 `before_simulation_step`（`ManiSkill/mani_skill/agents/base_agent.py:332`）；
- 推进物理并返回标准 Gym 五元组。

### 3) 其他原子动作与底层实现模式

可分三类：

1. 运动原语：`move_*` / `find` / `pick` / `put`
说明：最终都会走 `_move_to -> _build_action -> _step -> env.step`，是真实仿真步进。

2. 状态原语：`open/close/slice/turn on/turn off/break/cook/dirty/clean/fillLiquid/emptyLiquid`
说明：主要更新 `self.object_states` 的符号状态（并不总是对应真实物理变化）。

3. 释放/抛掷：`drop/throw/pour`
`drop` 通过夹爪张开释放；
`throw` 先前移再释放；
`pour` 通过短轨迹模拟倾倒并清空液体状态。

