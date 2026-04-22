import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class APIConfig:
    """API endpoint configuration. Keys read from environment variables."""

    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "sk-31191053cbb54460a55393173ec3892a")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    qwen_api_key: str = field(
        default_factory=lambda: os.getenv(
            "QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", "sk-1a68ae54b3fd424f91d0979d7b67b491")
        )
    )
    qwen_base_url: str = field(
        default_factory=lambda: os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-zk280d44e707a4a809a4c467266a213db66693bf03745f72")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1")
    )


@dataclass
class ModelConfig:
    """Default model assignments."""

    vlm_model: str = "qwen-vl-max"
    low_level_planner_model: str = "deepseek-chat"
    judge_model: str = "deepseek-chat"
    generate_model: str = "deepseek-chat"


@dataclass
class ControlConfig:
    """ManiSkill tabletop control defaults."""

    control_mode: str = "pd_ee_delta_pose"
    fallback_control_modes: Tuple[str, ...] = ("pd_joint_delta_pos", "pd_joint_pos")
    max_episode_steps: int = 240
    obs_mode: str = "state"
    render_mode: str = "rgb_array"
    render_backend: str = "cpu"
    sim_backend: str = "cpu"
    shader: str = "default"


@dataclass
class ExecutionConfig:
    """Execution loop parameters."""

    max_replan_attempts: int = 3
    replan_on_failure: bool = True
    feedback_capture_interval: int = 1


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    api: APIConfig = field(default_factory=APIConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


def load_config() -> PipelineConfig:
    """Load configuration with defaults and environment variable overrides."""
    return PipelineConfig()
