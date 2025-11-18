import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gate")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")


    # haohh Geometry encoder control config
    stage: str = field(default="")



@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    video_max_total_pixels: int = field(default=1664 * 28 * 28)
    video_min_total_pixels: int = field(default=256 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    model_type: str = field(default="qwen2.5vl")
    vsi_590k_dataRoot: str = field(default="/remote-home/share/_datasets/VSI-590k")

@dataclass    
class RLArguments:
    # 开关与阶段
    rl_stage: str = field(default="")        # "", "grpo"；当与 DataArguments.stage 一致为 "grpo" 时启用 RL
    # 数据
    rl_data_path: Optional[str] = None       # 指向 json/jsonl（包含 prompt / images / solution）
    rl_prompt_column: str = field(default="prompt")
    rl_image_column: str = field(default="images")
    rl_solution_column: str = field(default="solution")
    # GRPO 主要超参（与 TRL GRPOConfig 对齐）
    rl_num_generations: int = field(default=8)             # 组内采样数（group size）
    rl_reward_weights: Optional[List[float]] = None        # [alpha, beta, lambda]
    rl_max_prompt_length: Optional[int] = None             # VLM 建议 None，避免截断图像 token（官方建议）  # noqa
    rl_max_completion_length: int = field(default=512)
    rl_beta: float = field(default=0.0)                    # KL 系数（=0 则不加载 reference model）
    rl_scale_rewards: str = field(default="group")         # ["group","batch","none"]
    rl_loss_type: str = field(default="dapo")              # ["dapo","dr_grpo","grpo","cispo"]
    # 采样相关
    rl_temperature: float = field(default=1.0)
    rl_top_p: float = field(default=1.0)
    rl_use_vllm: bool = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
