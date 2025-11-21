# -*- coding: utf-8 -*-
"""
把你们的 LazySupervisedDataset 适配为 VERL 的 RL 数据集。
会从 Hydra 的 config.data.* 中收集参数，包装成 DataArguments，交给 LazySupervisedDataset。
对外输出 RL 需要的字段：prompt / image / data_source / reward_model / extra_info。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from types import SimpleNamespace
import os

from torch.utils.data import Dataset
from transformers import AutoProcessor

# 你们已有的 SFT 数据集与常量
from qwen_vl.data.data_qwen import LazySupervisedDataset, DEFAULT_IMAGE_TOKEN

# 可选：有 dataclass 就用，没有就回退 SimpleNamespace
try:
    from qwen_vl.train.argument import DataArguments  # dataclass
    _HAS_DATAARGS = True
except Exception:
    DataArguments = SimpleNamespace  # type: ignore
    _HAS_DATAARGS = False


def _to_messages(convs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """将 {from: human/gpt, value: ...} 转为 HF chat 格式 [{role, content}, ...]。"""
    out = []
    for m in convs:
        role = m.get("from") or m.get("role")
        role = "user" if role == "human" else ("assistant" if role == "gpt" else role)
        out.append({"role": role, "content": m.get("value") or m.get("content", "")})
    return out


def _abs_join(base: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base or "", p))


def _build_data_args_from_cfg(
    cfg: Any,
    tokenizer: Any,
    processor: Any,
    is_train: bool,
    max_samples_arg: int,
) -> Any:
    """
    把 Hydra 的 config.data.* 映射/注入到 DataArguments：
      - 必要：dataset_use, stage, model_type, image_processor
      - 你要求的：max_prompt_length, max_response_length, train/val_max_samples
      - 其余：min/max_pixels, video_* 等（若无则给默认）
    """
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg

    # 先实例化 DataArguments；若 dataclass 有必填项，这里可按需补默认
    try:
        data_args = DataArguments()  # dataclass 有默认值时可以直接构造
    except TypeError:
        data_args = DataArguments  # SimpleNamespace 回退
        data_args = data_args()

    # === 基本项 ===
    setattr(data_args, "dataset_use", getattr(data_cfg, "dataset_use", "spar_234k,llava_hound_64k"))
    setattr(data_args, "stage",       getattr(data_cfg, "stage", "rl"))
    setattr(data_args, "model_type",  getattr(data_cfg, "model_type", "qwen2.5vl"))

    # image_processor：优先用 VERL 传入的 processor.image_processor；否则从模型路径再构造
    iproc = getattr(processor, "image_processor", None)
    if iproc is None:
        mdl_path = getattr(getattr(cfg, "actor_rollout_ref", None), "model", {}).get("path", None) \
                   if hasattr(cfg, "actor_rollout_ref") else None
        trust = bool(getattr(data_cfg, "trust_remote_code", False))
        if mdl_path is None:
            raise RuntimeError("Cannot resolve image_processor: both processor.image_processor and model.path are None")
        iproc = AutoProcessor.from_pretrained(mdl_path, trust_remote_code=trust).image_processor
    setattr(data_args, "image_processor", iproc)

    # === 你要求“额外传给 Adapter”并写入 DataArguments 的字段 ===
    # 注意：SFT 数据集未必直接用到它们，但放入 data_args 便于后续逻辑访问
    setattr(data_args, "max_prompt_length",   int(getattr(data_cfg, "max_prompt_length", 20480)))
    setattr(data_args, "max_response_length", int(getattr(data_cfg, "max_response_length", 512)))

    # train/val 的 max_samples：优先 main_ppo 传来的 max_samples 参数；否则读 data.* 的默认
    if max_samples_arg and max_samples_arg > 0:
        max_samples = int(max_samples_arg)
    else:
        key = "train_max_samples" if is_train else "val_max_samples"
        max_samples = int(getattr(data_cfg, key, -1))
    setattr(data_args, "max_samples", max_samples)

    # === 视觉/视频等其它可选项（用 getattr 给默认即可）===
    setattr(data_args, "max_pixels", int(getattr(data_cfg, "max_pixels", 1536)))
    setattr(data_args, "min_pixels", int(getattr(data_cfg, "min_pixels", 448)))
    setattr(data_args, "video_min_frames", int(getattr(data_cfg, "video_min_frames", 4)))
    setattr(data_args, "video_max_frames", int(getattr(data_cfg, "video_max_frames", 8)))
    setattr(data_args, "base_interval",    int(getattr(data_cfg, "base_interval", 2)))

    # 几何编码器（如需）
    if hasattr(data_cfg, "use_geometry_encoder"):
        setattr(data_args, "use_geometry_encoder", bool(getattr(data_cfg, "use_geometry_encoder")))
    if hasattr(data_cfg, "geometry_encoder_type"):
        setattr(data_args, "geometry_encoder_type", getattr(data_cfg, "geometry_encoder_type"))
    if hasattr(data_cfg, "geometry_encoder_path"):
        setattr(data_args, "geometry_encoder_path", getattr(data_cfg, "geometry_encoder_path"))

    return data_args


class QwenSFTtoRLAdapter(Dataset):
    """
    让 VERL 通过 data.custom_cls.path/name 加载。
    预期构造签名（与你们版本一致）：(data_files, config, tokenizer, processor, is_train=True, max_samples=-1)

    - 内部复用 LazySupervisedDataset 的“样本读取与视觉解析”
    - 对外输出 RL 需要的字典：prompt / image / data_source / reward_model / extra_info
    """
    def __init__(
        self,
        data_files: Optional[Union[str, List[str]]],
        config: Any,
        tokenizer: Any,
        processor: Any,
        is_train: bool = True,
        max_samples: int = -1,
    ):
        self.cfg = config
        self.is_train = is_train

        # 组装 DataArguments（把 data.max_* 等也塞进去）
        data_args = _build_data_args_from_cfg(
            cfg=config,
            tokenizer=tokenizer,
            processor=processor,
            is_train=is_train,
            max_samples_arg=max_samples,
        )

        # 构造并复用你们的 SFT 数据集来“读入样本列表”
        inner = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        # 这里拿“原始标注字典”列表，而不是 tokenized batch
        self.rows = inner.list_data_dict
        # 控制样本数（再次兜底）
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]

        # 给 extra_info 用
        self.stage = getattr(inner, "stage", getattr(data_args, "stage", None))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.rows[i]

        # 1) conversations -> messages
        messages = _to_messages(ex.get("conversations", []))

        # 2) 多模态：返回“路径字符串”，让 rollout 的 processor 自己处理
        images = []
        if "image" in ex:
            v = ex["image"]
            images = v if isinstance(v, list) else [v]
        elif "images" in ex:
            v = ex["images"]
            images = v if isinstance(v, list) else [v]
        images = [_abs_join(ex.get("data_path", ""), p) for p in images]

        # 如有图片但 prompt 第一条没带 <image>，自动补在首个 user 末尾
        if images and all(DEFAULT_IMAGE_TOKEN not in m["content"] for m in messages):
            new0 = messages[0]["content"] + " " + (DEFAULT_IMAGE_TOKEN * len(images))
            messages = [{"role": messages[0]["role"], "content": new0}] + messages[1:]

        # 3) ground truth（按你的标注命名选择）
        gt = ex.get("answer", ex.get("label", ex.get("gt", "")))

        item = {
            "data_source": ex.get("dataset_name", "custom"),
            "prompt": messages,
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "id": ex.get("id"),
                "tag": ex.get("tag"),
                "stage": self.stage,
                # 也把长度限制塞进 extra_info，便于奖励或统计时查看
                "limits": {
                    "max_prompt_length":  getattr(getattr(self.cfg, "data", self.cfg), "max_prompt_length", None),
                    "max_response_length": getattr(getattr(self.cfg, "data", self.cfg), "max_response_length", None),
                },
            },
        }
        if images:
            item["image"] = images  # 与 data.image_key 对齐（默认 images；若你设为 image，这里就叫 image）

        return item
