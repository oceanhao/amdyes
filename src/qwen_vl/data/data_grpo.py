# -*- coding: utf-8 -*-
"""
GRPOPromptImageDataset
- 直接复用 LazySupervisedDataset.__getitem__ 的处理产物（保证 forward 依赖的张量齐全），
  然后基于原始对话构造 GRPO 需要的 prompt / solution，并解析 PIL images。
- 仅返回 forward 必需 + GRPO 必需的键，不返回 meta 等无关字段。
"""


import os
from typing import Any, Dict, List, Optional

import torch


class GRPOPromptImageDataset(torch.utils.data.Dataset):
    """复用 base_ds.__getitem__，并补齐 GRPO 需要的 prompt/images/solution。"""

    # forward + GRPO 的“保留白名单”
    KEYS_WHITELIST = {
        # ---- GRPO ----
        "prompt", "images", "solution",
        # ---- forward (按你的 forward 形参) ----
        "input_ids", "attention_mask", "position_ids", "labels",
        "pixel_values", "pixel_values_videos",
        "image_grid_thw", "video_grid_thw",
        "rope_deltas", "cache_position", "second_per_grid_ts",
        "geometry_encoder_inputs", "boxes", "tag",
    }

    def __init__(self, base_ds, processor: Any, solution_col: Optional[str] = None):
        """
        Args:
            base_ds:       LazySupervisedDataset 实例（已实现 __getitem__/list_data_dict/_resolve_images）
            processor:     AutoProcessor（外部已设置 tokenizer 左侧 padding / pad/bos/eos）
            solution_col:  若样本中存在该列则优先作为参考答案；否则退化到原对话最后一条 gpt/assistant
        """
        super().__init__()
        self.base = base_ds
        self.processor = processor
        self.solution_col = solution_col

        # 兼容：若 processor 无 apply_chat_template，则从 tokenizer 透传
        if not hasattr(self.processor, "apply_chat_template"):
            tok = getattr(self.processor, "tokenizer", None)
            if tok is None or not hasattr(tok, "apply_chat_template"):
                raise RuntimeError("processor/tokenizer 缺少 apply_chat_template")
            self.processor.apply_chat_template = tok.apply_chat_template

    # ---------- small helpers ----------
    @staticmethod
    def _norm_role(r: Optional[str]) -> str:
        r = (r or "").lower()
        if r in ("human", "user"):
            return "user"
        if r in ("gpt", "assistant"):
            return "assistant"
        if r == "system":
            return "system"
        return "user"

    @staticmethod
    def _text_of(m: Dict) -> str:
        return (m.get("value") or m.get("content") or "").strip()

    @staticmethod
    def _last_gpt_value(convs: List[Dict]) -> Optional[str]:
        for m in reversed(convs or []):
            if GRPOPromptImageDataset._norm_role(m.get("from") or m.get("role")) == "assistant":
                t = GRPOPromptImageDataset._text_of(m)
                if t:
                    return t
        return None

    def _build_prompt_msgs(
        self, convs: List[Dict], num_images: int
    ) -> List[Dict[str, Any]]:
        """
        用“原始完整对话”构造消息列表形式的 prompt：
        - 去掉最后一条 assistant（作为 solution）；
        - 在首个 user 消息中插入 num_images 个 {"type":"image"}；
        - 只做最小 role 归一，不附加自定义 system。
        """
        # 找到最后一条 assistant 的位置
        last_ass_idx = -1
        for j in range(len(convs) - 1, -1, -1):
            if self._norm_role(convs[j].get("from") or convs[j].get("role")) == "assistant":
                last_ass_idx = j
                break

        msgs: List[Dict[str, Any]] = []
        images_attached = False
        upto = len(convs) - 1 if last_ass_idx >= 0 else len(convs)

        for j in range(upto):
            role = self._norm_role(convs[j].get("from") or convs[j].get("role"))
            txt = self._text_of(convs[j])
            if role == "user":
                # 去掉文本里的 <image>/<video> 占位，真实图片用 typed content 提供
                txt = txt.replace("<image>", "").replace("<video>", "").strip()
                content: List[Dict[str, Any]] = []
                if (not images_attached) and num_images > 0:
                    content.extend({"type": "image"} for _ in range(num_images))
                    images_attached = True
                content.append({"type": "text", "text": txt})
                msgs.append({"role": "user", "content": content})
            else:
                msgs.append({"role": role, "content": txt})

        return msgs

    # ---------- dataset api ----------
    def __len__(self) -> int:
        return len(self.base.list_data_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:


        # A) 直接复用 LazySupervisedDataset.__getitem__（保证 forward 依赖的张量形状/类型完全一致）
        proc: Dict[str, Any] = self.base.__getitem__(idx)

        # B) 解析“原始样本”以获取 PIL images 与对话（不依赖 meta）
        raw_entry: Dict[str, Any] = self.base.list_data_dict[idx]
        try:
            pil_images: List[Any] = list(self.base._resolve_images(raw_entry))  # List[PIL.Image]
        except Exception:
            pil_images = []

        convs: List[Dict[str, Any]] = raw_entry.get("conversations", []) or []
        # 若 proc 中恰好含有 meta.orig_conversations，也可作为回退源（不放回 item）
        if (not convs) and isinstance(proc, dict):
            meta = proc.get("meta", {})
            if isinstance(meta, dict):
                convs = meta.get("orig_conversations", []) or convs

        # C) 构造 prompt（消息列表 → 模板 → 文本），和 solution（最后一条 assistant 文本）
        msgs = self._build_prompt_msgs(convs, num_images=len(pil_images))
        prompt: str = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        solution: Optional[str] = None
        if self.solution_col and (self.solution_col in raw_entry):
            solution = raw_entry[self.solution_col]
        if solution is None:
            solution = raw_entry.get("answer") or raw_entry.get("gt") or self._last_gpt_value(convs)

        # D) 组装返回：仅保留白名单键（不返回 meta）
        item: Dict[str, Any] = {
            "prompt": prompt,
            "images": pil_images,  # 始终是 List[PIL.Image]；即使一张也用列表
        }
        if solution is not None:
            item["solution"] = solution

        # 从 proc 中挑 forward 需要的键；忽略 meta/其它中间字段
        if isinstance(proc, dict):
            for k in self.KEYS_WHITELIST:
                if k in ("prompt", "images", "solution"):
                    continue
                if k in proc and proc[k] is not None:
                    item[k] = proc[k]

        if os.getenv("Debug", "False") == "dataset_grpo":
            from remote_pdb import set_trace
            set_trace()

        return item
