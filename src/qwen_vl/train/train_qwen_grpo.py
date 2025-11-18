# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# ======================
# Standard & Third-Party Imports
# ======================
import os
import re
import sys
import json
import shutil
import pathlib
import logging
from pathlib import Path
from typing import Dict

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLImageProcessor,  # kept for compatibility; not used directly here
    Qwen2VLForConditionalGeneration,  # kept for compatibility; dynamic import used later
    Trainer,
    enable_full_determinism,
    set_seed,
)

# TRL
from trl import GRPOConfig, GRPOTrainer

# ======================
# Project Imports & Path
# ======================
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer         # noqa: F401 (side effects / monkey patch)
import qwen_vl.train.sampler         # noqa: F401
from trainer import replace_qwen2_vl_attention_class
from qwen_vl.data.data_qwen import make_supervised_data_module
from qwen_vl.train.argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    RLArguments,
)
from qwen_vl.data.data_grpo import GRPOPromptImageDataset

# ======================
# Globals
# ======================
local_rank = None

# ======================
# Small Utilities
# ======================

def rank0_print(*args):
    """Print only on rank-0."""
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state = {k: v.cpu() for k, v in state.items()}
        del state
        trainer._save(output_dir, state_dict=cpu_state)  # noqa


def set_model_trainable_parts(model_args: ModelArguments, model: torch.nn.Module):
    """Keep original parameter freezing/unfreezing logic."""
    # vision
    for _, p in model.visual.named_parameters():
        p.requires_grad = bool(model_args.tune_mm_vision)
    # merger(mlp)
    for _, p in model.visual.merger.named_parameters():
        p.requires_grad = bool(model_args.tune_mm_mlp)
    # llm + head
    for _, p in model.model.named_parameters():
        p.requires_grad = bool(model_args.tune_mm_llm)
    model.lm_head.requires_grad = bool(model_args.tune_mm_llm)
    # geometry encoder is always frozen when enabled
    if model_args.use_geometry_encoder and hasattr(model, "geometry_encoder"):
        for _, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False


# ======================
# Reward Utilities (with backward-compat fallback)
# ======================

def _extract_text_from_completion(completion):
    """completion ~ [{'content':'...'}] -> '...'."""
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。、“”‘’：:;,.!?]+", "", s)
    return s


_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _extract_answer_span(s: str) -> str:
    """Heuristics: <answer>...</answer> > 'Final Answer:' > last number > raw."""
    m = re.search(r"<answer>(.*?)</answer>", s, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final\s*answer|answer)\s*[:：]\s*(.+)$", s, flags=re.I | re.M)
    if m:
        return m.group(1).strip()
    nums = _NUM_RE.findall(s)
    if nums:
        return nums[-1]
    return s.strip()


# Try TRL built-ins; fallback to local impl if unavailable
try:
    from trl.rewards import accuracy_reward as _hf_accuracy_reward
    from trl.rewards import think_format_reward as _hf_think_format_reward

    def think_format_reward(completions, **kwargs):
        return _hf_think_format_reward(completions, **kwargs)

    def accuracy_reward(completions, solution=None, **kwargs):
        return _hf_accuracy_reward(completions, solution=solution or kwargs.get("solution"))
except Exception:
    def think_format_reward(completions, **kwargs):
        rewards = []
        for comp in completions:
            text = _extract_text_from_completion(comp)
            rewards.append(1.0 if ("<think>" in text and "</think>" in text) else 0.0)
        return rewards

    def accuracy_reward(completions, solution=None, **kwargs):
        sols = solution or kwargs.get("solution")
        rewards = []
        for i, comp in enumerate(completions):
            pred = _extract_answer_span(_extract_text_from_completion(comp))
            gt = None
            if isinstance(sols, list) and i < len(sols):
                gt = sols[i]
            if gt is None:
                rewards.append(0.0)
                continue
            pred_str, gt_str = str(pred).strip(), str(gt).strip()
            pred_num = _NUM_RE.fullmatch(pred_str)
            gt_num = _NUM_RE.fullmatch(gt_str)
            if pred_num and gt_num:
                rewards.append(1.0 if float(pred_num.group(1)) == float(gt_num.group(1)) else 0.0)
            else:
                rewards.append(1.0 if _normalize_text(pred_str) == _normalize_text(gt_str) else 0.0)
        return rewards


# ======================
# Processing (Tokenizer/Processor) Helpers
# ======================

def build_processor(model_name_or_path: str):
    """
    Build AutoProcessor and ensure left padding + pad/bos/eos ids are present
    (old TRL versions access processing_class.<id> directly).
    """
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        processor.tokenizer = tok

    # Left padding is required by GRPO VLM
    tok.padding_side = "left"

    # Ensure pad/eos/bos ids exist; Qwen often has no BOS -> fallback to EOS
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass
        tok.pad_token_id = tok.eos_token_id
    if getattr(tok, "bos_token_id", None) is None and tok.eos_token_id is not None:
        try:
            tok.bos_token = getattr(tok, "eos_token", None)
        except Exception:
            pass
        tok.bos_token_id = tok.eos_token_id

    # Mirror required fields onto processor itself for older TRL that reads attributes directly
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "pad_token", "eos_token", "bos_token"):
        if getattr(processor, name, None) is None and getattr(tok, name, None) is not None:
            setattr(processor, name, getattr(tok, name))

    # Some TRL versions call processing_class.decode/batch_decode directly
    if not hasattr(processor, "batch_decode"):
        processor.batch_decode = tok.batch_decode
    if not hasattr(processor, "decode"):
        processor.decode = tok.decode

    # Quick assertions (early fail)
    assert callable(processor), "processing_class must be callable"
    assert processor.pad_token_id is not None
    assert processor.eos_token_id is not None
    assert processor.bos_token_id is not None
    return processor


# ======================
# GRPO Config & Argument Bridge
# ======================

def build_grpo_config(training_args: TrainingArguments, rl_args: RLArguments) -> GRPOConfig:
    """Create GRPOConfig from training + RL args (same as before)."""
    return GRPOConfig(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        deepspeed=training_args.deepspeed,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        optim=training_args.optim,

        num_generations=rl_args.rl_num_generations,
        max_prompt_length=rl_args.rl_max_prompt_length,      # VLM suggests None
        max_completion_length=rl_args.rl_max_completion_length,
        beta=rl_args.rl_beta,
        reward_weights=rl_args.rl_reward_weights or [1.0, 1.0, 1.0],  # [alpha, beta, lambda]
        scale_rewards=rl_args.rl_scale_rewards,
        temperature=rl_args.rl_temperature,
        top_p=rl_args.rl_top_p,
        use_vllm=rl_args.rl_use_vllm,
    )


def graft_missing_args(dst: GRPOConfig, src: TrainingArguments):
    """
    Bridge TrainingArguments-only fields expected by your custom optimizer code
    into GRPOConfig to avoid AttributeError (logic unchanged).
    """
    fields = [
        "mm_projector_lr",
        "vision_tower_lr",
        "group_by_modality_length",
        "model_max_length",
        "dataloader_num_workers",
        "weight_decay",
        "lr_scheduler_type",
        "warmup_ratio",
        "warmup_steps",
        "report_to",
        "cache_dir",
    ]
    for k in fields:
        if not hasattr(dst, k) and hasattr(src, k):
            setattr(dst, k, getattr(src, k))
    # explicit defaults to avoid AttributeError in create_optimizer
    if not hasattr(dst, "mm_projector_lr"):
        dst.mm_projector_lr = 0.0
    if not hasattr(dst, "vision_tower_lr"):
        dst.vision_tower_lr = 0.0


# ======================
# Core Train
# ======================

def train(attn_implementation: str = "flash_attention_2"):
    global local_rank

    # ---- Parse args & seed ----
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RLArguments))
    model_args, data_args, training_args, rl_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ---- Build/Load model (same branching & config edits) ----
    if "qwen2.5" in model_args.model_name_or_path.lower():
        if not model_args.use_geometry_encoder:
            from transformers import Qwen2_5_VLForConditionalGeneration
            print("========origin qwen=======")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError("The use_geometry_encoder in config and model_args are not consistent.")
            for k in [
                "use_geometry_encoder",
                "geometry_encoder_type",
                "reference_frame",
                "feature_fusion_method",
                "fusion_num_layers",
                "geometry_merger_type",
                "stage",
            ]:
                setattr(config, k, getattr(model_args, k))
            assert model_args.geometry_encoder_path is not None, "geometry_encoder_path must be set when use_geometry_encoder"
            model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=model_args.geometry_encoder_path,
            )

        data_args.image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor
        data_args.model_type = "qwen2.5vl"

    else:
        from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
            raise ValueError("The use_geometry_encoder in config and model_args are not consistent.")
        for k in [
            "use_geometry_encoder",
            "geometry_encoder_type",
            "reference_frame",
            "feature_fusion_method",
            "fusion_num_layers",
            "geometry_merger_type",
            "stage",
        ]:
            setattr(config, k, getattr(model_args, k))
        assert model_args.geometry_encoder_path is not None, "geometry_encoder_path must be set when use_geometry_encoder"
        model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            geometry_encoder_path=model_args.geometry_encoder_path,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor
        data_args.model_type = "qwen2.5vl"

    # ---- SFT & misc prep (unchanged) ----
    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    set_model_trainable_parts(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    print(model.config)
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    setattr(data_args, "stage", model_args.stage)

    # 保持原逻辑：此处会构造一次 data_module（即便 GRPO 分支中稍后又会构造一次）
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # ---- Branch: GRPO  ----
    # 1) Build processing class for VLM (left padding + padding tokens)
    processor = build_processor(model_args.model_name_or_path)

    # 2) 保持原行为：在 GRPO 分支里再构造一次 dataset（而不是复用上面的 data_module）
    base_dm = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    base_ds = base_dm["train_dataset"]

    train_dataset = GRPOPromptImageDataset(
        base_ds=base_ds,
        processor=processor,
        solution_col=rl_args.rl_solution_column,
    )

    # 3) Rewards = α·acc + β·format + λ·cfadv （保持原实现）
    def acc_reward(completions, **kwargs):
        solutions = kwargs.get("solution")
        if solutions is None:
            return [0.0] * len(completions)
        return accuracy_reward(completions, solution=solutions)

    fmt_reward = think_format_reward

    def make_cfadv_reward(group_size: int):
        def _fn(completions, **kwargs):
            solutions = kwargs.get("solution")
            base = accuracy_reward(completions, solution=solutions) if solutions is not None else [0.0] * len(completions)
            n = len(base)
            if not group_size or group_size <= 1 or n == 0:
                return [0.0] * n
            out = []
            for i in range(0, n, group_size):
                g = base[i:i + group_size]
                s = float(sum(g))
                denom = max(group_size - 1, 1)
                out.extend([ri - (s - ri) / denom for ri in g])
            return out
        return _fn

    cfadv_reward = make_cfadv_reward(group_size=rl_args.rl_num_generations)

    # 4) Build GRPO args & bridge missing custom fields
    grpo_args = build_grpo_config(training_args, rl_args)
    graft_missing_args(grpo_args, training_args)

    # 5) Launch GRPO trainer (processing_class 必须可调用且有必要属性)
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=processor,
        reward_funcs=[acc_reward, fmt_reward, cfadv_reward],
    )

    if list(pathlib.Path(grpo_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume GRPO")
        grpo_trainer.train(resume_from_checkpoint=True)
    else:
        grpo_trainer.train()

    grpo_trainer.save_state()
    processor.save_pretrained(grpo_args.output_dir)

    # Save chat template for inference parity
    src = os.path.join(model_args.model_name_or_path, "chat_template.json")
    dst = os.path.join(grpo_args.output_dir, "chat_template.json")
    if os.path.exists(src):
        shutil.copy2(src, dst)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=grpo_trainer, output_dir=grpo_args.output_dir)



# ======================
# Entry
# ======================
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
