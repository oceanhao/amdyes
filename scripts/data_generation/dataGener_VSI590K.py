import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler

from tqdm.auto import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, AutoConfig, set_seed, enable_full_determinism  # CHG
from torch.utils.data import DataLoader, Subset  # NEW
import re  # 放在文件顶部；若不方便，也可把它写到函数内部


# CHG: 允许任意参数，兼容 get_rank(group) 之类的调用
def _safe_get_rank(*args, **kwargs):  # CHG
    try:  # CHG
        import torch, os  # CHG
        if torch.distributed.is_available() and torch.distributed.is_initialized():  # CHG
            return torch.distributed._orig_get_rank(*args, **kwargs)  # CHG
    except Exception:  # CHG
        pass  # CHG
    import os  # CHG
    return int(os.environ.get("RANK", "0"))  # CHG

# NEW: 同样给 get_world_size 做一个安全补丁（有些路径会调用 get_world_size(group)）
def _safe_get_world_size(*args, **kwargs):  # NEW
    try:  # NEW
        import torch, os  # NEW
        if torch.distributed.is_available() and torch.distributed.is_initialized():  # NEW
            return torch.distributed._orig_get_world_size(*args, **kwargs)  # NEW
    except Exception:  # NEW
        pass  # NEW
    import os  # NEW
    return int(os.environ.get("WORLD_SIZE", "1"))  # NEW

# CHG: 打补丁时，保存原函数并覆盖为“安全版本”，注意也补丁 world_size
try:  # CHG
    import torch  # CHG
    if hasattr(torch.distributed, "get_rank") and not hasattr(torch.distributed, "_orig_get_rank"):  # CHG
        torch.distributed._orig_get_rank = torch.distributed.get_rank  # CHG
        torch.distributed.get_rank = _safe_get_rank  # CHG
    if hasattr(torch.distributed, "get_world_size") and not hasattr(torch.distributed, "_orig_get_world_size"):  # NEW
        torch.distributed._orig_get_world_size = torch.distributed.get_world_size  # NEW
        torch.distributed.get_world_size = _safe_get_world_size  # NEW
except Exception:  # CHG
    pass  # CHG


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# NEW
def maybe_init_distributed(training_args):  # NEW
    """根据 torchrun 环境变量初始化分布式；设置 CUDA 设备。"""  # NEW
    import os, torch  # NEW
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(training_args, "local_rank", 0) or 0))  # NEW
    if torch.cuda.is_available():  # NEW
        torch.cuda.set_device(local_rank)  # NEW
    if (not torch.distributed.is_initialized()) and int(os.environ.get("WORLD_SIZE", "1")) > 1:  # NEW
        torch.distributed.init_process_group(backend="nccl", timeout=torch.timedelta(hours=12))  # NEW
    rank = int(os.environ.get("RANK", "0"))  # NEW
    world_size = int(os.environ.get("WORLD_SIZE", "1"))  # NEW
    return local_rank, rank, world_size  # NEW

# NEW
def split_indices(num_items: int, world_size: int, rank: int, seed: int):  # NEW
    """将 [0..num_items-1] 随机打乱后，按 world_size 等分；返回当前 rank 的索引列表。"""  # NEW
    import torch  # NEW
    g = torch.Generator()  # NEW
    g.manual_seed(seed if seed is not None else 42)  # NEW
    perm = torch.randperm(num_items, generator=g).tolist()  # NEW
    chunks = [[] for _ in range(world_size)]  # NEW
    for i, idx in enumerate(perm):  # NEW
        chunks[i % world_size].append(idx)  # NEW
    return chunks[rank]  # NEW

# NEW
def move_to_device(batch, device):  # NEW
    """递归将张量移到 device；保持非张量原样（如图像列表）。"""  # NEW

    if isinstance(batch, dict):  # NEW
        return {k: move_to_device(v, device) for k, v in batch.items()}  # NEW
    elif isinstance(batch, (list, tuple)):  # NEW
        return type(batch)(move_to_device(v, device) for v in batch)  # NEW
    elif isinstance(batch, torch.Tensor):  # NEW
        return batch.to(device, non_blocking=True)  # NEW
    else:  # NEW
        return batch  # NEW


# NEW: 自定义推理类，构造接口严格对齐 Trainer(model=..., processing_class=..., args=..., **data_module)
class Inferencer:
    def __init__(self,
                 model,
                 processing_class=None,
                 args=None,
                 local_rank=None,  # NEW
                 rank=None,        # NEW
                 world_size=None,  # NEW
                 fixed_text_a=None,   # NEW
                 fixed_text_b=None,   # NEW
                 processor=None,
                 **data_module):
        self.model = model
        self.tokenizer = processing_class
        self.args = args
        self.fixed_text_a = fixed_text_a
        self.fixed_text_b = fixed_text_b
        # ---- 与 Trainer 一致：保存数据资源 ----
        self.train_dataset = data_module.get("train_dataset", None)
        self.eval_dataset = data_module.get("eval_dataset", None)
        self.data_collator = data_module.get("data_collator", None) or data_module.get("collate_fn", None)

        if self.eval_dataset is not None:
            self.dataset = self.eval_dataset
        elif self.train_dataset is not None:
            self.dataset = self.train_dataset
        else:
            raise ValueError("No dataset found for inference (need eval_dataset or train_dataset).")

        # CHG: 若外部已给 rank 信息则直接用，否则内部初始化
        if (local_rank is None) or (rank is None) or (world_size is None):  # CHG
            self.local_rank, self.rank, self.world_size = maybe_init_distributed(self.args)  # CHG
        else:
            self.local_rank, self.rank, self.world_size = local_rank, rank, world_size  # CHG
        self.processor=processor
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True

        # ---- 生成参数，模拟 Trainer 的 generation 配置读取 ----
        self.gen_kwargs = {}
        max_new = (getattr(self.args, "generation_max_new_tokens", None)
                   or getattr(self.args, "generation_max_length", None)
                   or 512)
        self.gen_kwargs["max_new_tokens"] = int(max_new)
        for name in ("do_sample", "temperature", "top_p", "num_beams", "repetition_penalty", "length_penalty"):
            val = getattr(self.args, name, None)
            if val is not None:
                self.gen_kwargs[name] = type(val)(val)
        # pad/eos（若下游要用到）
        if getattr(self.args, "eos_token_id", None) is not None:
            self.gen_kwargs["eos_token_id"] = int(self.args.eos_token_id)
        if getattr(self.args, "pad_token_id", None) is not None:
            self.gen_kwargs["pad_token_id"] = int(self.args.pad_token_id)

        # ---- 随机等分样本 ----
        idxs = split_indices(len(self.dataset), self.world_size, self.rank, seed=self.args.seed)
        self.subset = Subset(self.dataset, idxs)

        # ---- DataLoader（与 Trainer 同源信息）----
        per_bs = 1
        self.loader = DataLoader(
            self.subset,
            batch_size=per_bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

        # ---- 输出文件 ----
        self.out_dir = Path(self.args.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.out_dir / f"samples_rank{self.rank}.jsonl"  # CHG: 原本是 predictions_rank*.jsonl

        # ---- 允许的 forward/generate 输入键（与你的 forward 参数名对齐）----
        # 注：generate 并不吃所有 forward 参数，但我们会尽量原样透传；PyTorch/HF 会忽略未使用的键。
        self._allowed_input_keys = {
            "input_ids", "attention_mask", "position_ids",
            "past_key_values", "inputs_embeds",
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
            "rope_deltas", "cache_position", "second_per_grid_ts",
            "geometry_encoder_inputs", "boxes", "tag",
        }
        self.debugFlag = os.getenv("Debug", "False")
    # 与 Trainer._prepare_inputs 类似：把 batch 递归搬到设备，并可筛选键
    def _prepare_inputs(self, batch):
        # 1) 递归搬到 device
        batch = move_to_device(batch, self.device)
        # 2) 只保留模型可能需要的键（避免把无关对象传入）
        pruned = {k: v for k, v in batch.items() if k in self._allowed_input_keys}
        # 有些数据集会给 labels；推理我们不需要，但保留以防后续你想算 loss
        if "labels" in batch:
            pruned["labels"] = batch["labels"]
        return pruned
    def run(self):
        total_samples = len(self.subset)  # 以样本数为总量，ETA 更准确
        show_all = os.getenv("PROGRESS_ALL_RANKS", "0") == "1"
        disable_bar = (not show_all) and (self.rank not in (0, None))

        pbar = tqdm(
            total=total_samples,
            dynamic_ncols=True,
            unit="sample",
            desc=f"Infer r{self.rank}",
            disable=disable_bar,
            mininterval=0.5,   # 刷新间隔，避免过度刷新
            smoothing=0.3,     # 估计 ETA 的平滑
            leave=True
        )

        with self.tmp_path.open("w", encoding="utf-8", buffering=1) as fout, torch.no_grad():
            for raw_batch in self.loader:
                inputs = self._prepare_inputs(raw_batch)
                if os.getenv("Debug", "False") == "debug_datage":
                    from remote_pdb import set_trace; set_trace()

                # 生成
                cont = self.model.generate(**inputs, **self.gen_kwargs)

                # 去掉 prompt（若存在）
                trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], cont)] if "input_ids" in inputs else cont

                if os.getenv("Debug", "False") == "debug_datage":
                    from remote_pdb import set_trace; set_trace()

                answers = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                # 逐条写入（及时 flush）
                metas = raw_batch.get("meta")
                fsync_flag = bool(getattr(self.args, "fsync_every_write", False))

                if isinstance(metas, (list, tuple)):
                    for m, a in zip(metas, answers):
                        obj = dict(m or {})
                        obj["model_generate_answer"] = a
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        fout.flush()
                        if fsync_flag: os.fsync(fout.fileno())
                    n_written = len(answers)
                else:
                    obj = dict(metas or {})
                    obj["model_generate_answer"] = answers[0] if len(answers) == 1 else answers
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    fout.flush()
                    if fsync_flag: os.fsync(fout.fileno())
                    n_written = 1

                # 更新进度条：按写出的样本数更新，tqdm 会自动显示 ETA
                pbar.update(n_written)
                # 可选：附加吞吐显示
                rate = pbar.format_dict.get("rate")
                if rate:
                    pbar.set_postfix_str(f"{rate:.2f} samp/s")

        pbar.close()

        # 分布式同步与合并（保持不变）
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.rank == 0:
            merged = self.out_dir / "samples_all.jsonl"
            with merged.open("w", encoding="utf-8") as fm:
                fm.write("[\n")
                first = True
                for r in range(self.world_size):
                    p = self.out_dir / f"samples_rank{r}.jsonl"
                    if p.exists():
                        with p.open("r", encoding="utf-8") as fr:
                            for line in fr:
                                line = line.rstrip()
                                if not line:
                                    continue
                                fm.write((("" if first else ",\n") + line))
                                first = False
                fm.write("\n]\n")
            print(f"[OK] 合并完成 -> {merged}")




def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    # CHG: 使用统一的分布式初始化，设置 local_rank/rank/world_size 与 CUDA 设备
    local_rank, rank, world_size = maybe_init_distributed(training_args)  # CHG
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "zd11024" in model_args.model_name_or_path.lower():
        from qwen_vl.model.modeling_qwen2_5_vl_official import Qwen2_5_VLForConditionalGenerationWithVGGT
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
            raise ValueError(
                "The use_geometry_encoder in config and model_args are not consistent. "
                "Please check the model config."
            )

        for k in [
            "use_geometry_encoder", 
            "geometry_encoder_type", 
            "reference_frame",
            "feature_fusion_method", 
            "fusion_num_layers",
            "geometry_merger_type",
        ]:
            setattr(config, k, getattr(model_args, k))
    else:
        from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
            raise ValueError(
                "The use_geometry_encoder in config and model_args are not consistent. "
                "Please check the model config."
            )

        for k in [
            "use_geometry_encoder", 
            "geometry_encoder_type", 
            "reference_frame",
            "feature_fusion_method", 
            "fusion_num_layers",
            "geometry_merger_type",
            "stage"
        ]:
            setattr(config, k, getattr(model_args, k))
                
    model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        geometry_encoder_path=model_args.geometry_encoder_path
    )

    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen2.5vl"

    # CHG: 推理开启缓存
    model.config.use_cache = True  # CHG

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if torch.distributed.get_rank() == 0:
        # CHG: 推理无需打印可训练参数，但保留兼容
        try:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
        except Exception:
            pass

    print(model.config)
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)    
    setattr(data_args, "stage", model_args.stage)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    processor=AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            padding_side="left"
        )
    # CHG: 严格参考 Trainer 的调用方式 + 关键字参数，避免位置参数错位
    inferencer = Inferencer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        local_rank=local_rank,   # 可选：你已在外面计算好了，就传进来
        rank=rank,
        world_size=world_size,
        processor=processor,
        fixed_text_a=" To improve my reasoning, I need to use an external tool that provides additional geometric information to assist my reasoning and generate a more accurate answer. <vggt>", 
        fixed_text_b=" After invoking the above tool, additional geometric information has been obtained.",
        **data_module            # 关键：像 Trainer 一样用 **data_module
    )

    inferencer.run()  # NEW


if __name__ == "__main__":
    with torch.no_grad():
        train(attn_implementation="flash_attention_2")
