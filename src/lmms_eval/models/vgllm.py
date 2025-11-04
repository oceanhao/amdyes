import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import copy
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
from qwen_vl.data.utils import load_and_preprocess_images

try:
    # from qwen_vl_utils import process_vision_info
    from qwen_vl_utils import extract_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")
# NEW: 直接使用现成的 DataArguments 和数据模块构造函数
from qwen_vl.train.argument import DataArguments  # NEW
from qwen_vl.data.data_qwen import make_supervised_data_module  # NEW
import types, copy  # NEW
import os


@register_model("vgllm")
class VGLLM(lmms):

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        max_length: Optional[int] = None,
        add_frame_index: bool=False,
        stage: Optional[str] = "inference",  
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)

        if getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False):
            load_class = Qwen2_5_VLForConditionalGenerationWithVGGT
            
            eval_logger.info("Using Qwen2_5_VLForConditionalGenerationWithVGGT")
        else:
            load_class = Qwen2_5_VLForConditionalGeneration
            eval_logger.info("Using Qwen2_5_VLForConditionalGeneration")
        if use_flash_attention_2:
            self._model = load_class.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = load_class.from_pretrained(pretrained, config=config, torch_dtype="auto", device_map=self.device_map).eval()

        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        # NEW: 用 DataArguments 作为默认 data_args，并补齐 dataset 真实所需字段
        self._data_args = DataArguments()  # NEW: 直接实例化默认
        # 覆盖/补齐与当前模型/处理器一致的字段
        self._data_args.use_geometry_encoder=True
        self._data_args.video_max_frames = max_num_frames  # NEW
        self._data_args.image_processor=self.processor.image_processor

        # NEW: DataArguments 原定义里没有 stage / use_geometry_encoder / image_processor，这里按需补充
        setattr(self._data_args, "stage", stage)  # NEW
        


        data_module = make_supervised_data_module(tokenizer=self._tokenizer, data_args=self._data_args)
        self.train_dataset = data_module["train_dataset"]
        self.data_collator = data_module["data_collator"]

        self._allowed_input_keys = {
            "input_ids", "attention_mask", 
            "past_key_values", "inputs_embeds",
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
            "rope_deltas", "cache_position", "second_per_grid_ts",
            "geometry_encoder_inputs", "boxes",
        }

        # import os
        # if os.getenv("Debug", "False")=="True":
        #     from remote_pdb import RemotePdb
        #     RemotePdb('127.0.0.1', 18457).set_trace()

        

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    def _target_device(self):
        # 与你“原始正常的代码”保持一致的判定
        return "cuda" if self.device_map == "auto" else self._device


    def _to_device_tree(self, obj, device):
        # 递归把 dict / list / tuple / Tensor 全搬到同一 device
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device_tree(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: self._to_device_tree(v, device) for k, v in obj.items()}
        return obj

    def _batch_from_entries(self, entries: list[dict]) -> dict:
        """
        用现有 dataset 的 build_from_entry 处理“原始样本字典”，再交给 data_collator 打包。
        这一步完全复用 _get_item 的逻辑（不会重复写预处理）。
        """
        assert self.train_dataset is not None and self.data_collator is not None, \
            "train_dataset / data_collator 还未初始化；请先用 make_supervised_data_module(...) 注入。"

        samples = [self.train_dataset.build_from_entry(e) for e in entries]  # CPU 上的单条样本
        batch = self.data_collator(samples)  # 仍在 CPU
        return batch


    def _generate_batch(self, batch: dict, gen_kwargs: dict | None = None) -> list[str]:
        """
        单点抽象：把 collator 打出来的 batch 统一搬到正确设备，调用 self.model.generate，
        然后 decode 成字符串。
        """
        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 4096)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        model_inputs = {k: v for k, v in batch.items() if k in self._allowed_input_keys and v is not None}

        # 统一搬到同一逻辑设备
        device = self._target_device()
        model_inputs = self._to_device_tree(model_inputs, device)

        pad_token_id = self.tokenizer.pad_token_id
        import os
        if os.getenv("Debug", "False")=="eval":
            from remote_pdb import RemotePdb
            RemotePdb('127.0.0.1', 18457).set_trace()
    # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'position_ids', 'geometry_encoder_inputs'])
        outputs = self.model.generate(
            **model_inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=bool(gen_kwargs["temperature"] > 0),
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
        )

        # 用真实输入长度裁剪
        in_lens = []
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            am = batch["attention_mask"]
            if am.dim() == 2:
                # 注意：collator 在 CPU 上，没关系；我们只读取 mask 统计长度
                in_lens = am.sum(dim=1).tolist()
        if not in_lens:
            pad_id = self.tokenizer.pad_token_id
            for row in batch["input_ids"]:
                in_lens.append(int((row != pad_id).sum().item()) if pad_id is not None else int(row.numel()))

        trimmed = [out_ids[L:] for L, out_ids in zip(in_lens, outputs)]
        answers = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # os.makedirs("/remote-home/haohh/_cvpr2025/VG-LLM/tmp", exist_ok=True)
        # with open(f"/remote-home/haohh/_cvpr2025/VG-LLM/tmp/vgllm_dbg_r{getattr(self,'rank',0)}.log", "a", encoding="utf-8") as f:
        #     f.write(str(answers))
        #     f.flush()
        print(answers)
        return answers

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # 这两行一定要有（上次 NameError 就是这里漏了）
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]; split = split[0]

            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            import os
            if os.getenv("Debug", "False")=="True":
                from remote_pdb import RemotePdb
                RemotePdb('127.0.0.1', 18457).set_trace()

            # 1) 构造 entries（与原逻辑一致；不做任何图像解码，这一步只“挂路径/对象”）
            entries = []
            for i, ctx in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                entry = {
                    "id": str(doc_id[i]),
                    "conversations": [],
                    "data_source": "lmms_eval",
                    "data_path": "",   # 若后续需要拼路径，可在 _get_item 里兼容
                    "tag": "2d",
                }
                if visual is None:
                    human_val = ctx
                elif isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov")):
                    entry["video"] = visual
                    human_val = "<video>\n" + ctx
                elif isinstance(visual, Image.Image):
                    entry["image"] = [visual]
                    human_val = "<image>\n" + ctx
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                    entry["image"] = list(visual)
                    human_val = ("<image>" * len(visual)) + "\n" + ctx
                else:
                    human_val = ctx

                entry["conversations"].append({"from": "human", "value": human_val})
                entries.append(entry)

            # 2) ——关键：复用 dataset + collator + _get_item——
            batch = self._batch_from_entries(entries)
            import os
            if os.getenv("Debug", "False")=="True":
                from remote_pdb import RemotePdb
                RemotePdb('127.0.0.1', 18457).set_trace()
            # 3) 统一走一次 _generate_batch
            gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}
            answers = self._generate_batch(batch, gen_kwargs)

            for ans, ctx in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res



    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")