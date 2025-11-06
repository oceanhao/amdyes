# qwen2_5_vl_vgllm_style.py
import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

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

# —— 不再使用 qwen_vl_utils / process_vision_info —— #

# 若启用了几何/外部特征，加载带 VGGT 的模型类（与 vgllm 一致）
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

# 复用数据模块（与 vgllm 一致）
from qwen_vl.train.argument import DataArguments          # NEW
from qwen_vl.data.data_qwen import make_supervised_data_module  # NEW
# 如需在本类内做图像预处理，可引入（默认仍交给 dataset/collator 处理）
from qwen_vl.data.utils import load_and_preprocess_images       # 可选：这里未直接使用

@register_model("qwen2_5_vl")  # 与原注册名一致：替换为“按 vgllm 风格”的实现
class Qwen2_5_VL(lmms):
    """
    Qwen2.5-VL 按 vgllm 的方式构造 batch 与推理的实现。
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,  # 保留形参占位，与老接口兼容
        fps: Optional[float] = None,                      # 同上
        max_image_size: Optional[int] = None,             # 同上
        max_length: Optional[int] = None,                 # 允许外部限制 tokenizer 的 max_length
        add_frame_index: bool = False,                    # 与 vgllm 参数集对齐
        stage: Optional[str] = "inference",               # 透传给 DataArguments
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # —— 设备 & 加速器 —— #
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

        # —— 选择模型类（是否带 VGGT） —— #
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
            self._model = load_class.from_pretrained(
                pretrained, config=config, torch_dtype="auto", device_map=self.device_map
            ).eval()

        # —— 处理器 / 分词器 —— #
        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.max_num_frames = max_num_frames
        self.add_frame_index = add_frame_index

        # —— 多卡封装 —— #
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], \
                "Only DDP and FSDP are supported."
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

        # —— 关键：复用 data module（与 vgllm 完全一致） —— #
        self._data_args = DataArguments()
        # 与当前模型/处理器对齐的字段
        setattr(self._data_args, "stage", stage)
        # 注意：DataArguments 原定义如无下列字段，请在你项目里对应补齐
        setattr(self._data_args, "video_max_frames", max_num_frames)
        setattr(self._data_args, "image_processor", self.processor.image_processor)
        # 若你的 pipeline 里需要几何编码开关，可从 config 透传：
        setattr(self._data_args, "use_geometry_encoder",
                bool(getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False)))

        data_module = make_supervised_data_module(tokenizer=self._tokenizer, data_args=self._data_args)
        self.train_dataset = data_module["train_dataset"]
        self.data_collator = data_module["data_collator"]

        # 仅保留模型真正会吃的键（与 vgllm 一致）
        self._allowed_input_keys = {
            "input_ids", "attention_mask",
            "past_key_values", "inputs_embeds",
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
            "rope_deltas", "cache_position", "second_per_grid_ts",
            "geometry_encoder_inputs", "boxes",
        }

    # --------- 性能/工具函数，与 vgllm 完全一致 --------- #
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

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
        return [j for i in input for j in i]

    def _target_device(self):
        return "cuda" if self.device_map == "auto" else self._device

    def _to_device_tree(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device_tree(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: self._to_device_tree(v, device) for k, v in obj.items()}
        return obj

    def _batch_from_entries(self, entries: list[dict]) -> dict:
        """
        与 vgllm 同步：把原始 entry 交给 dataset.build_from_entry，再由 data_collator 打包成 batch。
        """
        assert self.train_dataset is not None and self.data_collator is not None, \
            "train_dataset / data_collator 未初始化；请检查 make_supervised_data_module(...)。"
        samples = [self.train_dataset.build_from_entry(e) for e in entries]  # CPU
        batch = self.data_collator(samples)  # 仍在 CPU
        return batch

    def _generate_batch(self, batch: dict, gen_kwargs: dict | None = None) -> list[str]:
        """
        统一的推理入口：过滤允许键 → 搬到设备 → model.generate → 裁剪解码。
        """
        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 4096)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        model_inputs = {k: v for k, v in batch.items() if k in self._allowed_input_keys and v is not None}

        device = self._target_device()
        model_inputs = self._to_device_tree(model_inputs, device)

        pad_token_id = self.tokenizer.pad_token_id

        outputs = self.model.generate(
            **model_inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=bool(gen_kwargs["temperature"] > 0),
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=True,
        )

        # 用 attention_mask 或 非 pad 计数 计算输入长度
        in_lens = []
        if "attention_mask" in batch and batch["attention_mask"] is not None and batch["attention_mask"].dim() == 2:
            in_lens = batch["attention_mask"].sum(dim=1).tolist()
        if not in_lens:
            pad_id = self.tokenizer.pad_token_id
            for row in batch["input_ids"]:
                in_lens.append(int((row != pad_id).sum().item()) if pad_id is not None else int(row.numel()))

        trimmed = [out_ids[L:] for L, out_ids in zip(in_lens, outputs)]
        answers = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return answers

    # --------- 推理接口（与 vgllm 的 generate_until 对齐） --------- #
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        严格 vgllm 风格：把 (context, visual) 先转成 entries（只挂路径/对象与 <image>/<video> 标记），
        再交给 dataset.build_from_entry + data_collator 构造 batch，最后统一 _generate_batch。
        """
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]; split = split[0]

            # 与 vgllm 保持一致的视觉解析：不解码，只拿“可引用对象/路径”
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            # 1) 组装 entries（只构造 human 一轮；多轮对话见下方说明）
            entries = []
            for i, ctx in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                entry = {
                    "id": str(doc_id[i]),
                    "conversations": [],
                    "data_source": "lmms_eval",
                    "data_path": "",
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
                    # 其他自定义视觉对象：直接作为纯文本
                    human_val = ctx

                entry["conversations"].append({"from": "human", "value": human_val})
                entries.append(entry)

            # 2) dataset + collator 打包
            batch = self._batch_from_entries(entries)

            # 3) 统一推理
            gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}
            answers = self._generate_batch(batch, gen_kwargs)

            for ans, ctx in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    # 若要做“多轮对话推理”，建议直接把多轮对话按 entries["conversations"] 填满；
    # 这里与 vgllm 保持一致，函数留空，避免与外部评测框架假定的输入协议冲突。
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Use entries['conversations'] with multiple human/assistant turns via _batch_from_entries.")
