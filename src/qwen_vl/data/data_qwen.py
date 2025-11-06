import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2
from .utils import prepare_image_inputs

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path, max_samples: int=-1):
    with open(path, "r") as f:
        # return [json.loads(line) for line in f]
        ret = []
        for line in f:
            ret.append(json.loads(line))
            if max_samples !=-1 and len(ret) >= max_samples:
                break
    return ret


# CHG: 增加 stage 参数
def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
    vggt_use: bool=False,
    stage: str = None,               # NEW
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    if stage=="stage2-1_rlColdStart":
        system_message = (
            "You are a helpful assistant. Answer the user's question based on the provided images. "
            "Whenever you determine that the question requires spatial or geometric reasoning, you may invoke an external tool that provides additional geometric information to assist in your reasoning and answer generation by outputting <vggt>."
        )

    else:
        system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # NEW: 标记是否进入 RL ColdStart 推理阶段（仅取 user）
    rl_coldstart = (stage == "stage2-1_rlColdStart")  # NEW

    input_ids, targets = [], []
    visual_replicate_index = 0 
    for i, source in enumerate(sources):

        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        # NEW: 在 RL ColdStart 阶段，仅保留“用户”消息
        if rl_coldstart:  # NEW
            filtered = []  # NEW
            for _conv in source:  # NEW
                try:  # NEW
                    role = _conv["role"]  # NEW
                    content = _conv["content"]  # NEW
                except:  # NEW
                    role = _conv["from"]  # NEW
                    content = _conv["value"]  # NEW
                role = roles.get(role, role)  # NEW
                if role == "user":  # 只保留 user  # NEW
                    filtered.append({"role": role, "content": content})  # NEW
            source = filtered  # NEW
            # 若该条样本中没有 user 内容，则直接跳过  # NEW
            if len(source) == 0:  # NEW
                continue  # NEW

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        if vggt_use:
                            replacement = (
                                "<|vision_start|>"
                                + f"<|{visual_type}_pad|>"
                                * grid_thw[visual_replicate_index]
                                + "<|vision_end|>"
                                + "<|vggt_start|>"
                                + f"<|vggt_pad|>"
                                * grid_thw[visual_replicate_index]
                                + "<|vggt_end|>"
                            )
                        else:
                            replacement = (
                                "<|vision_start|>"
                                + f"<|{visual_type}_pad|>"
                                * grid_thw[visual_replicate_index]
                                + "<|vision_end|>"
                            )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                # 训练时 assistant 才会走到这里；RL ColdStart 阶段不会进入
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        # NEW: 在 RL ColdStart 阶段，为生成回答补上 assistant 起始提示（不含内容）
        if stage not in ["cold_start","cold_startv2","qwen"] :#从llava houd、spar等读取的都不需要.但"stage2-1_rlColdStart"需要，因为他在前面去掉了user
            add_prompt_str = "<|im_start|>assistant\n"  # 和你上面 chat_template 的生成提示严格一致
            add_tokens = tokenizer.encode(add_prompt_str, add_special_tokens=False)
            input_id += add_tokens
            target  += [IGNORE_INDEX] * len(add_tokens)

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        if data_args.dataset_use:
            dataset = data_args.dataset_use.split(",")
            dataset_list = data_list(dataset)
            print(f"Loading datasets: {dataset_list}")
            self.oneDatainference_mode=False
        else:
            print("########  inference mode   ###########")
            self.oneDatainference_mode=True
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []
        if data_args.dataset_use:
            for data in dataset_list:
                file_format = data["annotation_path"].split(".")[-1]
                if file_format == "jsonl":
                    annotations = read_jsonl(data["annotation_path"], max_samples=data_args.max_samples)
                else:
                    annotations = json.load(open(data["annotation_path"], "r"))
                sampling_rate = data.get("sampling_rate", 1.0)
                if sampling_rate < 1.0:
                    annotations = random.sample(
                        annotations, int(len(annotations) * sampling_rate)
                    )
                    print(f"sampling {len(annotations)} examples from dataset {data}")
                else:
                    rank0_print(f"dataset name: {data}")
                for ann in annotations:
                    ann["data_path"] = data["data_path"]
                    ann["tag"] = data["tag"]
                    ann['dataset_name']=data['dataset_name']
                list_data_dict += annotations

            print(f"Total training samples: {len(list_data_dict)}")

            random.shuffle(list_data_dict)  # Randomly shuffle the data for training

            print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        #hhh
        self.stage=data_args.stage
        print("==================stage:",self.stage,"========================")
        self.use_vggt_epoch=False

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            length_list.append(image_num * 252 + cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            cur_len += image_num*252
            tag = sample.get("tag", "2d")
            cur_len = -cur_len if tag == "2d" else cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def draw_visual_marks(self, images, spar_info):

        if spar_info is None:
            return
        info = json.loads(spar_info)
        task_type = info["type"]
        from .draw_marker import DRAW_FUNCTIONS
        draw_fn = DRAW_FUNCTIONS[task_type]
        if len(images) == 1:
            draw_fn(images[0], info)
        else:
            draw_fn(images, info)
        # for j, img in enumerate(images):
        #     # write to local
        #     img.save(f"images/img_{j}.jpg", format="JPEG")

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
    # 放在 Dataset 类中
    def _resolve_images(self, source: dict) -> List[Image.Image]:
        """
        返回 PIL.Image 列表，兼容多种来源：
        - source["image"] 为绝对/相对路径字符串
        - source["image"] 为路径列表
        - source["image"] 为 PIL.Image 或其列表
        - source["image_dir"] 或 "frames_dir"：目录下所有图片（排序后采样）
        - base64 data URI: "data:image/jpeg;base64,...."
        """
        imgs: List[Image.Image] = []

        def _open_one(p):
            if isinstance(p, Image.Image):
                return p.convert("RGB")
            if isinstance(p, str) and p.startswith("data:image"):
                head, b64 = p.split("base64,", 1)
                from io import BytesIO
                import base64
                return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            if isinstance(p, str):
                # 绝对路径 or 相对 data_path
                full = p if os.path.isabs(p) else os.path.join(source.get("data_path", ""), p)
                return Image.open(full).convert("RGB")
            raise NotImplementedError(f"Unsupported image spec: {type(p)}")

        if "image" in source:
            v = source["image"]
            if isinstance(v, (list, tuple)):
                for x in v:
                    imgs.append(_open_one(x))
            else:
                imgs.append(_open_one(v))

        # 目录帧
        frames_dir = source.get("image_dir") or source.get("frames_dir")
        if frames_dir:
            full_dir = frames_dir if os.path.isabs(frames_dir) else os.path.join(source.get("data_path", ""), frames_dir)
            if os.path.isdir(full_dir):
                files = [os.path.join(full_dir, f) for f in os.listdir(full_dir)
                        if os.path.isfile(os.path.join(full_dir, f))
                        and f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
                files.sort()
                for f in files:
                    imgs.append(Image.open(f).convert("RGB"))

        if not imgs:
            raise FileNotFoundError("No images resolved from source.")
        return imgs
    def read_video_images(self, source):
        """
        兼容：
        - source["video"] 为绝对/相对路径视频文件
        - source["video"] 指向帧目录
        """
        assert isinstance(source["video"], str), "video should be a string"
        v = source["video"]
        video_path = v if os.path.isabs(v) else os.path.join(source.get("data_path",""), v)

        if os.path.isdir(video_path):
            frame_files = [os.path.join(video_path, f) for f in os.listdir(video_path)
                        if os.path.isfile(os.path.join(video_path, f))]
            frame_files.sort()
            # 与你原来的采样策略一致
            def get_frame_indices(n, fps=1):
                video_length = n / fps
                interval = getattr(self.data_args, "base_interval", 2)
                n_to_sample = round(video_length / interval)
                vmin = getattr(self.data_args, "video_min_frames", 4)
                vmax = getattr(self.data_args, "video_max_frames", 8)
                tgt = min(max(n_to_sample, vmin), vmax)
                idx = np.linspace(0, n-1, tgt, dtype=int)
                return np.unique(idx)
            idx = get_frame_indices(len(frame_files), 1)
            return [Image.open(frame_files[i]).convert("RGB") for i in idx]

        if any(video_path.lower().endswith(ext) for ext in [".mp4",".avi",".mov"]):
            try:
                vr = VideoReader(video_path, num_threads=4)
                total = len(vr)
                avg_fps = vr.get_avg_fps()
                def get_frame_indices(total_frames, fps):
                    video_length = total_frames / max(fps, 1e-6)
                    interval = getattr(self.data_args, "base_interval", 2)
                    n_to_sample = round(video_length / interval)
                    vmin = getattr(self.data_args, "video_min_frames", 4)
                    vmax = getattr(self.data_args, "video_max_frames", 8)
                    tgt = min(max(n_to_sample, vmin), vmax)
                    return np.unique(np.linspace(0, total_frames-1, tgt, dtype=int))
                idx = get_frame_indices(total, avg_fps)
                frames = vr.get_batch(idx).asnumpy()
                return [Image.fromarray(fr).convert("RGB") for fr in frames]
            except Exception:
                # decord失败降级到OpenCV
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    ok, frame_bgr = cap.read()
                    while ok:
                        frames.append(Image.fromarray(frame_bgr[:, :, ::-1]).convert("RGB"))
                        ok, frame_bgr = cap.read()
                    cap.release()
                    if not frames:
                        raise RuntimeError("No frames decoded by OpenCV.")
                    # 简单等间隔采样
                    vmin = getattr(self.data_args, "video_min_frames", 4)
                    vmax = getattr(self.data_args, "video_max_frames", 8)
                    tgt = min(max(len(frames), vmin), vmax)
                    idx = np.unique(np.linspace(0, len(frames)-1, tgt, dtype=int))
                    return [frames[i] for i in idx]
                except Exception as e:
                    raise FileNotFoundError(f"Cannot read video: {video_path}, err={e}")
        raise FileNotFoundError(f"Invalid video path: {video_path}")


    def _get_item(self, i) -> Dict[str, torch.Tensor]:

        if self.stage == "force_use":
            self.use_vggt_epoch = True

        elif self.stage =="force_notuse":
            self.use_vggt_epoch = False
        elif self.stage == "cold_startv2":
            dataset_name = self.list_data_dict[i]['dataset_name']
            if "spar" in dataset_name:
                self.use_vggt_epoch = bool(random.getrandbits(1))
            elif "llava_hound" in dataset_name:
                self.use_vggt_epoch = False
        elif self.stage =="stage2-1_rlColdStart":
            self.use_vggt_epoch = False
        else:
            if self.stage == "cold_start" or self.stage == "force_half":
                self.use_vggt_epoch = bool(random.getrandbits(1))
        # NEW: 先对原始样本做一次完整快照，用于 meta（避免后续改写影响）
        orig = copy.deepcopy(self.list_data_dict[i])  # NEW
        if os.getenv("Debug", "False")=="debug_dataset":
            from remote_pdb import set_trace
            set_trace() # you'll see the port number in the logs

        # NEW: 提取首条 human/gpt（如需最后一条可自行调整）
        orig_human_first, orig_gpt_first = None, None  # NEW
        for msg in orig.get("conversations", []):  # NEW
            if msg.get("from") == "human" and orig_human_first is None:  # NEW
                orig_human_first = msg.get("value")  # NEW
            if msg.get("from") == "gpt" and orig_gpt_first is None:  # NEW
                orig_gpt_first = msg.get("value")  # NEW

        meta = {  # NEW
            "id": orig.get("id"),
            "data_source": orig.get("data_source"),
            "video": orig.get("video"),          # 注意：此处保留原始 video，不受后续改写影响
            "data_path": orig.get("data_path"),
            "tag": orig.get("tag"),
            "orig_conversations": copy.deepcopy(orig.get("conversations", [])),
            "orig_human_first": orig_human_first,
            "orig_gpt_first": orig_gpt_first,
        }


        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None
        
        if "video" in sources[0]:
            sources[0]["images"] = self.read_video_images(sources[0])
            num_image = len(sources[0]["images"])
            sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
                DEFAULT_VIDEO_TOKEN, "".join([DEFAULT_IMAGE_TOKEN] * num_image)
            )
            del sources[0]["video"]
        
        # # replace <image>\n with <image>
        sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
            f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN
        )

        # rename images tag
        if "images" in sources[0]:
            sources[0]["image"] = sources[0]["images"]

        # notice that we use images as the tag
        # ---- 替换 _get_item 里 'if "image" in sources[0]:' 的主体 ----
        if "image" in sources[0] or "image_dir" in sources[0] or "frames_dir" in sources[0]:
            # 统一解析为 PIL.Image 列表
            images = self._resolve_images(sources[0])

            # 画可视化标记（如有）
            self.draw_visual_marks(images, sources[0].get("spar_info", None))

            image, grid_thw, geometry_encoder_inputs = [], [], []
            for img in images:
                ret = prepare_image_inputs(img, self.data_args.image_processor)
                image.append(ret["pixel_values"])
                geometry_encoder_inputs.append(ret["geometry_encoder_inputs"])
                grid_thw.append(ret["image_grid_thw"])

            grid_thw_merged = [g.prod() // self.data_args.image_processor.merge_size**2 for g in copy.deepcopy(grid_thw)]
            sources_conv = copy.deepcopy([e["conversations"] for e in [sources[0]]])
            data_dict = preprocess_qwen_2_visual(
                sources_conv, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image",
                vggt_use=self.use_vggt_epoch, stage=self.stage
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video",vggt_use=self.use_vggt_epoch
                , stage=self.stage
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged,vggt_use=self.use_vggt_epoch
                , stage=self.stage
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
            if getattr(self.data_args, "use_geometry_encoder", False):
                data_dict["geometry_encoder_inputs"] = geometry_encoder_inputs
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        if self.stage=="stage2-1_rlColdStart":
            data_dict["meta"] = meta  # NEW
        data_dict["tag"] = self.list_data_dict[i].get("tag", "2d")

        if os.getenv("Debug", "False")=="debug_dataset":
            from remote_pdb import set_trace
            set_trace() # you'll see the port number in the logs

        return data_dict
    # new: 复用现有 _get_item 的所有逻辑来处理“单条原始样本 dict”
    def build_from_entry(self, entry: dict) -> dict[str, torch.tensor]:
        """
        给我一条原始样本（与 self.list_data_dict[i] 同结构），
        我临时把它放到 list_data_dict=[entry]，调用现有 _get_item(0)，
        返回与 __getitem__ 完全一致的一条 processed sample。
        """
        _backup = self.list_data_dict
        try:
            self.list_data_dict = [entry]
            return self._get_item(0)
        finally:
            self.list_data_dict = _backup


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
                
        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            geometry_encoder_inputs = [torch.stack(instance["geometry_encoder_inputs"]) for instance in instances]
            batch["geometry_encoder_inputs"] = geometry_encoder_inputs
            assert len(set([instance["tag"] for instance in instances])) == 1, "all data in a batch should have the same tag"
            batch["tag"] = instances[0]["tag"]
        if "meta" in instances[0]:
            batch["meta"] = [inst.get("meta") for inst in instances]  # NEW   
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor(
            [0] + [len(seq) for seq in input_ids], dtype=torch.int32
        )
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

                
        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            raise NotImplementedError("FlattenedDataCollatorForSupervisedDataset does not support geometry_encoder_inputs")
        if self.stage=="stage2-1_rlColdStart":
            batch["meta"] = [inst.get("meta") for inst in instances]  # NEW
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
