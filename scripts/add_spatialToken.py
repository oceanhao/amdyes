from transformers import AutoProcessor, AutoModelForImageTextToText, AddedToken
import torch, os, shutil


src_model_dir = "/remote-home/share/_hf_models/hfmodel/Qwen/Qwen2.5-VL-3B-Instruct"                      # 你的原模型目录（或模型ID）
dst_save_dir  = "/remote-home/haohh/_cvpr2025/VG-LLM/ckpt_saves/qwen2.5-with-vggt-special"    # 想把更新后的分词器/模型保存到这里
os.makedirs(dst_save_dir, exist_ok=True)

processor = AutoProcessor.from_pretrained(src_model_dir, trust_remote_code=True)
tokenizer = processor.tokenizer
model = AutoModelForImageTextToText.from_pretrained(src_model_dir, trust_remote_code=True, torch_dtype="auto")

new_tokens = [
    AddedToken("<|vggt_start|>", lstrip=False, rstrip=False, special=True, normalized=False),
    AddedToken("<|vggt_end|>",   lstrip=False, rstrip=False, special=True, normalized=False),
    AddedToken("<|vggt_pad|>",   lstrip=False, rstrip=False, special=True, normalized=False),
]
num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))

vggt_start_id = tokenizer.convert_tokens_to_ids("<|vggt_start|>")
vggt_end_id   = tokenizer.convert_tokens_to_ids("<|vggt_end|>")
print("vggt_start_id =", vggt_start_id, "vggt_end_id =", vggt_end_id)

with torch.no_grad():
    emb = model.get_input_embeddings().weight
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_start_id is not None and im_end_id is not None:
        ref = 0.5 * (emb[im_start_id] + emb[im_end_id])
        emb[vggt_start_id].copy_(ref)
        emb[vggt_end_id].copy_(ref)

# 保存（Processor 会把 tokenizer 一起存）
processor.tokenizer.save_pretrained(dst_save_dir)
model.save_pretrained(dst_save_dir)

# 覆盖原目录（可选）
def replace_tokenizer_files(src_dir, new_dir, backup_suffix=".bak"):
    import os, shutil
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ]
    for name in candidates:
        new_fp = os.path.join(new_dir, name)
        old_fp = os.path.join(src_dir, name)
        if os.path.exists(new_fp):
            if os.path.exists(old_fp):
                shutil.move(old_fp, old_fp + backup_suffix)
            shutil.copy2(new_fp, old_fp)
            print(f"replaced {name}")
# replace_tokenizer_files(src_model_dir, dst_save_dir)