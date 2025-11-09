from pathlib import Path
import shutil

def copy_all_files(src_dir: str, dst_dir: str):
    """
    将 src_dir 目录下的所有文件复制到 dst_dir 下。
    若目标目录已存在同名文件，会直接覆盖。
    """
    src = Path(src_dir)
    dst = Path(dst_dir)

    if not src.is_dir():
        raise NotADirectoryError(f"源目录不存在或不是目录：{src}")
    dst.mkdir(parents=True, exist_ok=True)

    for file in src.iterdir():
        if file.is_file():
            target = dst / file.name
            # shutil.copy2 会自动覆盖同名文件
            shutil.copy2(file, target)
            print(f"已复制并覆盖：{file} -> {target}")
        else:
            print(f"跳过子目录：{file}")

# 示例调用
json_path="/remote-home/haohh/_cvpr2025/VG-LLM/_post_add_file_/SpatialSense_json_use"
myckpt_path="/remote-home/haohh/_cvpr2025/VG-LLM/ckpt_saves/mhan/flex-percept-coldv2-3e-s18k"
copy_all_files(json_path, myckpt_path)
