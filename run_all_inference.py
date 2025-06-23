import os
import subprocess
from pathlib import Path

# === 設定 ===
ROOT_INPUT_DIR = Path("C:/Users/kihar/programs/data/img/alan/bg10")
ROOT_OUTPUT_DIR = Path("C:/Users/kihar/programs/data/parsing/alan/bg10")
SNAPSHOT_FROM = "C:/Users/kihar/programs/6/CDGNet/model_best.pth"
SCRIPT_PATH = "inference.py"

GPU_IDS = "0"
BS = 1
INPUT_SIZE = "256,256"
NUM_CLASSES = 12
VIS = "yes"

# === 各CAMフォルダを探索して実行 ===
for set_dir in sorted(ROOT_INPUT_DIR.glob("*")):
    if not set_dir.is_dir():
        continue
    for cam_dir in sorted(set_dir.glob("CAM*")):
        if not cam_dir.is_dir():
            continue

        # 入力パス・出力パスの定義
        input_path = str(cam_dir)
        output_path = str(ROOT_OUTPUT_DIR / set_dir.name / cam_dir.name)

        print(f"Running inference on: {input_path}")

        cmd = [
            "python", SCRIPT_PATH,
            "--data-dir", input_path,
            "--gpu", GPU_IDS,
            "--batch-size", str(BS),
            "--input-size", INPUT_SIZE,
            "--restore-from", SNAPSHOT_FROM,
            "--dataset", "val",
            "--num-classes", str(NUM_CLASSES),
            "--output-path", output_path,
            "--vis", VIS
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print(f"[OK] Finished: {input_path}")
        else:
            print(f"[ERROR] Failed: {input_path}")
            print(result.stderr)
