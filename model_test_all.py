import os
import subprocess


def process_A(file_path):
    command = [
    "python",
    "C:/Users/suppo/YOLOX/tools/demo.py",
    "video",
    "-f",
    "C:/Users/suppo/YOLOX/exps/example/custom/number_plate.py",
    "-c",
    "./YOLOX_outputs/number_plate/best_ckpt.pth",
    "--path",
    # "C:/Users/suppo/YOLOX/test_data/videos/shinonome03.mp4",
    f"{file_path}",
    "--save_result",
    "--device",
    "gpu",
    "--conf",
    "0.7"
    ]
    subprocess.run(command)
    print(f"Processing video file: {file_path}")

def process_B(file_path):
    import subprocess

    command = [
        "python",
        "C:/Users/suppo/YOLOX/tools/demo.py",
        "image",
        "-f",
        "C:/Users/suppo/YOLOX/exps/example/custom/number_plate.py",
        "-c",
        "./YOLOX_outputs/number_plate/best_ckpt.pth",
        "--path",
        # "./test_data/images",
        f"{file_path}",
        "--conf",
        "0.7",
        "--nms",
        "0.45",
        "--tsize",
        "640",
        "--save_result",
        "--device",
        "gpu"
        ]
    subprocess.run(command)
    print(f"Processing image file: {file_path}")

# フォルダのパス
folder_path = 'C:\\Users\\suppo\\YOLOX\\test_data\\test'

# フォルダ内の全てのファイルをチェック
for filename in os.listdir(folder_path):
    # ファイルのフルパスを取得
    file_path = os.path.join(folder_path, filename)

    # ファイルの拡張子を取得
    _, ext = os.path.splitext(file_path)

    # 拡張子に基づいて処理を分岐
    if ext.lower() in ['.mp4', '.avi', '.mov']:  # 映像ファイルの拡張子
        process_A(file_path)
    elif ext.lower() in ['.jpg', '.png', '.bmp']:  # 画像ファイルの拡張子
        process_B(file_path)
