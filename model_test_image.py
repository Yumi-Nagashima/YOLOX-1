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
    "./test_data/images",
    "--conf",
    "0.3",
    "--nms",
    "0.45",
    "--tsize",
    "640",
    "--save_result",
    "--device",
    "gpu"
]

subprocess.run(command)
