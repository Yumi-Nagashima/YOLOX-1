import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import demo_postprocess, mkdir, multiclass_nms, vis

import argparse
import os
import pandas as pd

output_dir ='onnx_out'
image_path = 'C:\\Users\\suppo\\YOLOX\\test_data\\images\\image10.png'
model = 'yolox_s.onnx'

input_shape = (640,640)
origin_img = cv2.imread(image_path)
img, ratio = preprocess(origin_img, input_shape)
session = onnxruntime.InferenceSession(model)
ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
output = session.run(None, ort_inputs)
predictions = demo_postprocess(output[0], input_shape)[0]
boxes = predictions[:, :4]
scores = predictions[:, 4:5] * predictions[:, 5:]
boxes_xyxy = np.ones_like(boxes)
boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
boxes_xyxy /= ratio
dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.5)

#出力結果を表示
result = []
if dets is not None:
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    [result.extend((final_cls_inds[x],COCO_CLASSES[int(final_cls_inds[x])],final_scores[x],final_boxes[x][0],final_boxes[x][1],final_boxes[x][2],final_boxes[x][3]) for x in range(len(final_scores)))]
df = pd.DataFrame(result, columns = ['class-id','class','score','x-min','y-min','x-max','y-max'])
print(df)

mkdir(output_dir)

# 画像から物体を切り取る
for i, row in df.iterrows():
    x_min, y_min, x_max, y_max = map(int, [row['x-min'], row['y-min'], row['x-max'], row['y-max']])
    cropped_img = origin_img[y_min:y_max, x_min:x_max]

    # 切り取った画像を保存
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.jpg")
    cv2.imwrite(output_path, cropped_img)
