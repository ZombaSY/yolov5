import tensorflow as tf
import cv2
import numpy as np
import os
import copy


INPUT_PATH = 'data/images/2023_01_17_15_50_01_524.png'
model_tf = tf.lite.Interpreter('pretrained/best.tflite')


# function for non-utf-8 string
def cv2_imread(fns_img, color=cv2.IMREAD_UNCHANGED):
    img_array = np.fromfile(fns_img, np.uint8)
    img = cv2.imdecode(img_array, color)
    return img


def cv2_imwrite(fns_img, img):
    extension = os.path.splitext(fns_img)[1]
    result, encoded_img = cv2.imencode(extension, img)

    if result:
        with open(fns_img, mode='w+b') as f:
            encoded_img.tofile(f)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def prediction_to_box(pred, img):
    score_th = 0.35
    # @@@@@@@@@@@@@@@@@ obj class confidence 기준으로 자르기!!!
    pred_high_conf = copy.deepcopy(pred[0])
    # pred_high_conf = pred[pred[..., 4] > score_th]
    for i in range(5, len(pred_high_conf[0, 5:])):  # for all classes
        pred_high_conf[:, i] = pred_high_conf[:, 4] * pred_high_conf[:, i]  # object_conf * class_conf
        # pred_high_conf[:, i] = pred_high_conf[:, i][pred_high_conf[:, i] > score_th]    # confidence로 자르기 @@@@@@@@@@@@@@@@@

    obj_boxes = xywh2xyxy(pred_high_conf[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
    obj_scores = pred_high_conf[:, 4]
    obj_classes = pred_high_conf[:, 5:].argmax(axis=1)  # sort class_conf

    # NMS
    out = cv2.dnn.NMSBoxes(obj_boxes, obj_scores, score_th, 0.8)
    obj_boxes = obj_boxes[out]
    obj_classes = obj_classes[out]
    pred_high_conf = pred_high_conf[out]

    for i in range(len(obj_boxes)):
        p1 = [int(obj_boxes[i][0]), int(obj_boxes[i][1])]
        p2 = [int(obj_boxes[i][2]), int(obj_boxes[i][3])]
        # if obj_classes[i] == 0:     # class
        #     cv2.rectangle(img, p1, p2, (0, 0, 255), 5)
        # else:
        #     cv2.rectangle(img, p1, p2, (255, 0, 0), 5)
        # cv2.putText(img, f'Class {obj_classes[i]} {str(class_score[i][5 + obj_classes[i]])[:4]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 2)

        if obj_classes[i] == 0:
            cv2.putText(img, f'hair1 {str(pred_high_conf[i][5 + obj_classes[i]])[:4]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
        elif obj_classes[i] == 1:
            cv2.putText(img, f'hair2 {str(pred_high_conf[i][5 + obj_classes[i]])[:4]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(img, p1, p2, (0, 255, 255), 3)
        elif obj_classes[i] == 2:
            cv2.putText(img, f'hair3 {str(pred_high_conf[i][5 + obj_classes[i]])[:4]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
        elif obj_classes[i] == 3:
            cv2.putText(img, f'hair4 {str(pred_high_conf[i][5 + obj_classes[i]])[:4]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(img, p1, p2, (255, 0, 0), 3)
        elif obj_classes[i] == 4:
            cv2.putText(img, f'hair5 {str(pred_high_conf[i][5 + obj_classes[i]])[:4]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.rectangle(img, p1, p2, (255, 255, 0), 3)
        elif obj_classes[i] == 5:
            cv2.putText(img, f'hair_white {str(pred_high_conf[i][5 + obj_classes[i]])[:4]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(img, p1, p2, (255, 255, 255), 3)

    cv2.imwrite('out.png', img)


model_tf.allocate_tensors()
input_details = model_tf.get_input_details()
output_details = model_tf.get_output_details()

img_src = cv2_imread(INPUT_PATH, cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, (512, 512))

img_tf = img_src / 255.0
img_tf = img_tf.astype(np.float32)
img_tf = np.transpose(img_tf, [2, 0, 1])
img_tf = img_tf.astype(np.float32)
img_tf = np.expand_dims(img_tf, axis=0)
img_tf = tf.convert_to_tensor(img_tf)

model_tf.set_tensor(input_details[0]['index'], img_tf)
model_tf.invoke()
output_data = model_tf.get_tensor(output_details[0]['index'])   # (1, 25200, 85) -> (b, input_size, 5 + class_num)
prediction_to_box(output_data, img_src)
