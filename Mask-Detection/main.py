import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
from components import config
from components.prior_box import priors_box
from components.utils import decode_bbox_tf, compute_nms, show_image
from network.network import SlimModel
from playsound import playsound
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
capture = None
video = None
cameraflag = False
alertflag = False
modelflag = False
filename = None
cfg = None
model = None
priors = None
countFrame = 0


def parse_predict(predictions, priors, cfg):
    label_classes = cfg["labels_list"]
    bbox_regressions, confs = tf.split(predictions[0], [4, -1], axis=-1)
    boxes = decode_bbox_tf(bbox_regressions, priors, cfg["variances"])
    confs = tf.math.softmax(confs, axis=-1)
    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(label_classes)):
        cls_scores = confs[:, c]
        score_idx = cls_scores > cfg["score_threshold"]
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]
        nms_idx = compute_nms(cls_boxes, cls_scores,
                              cfg["nms_threshold"], cfg["max_number_keep"])
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]
        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)
    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)
    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()
    return boxes, classes, scores


def SelectModel():
    global modelflag
    global filename
    global cfg
    global model
    global priors
    filename = filedialog.askopenfilename(filetypes={('Model files', '.h5')})
    if filename and os.path.exists(filename):
        cfg = config.cfg
        min_sizes = cfg["min_sizes"]
        num_cell = [len(min_sizes[k]) for k in range(len(cfg["steps"]))]
        model = SlimModel(cfg=cfg, num_cell=num_cell, training=False)
        model.load_weights(filename)
        priors, _ = priors_box(cfg, image_sizes=(480, 640))
        priors = tf.cast(priors, tf.float32)
        modelflag = True
        messagebox.showinfo('Success', 'Model loaded')


def DetectMask():
    global capture
    global video
    global cameraflag
    global alertflag
    global cfg
    global model
    global priors
    global countFrame
    _, frame = capture.read()
    if not modelflag:
        messagebox.showinfo('Error', 'Please select a model first')
        CloseCamera()
    elif frame is None and cameraflag:
        messagebox.showinfo('Error', 'No camera found')
        CloseCamera()
    elif cameraflag:
        h, w, _ = frame.shape
        countFrame += 1
        img = np.float32(frame.copy())
        img = img / 255.0 - 0.5
        predictions = model(img[np.newaxis, ...])
        boxes, classes, scores = parse_predict(predictions, priors, cfg)
        for prior_index in range(len(classes)):
            show_image(frame, boxes, classes, scores, h,
                       w, prior_index, cfg["labels_list"])
        if str(classes) == '[2]' and countFrame % 16 == 1 and alertflag:
            playsound(r'audio\RedAlert.mp3', False)
            countFrame = 1
        cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2img)
        imgtk = ImageTk.PhotoImage(image=img)
        video.imgtk = imgtk
        video.configure(image=imgtk)
        video.after(1, DetectMask)


def OpenCamera():
    global capture
    global cameraflag
    if not cameraflag:
        cameraflag = True
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        DetectMask()


def CloseCamera():
    global capture
    global cameraflag
    if cameraflag:
        cameraflag = False
        capture.release()
        video.imgtk = None
        video.configure(image=None)


def OpenAlert():
    global alertflag
    alertflag = True


def CloseAlert():
    global alertflag
    alertflag = False


def main():
    global video
    root = tk.Tk()
    root.title('Mask Detection')
    root.geometry('720x565+620+268')
    video = tk.Label(root)
    video.place(x=40, y=20)
    SelectModelBtn = tk.Button(
        text='Select Model',
        command=SelectModel,
        font=(
            'Arial',
            12,
            'bold'))
    SelectModelBtn.place(height=25, width=120, x=20, y=520)
    CloseCameraBtn = tk.Button(
        text='Close Camera',
        command=CloseCamera,
        font=(
            'Arial',
            12,
            'bold'))
    OpenCameraBtn = tk.Button(
        text='Open Camera',
        command=OpenCamera,
        font=(
            'Arial',
            12,
            'bold'))
    OpenCameraBtn.place(height=25, width=120, x=160, y=520)
    CloseCameraBtn = tk.Button(
        text='Close Camera',
        command=CloseCamera,
        font=(
            'Arial',
            12,
            'bold'))
    CloseCameraBtn.place(height=25, width=120, x=300, y=520)
    OpenAlertBtn = tk.Button(
        text='Open Alert',
        command=OpenAlert,
        font=(
            'Arial',
            12,
            'bold'))
    OpenAlertBtn.place(height=25, width=120, x=440, y=520)
    CloseAlertBtn = tk.Button(
        text='Close Alert',
        command=CloseAlert,
        font=(
            'Arial',
            12,
            'bold'))
    CloseAlertBtn.place(height=25, width=120, x=580, y=520)
    root.mainloop()


if __name__ == '__main__':
    main()
