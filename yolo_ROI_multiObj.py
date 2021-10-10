import cv2
import numpy as np
import argparse
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="Data_Uji_Video/data uji (1).mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Data_Uji_image/3002.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)

#Load model yolo
def load_yolo():
    net = cv2.dnn.readNet("yolo_training_3000.weights", "yolo_testing.cfg")
    classes = ["Car"]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #memberi warna bounding box
    colors = (127,255,3)
    return net, classes, colors, output_layers

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap

def display_blob(blob):
    '''
    There images each fir RGB Channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320,320), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_labels(boxes, confidences, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x,y), (x+y, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)

# def detect_image(img_path):
#     model, classes, colors, output_layers = load_yolo()
#     image, height, width, channel, load_image(img_path)