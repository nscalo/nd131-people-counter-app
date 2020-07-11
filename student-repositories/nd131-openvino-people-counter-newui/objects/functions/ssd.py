from protocol_buffer import *
import numpy as np
import cv2

face_detector = proto_create_face()
people_detector = proto_create_people()

# use haar cascade to extract face points
def opencv_haar(img, face_detector):
    face_cascade = cv2.CascadeClassifier(face_detector.HAAR_PATH)
    faces_multi = face_cascade.detectMultiScale(img, face_detector.scale_factor, face_detector.min_neighbors)
    
    def extract_face(face):
        return face.astype(np.int64)

    return np.apply_along_axis(extract_face, 1, faces_multi)

def draw_color(color):
    if(color == "red"):
        return (0,0,255)
    elif(color == "blue"):
        return (255,0,0)
    elif(color == "green"):
        return (0,255,0)

def draw_boxes(img, x1, y1, x2, y2, face_detector):
    cv2.rectangle(img, (x1,y1), (x2,y2), draw_color(face_detector.box_color))
    return img

# use caffe net to perform opencv dnn
def opencv_caffe_postprocess(net, img, face_detector):
    retval = cv2.dnn.blobFromImage(img, face_detector.dnn_scale_factor, 
    (face_detector.width, face_detector.height), 
    (face_detector.R_scale, face_detector.G_scale, face_detector.B_scale), 
    False, True)
    net.setInput(retval)
    out = net.forward()

    def extract(detection):
        _, label, conf, xmin, ymin, xmax, ymax = tuple(detection)
        if conf > face_detector.confidence_level:
            xmin = int(xmin * img.shape[1])
            ymin = int(ymin * img.shape[0])
            xmax = int(xmax * img.shape[1])
            ymax = int(ymax * img.shape[0])

            x = max(0, min(xmin, img.shape[1] - 1))
            y = max(0, min(ymin, img.shape[0] - 1))
            w = max(0, min(xmax - x + 1, img.shape[1] - x))
            h = max(0, min(ymax - y + 1, img.shape[0] - y))

            img = draw_boxes(img, x, y, x+w, y+h, face_detector)
        return np.expand_dims(img,0)

    return np.apply_along_axis(extract, -1, out[0][0]).sum(axis=0)

# use single-shot detector detections output to draw bounding boxes
def ssd_postprocess(detections, face_detector=None, people_detector=None):

    detector = None
    if face_detector is not None:
        detector = face_detector
    elif people_detector is not None:
        detector = people_detector

    if detector is None:
        raise Exception("detector not available")
    
    img = np.zeros((detector.output_ssd.height, detector.output_ssd.width))
    
    def extract(detection):
        _, label, conf, xmin, ymin, xmax, ymax = tuple(detection)
        if conf > face_detector.confidence_level:
            xmin = int(xmin * img.shape[1])
            ymin = int(ymin * img.shape[0])
            xmax = int(xmax * img.shape[1])
            ymax = int(ymax * img.shape[0])

            img = draw_boxes(img, xmin, ymin, xmax, ymax, face_detector)
            return np.expand_dims(img,0)

    return np.apply_along_axis(extract, -1, detections[0][0]).sum(axis=0)

# use single-shot detector detections output to extract image points
def ssd_extract(img, detections, face_detector):

    def extract(detection):
        _, label, conf, xmin, ymin, xmax, ymax = tuple(detection)
        if conf > face_detector.confidence_level:
            xmin = int(xmin * img.shape[1])
            ymin = int(ymin * img.shape[0])
            xmax = int(xmax * img.shape[1])
            ymax = int(ymax * img.shape[0])

            return np.array([xmin,xmax,ymin,ymax])
    
    return np.apply_along_axis(extract, -1, detections[0][0])

