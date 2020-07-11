import numpy as np
from protocol_buffer import *
import cv2
import timer
from scipy import signal

face_detector = proto_create_face()

callback = lambda: print("Timer finished!")

def people_count(detection):
    _, label, conf, xmin, ymin, xmax, ymax = tuple(detection)
    return 1 if conf > face_detector.confidence_level else 0

# create the timer for 1 second
def create_timer(init_start=1000000):
    return timer.Timer(init_start, callback)

def time_spent(t: timer.Timer):
    t.stop()
    return t.elapsed

# https://home.ubalt.edu/ntsbarsh/Business-stat/StatistialTables.pdf
# two-tailed
def zscore(prob=0.05):
    if prob == 0.01:
        return 2.57
    elif prob == 0.02:
        return 2.33
    elif prob == 0.03:
        return 2.17
    elif prob == 0.04:
        return 2.05
    elif prob == 0.05:
        return 1.96

def face_recognize_error(face_recognizer, prob_vector1=None, prob_vector2=None, num=1):
    neg_prob_array1 = 1 - prob_vector1
    neg_prob_array2 = 1 - prob_vector2
    return (prob_vector2 * neg_prob_array2 / num).sum() + (0.0 if num == 1 else (prob_vector1 * neg_prob_array1 / (num-1)).sum())

def face_difference(face_recognizer, person=None, new_person=None, num=1):
    error = face_recognize_error(face_recognizer, person, new_person, num)
    significance = zscore(face_recognizer.statistical_significance) * np.sqrt(error)
    if 2*significance <= face_recognizer.statistical_significance:
        return True, (new_person - person - significance, new_person - person + significance)
    else:
        return False, None

def face_recognize_risk(face_detection_enhancer, risk_vector1=None, risk_vector2=None):
    error = np.isclose(risk_vector1, risk_vector2, rtol=None, 
    atol=face_detection_enhancer.rd)
    if np.prod(error.astype(np.int8)) == 1:
        return True
    else:
        return False
