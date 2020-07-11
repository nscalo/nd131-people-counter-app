import cv2
import numpy as np
import sys
sys.path.append("../")

class Person():

    def __init__(self):
        pass

    def preprocess_frame(self, frame, net_input_shape):
        ### TODO: Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame
        
    def preprocess_outputs(self, frame, result, args, confidence_level=0.0, t=None):
        '''
        TODO: This method needs to be completed by you
        '''
        confs = []
        boxes = []
        height, width = frame.shape[:2]
        y_pixel = 160
        out_text = "Inference Time: " + str(t*10)
        if len(result[0][0]) > 0:
            for res in result[0][0]:
                _, __, conf, xmin, ymin, xmax, ymax = res
                if conf > confidence_level:
                    xmin = int(xmin*width)
                    ymin = int(ymin*height)
                    xmax = int(xmax*width)
                    ymax = int(ymax*height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)
                    cv2.putText(frame, out_text, (30, y_pixel),
                    cv2.FONT_HERSHEY_COMPLEX, 0.55, (0, 0, 255), 1)
                    confs.append(conf)
                    boxes.append([xmin,ymin,xmax,ymax])
        
        return boxes, confs