import numpy as np
from PIL import Image
import argparse
import cv2

def parse_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run face detection on image")
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", required=True, 
    default='/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    required.add_argument("-t", required=True, default='OpenCV-Haar')
    optional.add_argument("-i", default=None, required=True)
    optional.add_argument("-d", default='CPU')
    optional.add_argument("-s", default=1.1, type=float)
    optional.add_argument("-n", default=4, type=int)
    optional.add_argument("-c", default='GREEN')
    optional.add_argument("-th", default=1, type=int)
    optional.add_argument("-recog", default=False, type=bool)
    optional.add_argument("-next_image", default=None, required=False)
    optional.add_argument("-size", default=70, required=False)
    optional.add_argument("-save_faces", default=False, required=False)
    optional.add_argument("-draw_image", default=False, required=False)
    args = parser.parse_args()

    return args

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,210,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']

def draw_boxes(frame, xmin, ymin, xmax, ymax, args):
    '''
    Draw bounding boxes onto the frame.
    '''
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)
    return frame

def extract_faces(faces, img):
    images = []
    for yes_face in faces:
        images.append(img[yes_face[1]:yes_face[1]+yes_face[3],
        yes_face[0]:yes_face[0]+yes_face[2]])
    return images

if __name__ == "__main__":

    def path_to_tensor(img, args):
        face_cascade = cv2.CascadeClassifier(args.m)
        faces_multi = face_cascade.detectMultiScale(img, args.s, args.n)
        faces = [face.astype(np.int64).tolist() for face in faces_multi]
        return faces
    
    def draw(yes_face, frame, args):
        frame = draw_boxes(frame, yes_face[0], 
        yes_face[1], yes_face[0]+yes_face[2], yes_face[1]+yes_face[3], args)
        return frame
    
    args = parse_args()
    args.c = convert_color(args.c)
    
    img = Image.open(args.i).convert('RGB')
    image = np.asarray(img)
    gray = img.convert('L')
    gray = np.asarray(gray)
    faces = path_to_tensor(gray, args)
    orig_images = extract_faces(faces, image)
    if args.draw_image:
        if len(faces) > 0:
            for face in faces:
                image = draw(face, image, args)

    if not args.recog:
        cv2.putText(image, "Parameters (face): n:" + args.n.__str__() + " scale: " + str(args.s), 
        (800,15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)

        cv2.imwrite("face-detector-1-1-amplitude-ans.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        _img = Image.open(args.next_image).convert('RGB')
        _image = np.asarray(_img)
        gray = _img.convert('L')
        gray = np.asarray(gray)
        faces = path_to_tensor(gray, args)
        next_images = extract_faces(faces, _image)
        if args.draw_image:
            if len(faces) > 0:
                for face in faces:
                    _image = draw(face, _image, args)

        s = args.size
        _image = _image.copy()
        image = image.copy()
        for ii,i in enumerate(orig_images):
            if args.save_faces:
                cv2.imwrite("data/faces/face-"+ii.__str__()+".png", orig_images[ii])
            f = cv2.resize(orig_images[ii], (s,s))
            _image[20:(20+s),ii*s+20:ii*s+20+s] = f
            image[20:(20+s),ii*s+20:ii*s+20+s] = f
        for ij,i in enumerate(next_images):
            if args.save_faces:
                cv2.imwrite("data/faces/face-"+(ii+ij).__str__()+".png", next_images[ij])
            f = cv2.resize(next_images[ij], (s,s))
            _image[5*20:5*20+s,ij*s+20:ij*s+20+s] = f
            image[5*20:5*20+s,ij*s+20:ij*s+20+s] = f

        # writing images to export to gifs
        cv2.imwrite('face1.png', cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('face2.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('face3.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('face4.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('face5.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
