from __future__ import division
import torch
from torch.autograd import Variable
import cv2
from utilities.util import *
import argparse
from utilities.darknet import Darknet
import pickle as pkl
import datetime as dt
import serial
 
#global variables and parameters
VISIBLE_CLASSES = ["person"] # makes the classes in list the only ones visible at output, leave empty to do nothing
AREA_THRESH = 0.35 #a bbox that takes up this ratio of the detection image relates to a 1.5m distance, may need to calibrate
CAMANGLE = 63 #the FOV angle of the camera

ser = serial.Serial('COM3', 9600, writeTimeout = 0) # Establish the connection on a specific port
det_status = 0 #0 = nothing, 1 = distant, 2 = closeby 

def arg_parse():
    """
    Parse CLAs to the detect module
    """
    parser = argparse.ArgumentParser(description="Yolo v3 Detection Module")
    parser.add_argument(
        "--bs",
        dest="bs",
        help="Batch size",
        default=1
    )
    parser.add_argument(
        "--confidence",
        dest="confidence",
        help="Object Confidence to filter predictions",
        default=0.6
    )
    parser.add_argument(
        "--nms_thresh",
        dest='nms_thresh',
        help="NMS Threshold",
        default=0.3
    )
    parser.add_argument(
        "--cfg",
        dest="cfgfile",
        help="Config File",
        default="cfg/yolov3-tiny.cfg",
        type=str
    )
    parser.add_argument(
        "--weights",
        dest="weightsfile",
        help="weightsfile",
        default="cfg/yolov3-tiny.weights",
        type=str
    )
    parser.add_argument(
        "--reso",
        dest='reso',
        help="Input resolution of the network. Increase to increase accuracy, decrease to increase speed",
        default="256",
        type=str
    )
    parser.add_argument(
        "--video",
        dest='video_file',
        help="Video path to run detection on",
        default="vid.avi",
        type=str
    )

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80  # to suit coco dataset
classes = load_classes("data/coco.names")

# initialise network and load weights
print("Loading network.......")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# if GPU is available, put the model on the GPU
if CUDA:
    model.cuda()

# set model to evaluation mode since it isn't being trained
model.eval()
colors = pkl.load(open("utilities/pallete", 'rb'))

# load feedback images
nodets_img = cv2.imread("imgs/NoDets.jpg")
nodets_img = cv2.resize(nodets_img, (1200, 800))
close_img = cv2.imread("imgs/CloseDet.jpg")
close_img = cv2.resize(close_img, (1200, 800))
dist_img = cv2.imread("imgs/DistDet.jpg")
dist_img = cv2.resize(dist_img, (1200, 800))


def draw(x, frame): 
    """
     Draws bboxes and other perdiction info onto a frame
    """
    cls = int(x[-1])

    if not VISIBLE_CLASSES or classes[cls] in  VISIBLE_CLASSES: #if VISIBLE_CLASSES is empty or cls is visible
        c1 = tuple(x[1:3].int()) # corner 1 of bbox
        c2 = tuple(x[3:5].int()) # corner 2 of bbox
        img = frame
        # x[img_num, dim, dim, dim, dim, obj_score, cls_score, cls]
        
        obj_score = float(x[-3])
        color = colors[cls]
        midx = (c1[0] + c2[0])/2 
        
        cv2.circle(img, (midx, 1), 50 ,color, -1)

        label = "{0} {1:.2f}".format(classes[cls], obj_score)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img
    else:
        return frame


def send(x, fdim):   
    """
    Checks the proximity / location of a person in the prediction x and sends the information to the microcontoller
    """
 
    global det_status
    cls = int(x[-1])
    
    if classes[cls] == "person": 
        c1 = (x[1].item(), x[2].item()) # corner 1 of bbox
        c2 =(x[3].item(), x[4].item()) # corner 2 of bbox
        mid = (c1[0] + c2[0])/2 #horizontal middle of bbox
        pos = mid * (180/fdim[1]) #horizontal angle of bbox from 0 (far left) to 180 (far right)
        relpos = 90 + (pos - 90) * (CAMANGLE/180) #horizontal angle of bbox normalized to cam FOV
        area = (c2[0] - c1[0])*(c2[1] - c1[1])/(fdim[0]*fdim[1]) #percentage of frame taken up be bbox
        
        if area < AREA_THRESH and det_status != 1:
            cv2.imshow("Detection Status", dist_img)
            det_status = 1
        elif area > AREA_THRESH and det_status != 2: #if person gets closer than reccomended 1.5m distance
            cv2.imshow("Detection Status", close_img)
            det_status = 2

        setController(det_status, relpos, ser)



# detection
videofile = args.video_file
cap = cv2.VideoCapture(0)  # videofile for video, 0 for webcam

assert cap.isOpened(), 'Cannot capture video source'

frames = 0

while cap.isOpened():
    start = dt.datetime.today().timestamp()
    ret, frame = cap.read() # get next frame from video source

    if ret:  # if frame read successfully
        img = prep_image(frame, inp_dim)
        # cv2.imshow("raw", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA) # run forward pass through network
        output = write_results(output, confidence, num_classes, nms_conf=nms_thresh)
       
        if type(output) == int: # if no detections happened
            frames += 1
            print("Video FPS: {0:6.2f}      Inference Time: {1:6.2f}".format(
                1/(dt.datetime.today().timestamp() - start), (dt.datetime.today().timestamp() - start)))
            frame = cv2.flip(frame, 1) 
            cv2.imshow("Video Feed", frame)
            key = cv2.waitKey(1)

            cv2.imshow("Detection Status", nodets_img)
            det_status = 0
            setController(det_status, 999 , ser)
    
            if key & 0xFF == ord('q'):
                break
            continue

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(
            inp_dim))  # clamp output between 0 and inp_dim

        im_dim = im_dim.repeat(output.size(0), 1) / inp_dim
        output[:, 1:5] *= im_dim  # resize bbox dimensions

        list(map(lambda x: draw(x, frame), output))
        list(map(lambda x: send(x, frame.shape), output))

        frame = cv2.flip(frame, 1) 
        cv2.imshow("Video Feed", frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1

        print("Video FPS: {0:6.2f}      Inference Time: {1:6.2f}".format(
            1/(dt.datetime.today().timestamp() - start), (dt.datetime.today().timestamp() - start)))
    else:
        break
