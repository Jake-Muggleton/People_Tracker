Getting Started:
____________________________


1). Install required packages (via conda using req.txt)
2). Install weights in cfg folder:
	wget -c https://pjreddie.com/media/files/yolov3.weights
	wget -c https://pjreddie.com/media/files/yolov3-tiny.weights

3). Run detect.py to perform detection on images (default yolov3, input: /imgs, output: /det)
4). Run video.py to perform detection on video feed (default yolov3-tiny, input: webcam)

5). To change I/O directories and adjust hyperparameters, use the --help argument when running scripts
____________________________


Based on Sources:

https://github.com/ultralytics/yolov3
https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
https://github.com/ayooshkathuria/pytorch-yolo-v3
