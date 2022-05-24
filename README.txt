This project was a result of a research bursary program in late 2019 where I studied CNNs (Computational Neural Networks) for efficient computer vision people detection. Following on from this project, I added a method of approximating a person's distance from the camera using the 2D images from a single video source. I also added a microcontroller that is driven by the detector and points to any person in the FOV of the camera, making a noise if that person gets too close. 


Demo Video:
____________________________
https://youtu.be/rgLq40HvlXc 



Getting Started:
____________________________
1). Install required packages (via conda using req.txt)

2). Install weights in cfg folder:
	wget -c https://pjreddie.com/media/files/yolov3-tiny.weights

4). Run detector.py to perform detection on video feed 

5). To change I/O directories and adjust hyperparameters, use the --help command line argument flag when running scripts



Based on Sources:
____________________________
https://github.com/ultralytics/yolov3
https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
https://github.com/ayooshkathuria/pytorch-yolo-v3
