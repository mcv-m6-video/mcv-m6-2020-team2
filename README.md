# Video Surveillance for Road Traffic Monitoring 

The goal of the project is to provide a robust system for road traffic monitoring using computer vision techniques for video analysis. 


## Team Members

| Roger Casals | María Gil |Oscar Mañas| Laura Mora|
|--------------|-----------|-----------|-----------|
|rogercasalsvilardell@gmail.com| maria.gilaragones@gmail.com |oscmansan@gmail.com| lmoraballestar@gmail.com|

## Execution
 
Execute the program as follows:
 
 ```bash
python main.py -h
usage: main.py [-h] [-w WEEK] [-t TASK]

M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring

optional arguments:
  -h, --help            show this help message and exit
  -w WEEK, --week WEEK  week to execute. Options are [1,2]
  -t TASK, --task TASK  task to execute. Options are [1,2,3,4]
```
Figures for the results will be saved in `results/` 


## Week 1
 
The first step of the project is to implement the evaluation metrics needed to prove the robustness of the tool. We implement metrics for:
 
* **Object Detection**:
    * Mean Intersection over Union
    * Mean Average Precision

* **Optical Flow**:
    * Mean Square Error in Non-occluded areas
    * Percentage of Erroneous Pixels in Non-occluded areas

[Report](https://docs.google.com/presentation/d/1AVxaY5epmUaZSwrJ3hu4xbOBTjpmJxgu0Pms7UMf96Q/edit#slide=id.g81008797c9_19_94)

## Week 2

The goal of this week is to model the background of a video sequence in order to estimate the foreground objects. The background is modelled using a single gaussian method, both non-adaptive and adaptive. 

For this, we use grayscale videos as well as colour information from different color spaces. We also compare our method with several SOTA methods from public python libraries.

[Report](https://docs.google.com/presentation/d/1u4jSk3mfiY-k0kEzO-j3TatIRomean1OOWEv-l1FgTk/edit#slide=id.g613c54889_097)

## Week 3

The goal of this week is to perform object detection for car and from those detections track each one of the cars independently.

* Object Detection:
    * Mask-RCNN and Faster-RCNN are used "off-the-shelf", as they are trained with the COCO dataset (which provides with the necessary class **car**)
    * Fine tuned Mask-RCNN to our dataset.
    
* Tracking
    * By overlap, we consider the IoU between boxes in consecutives frames in order to assign a unique id to a car 
    * Using the Kalman filter

[Report](https://docs.google.com/presentation/d/1wegcV2f-nD0tWDgEJ-S8RpuJD-X4PL5EHW8cjQ3wiMo/edit#slide=id.p)

## Week 4

[Report](https://docs.google.com/presentation/d/1-bFIreSw1XCeDrp7cc1dxFsglaH_gSK5kbGmUOzlrus/edit?usp=sharing)
