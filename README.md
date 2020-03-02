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
  -w WEEK, --week WEEK  week to execute. Options are [1]
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




