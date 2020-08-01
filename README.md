
# Smart India Hackathon 2020 MS335 CODE MONK

DEEP SEE CRIME

# The Goal

>Can state-of-the-art deep neural networks “See” violence in images and videos ??
To signal an activity that deviates normal patterns with time window.
Video annotation, Video retrieval, and Real-time monitoring. 
Identify and track down the suspects.
Note: 
Real-world anomalous events are complicated and diverse. It is difficult to classify all of the possible anomalous events. 

# The Data

>UCF Crime Dataset 
128 hours long real-world surveillance videos
13 realistic anomalies includes fighting, assault, road accidents.
Weakly labelled training videos. i.e.  data is labelled video level, but which duration isn’t tagged.
https://www.crcv.ucf.edu/projects/real-world/

# The Method

>Our approach considers anomalous and normal events for improper behaviour detection.
1. Formulates anomaly score for a video clip and provides time window of the crime event.
2. Classifies crimes based on Action Recognition task for Video Retrieval and monitoring.
3. Tag the suspects present in the time frame and track them.
# Overall Pipeline
![](https://drive.google.com/uc?export=view&id=1iXy2nLzmF6dJRWgbRxuskNVZlaxjhSVt)

# Technology Stack

#### Pipeline Backend
>Python 3,
Open CV,
Tensorflow,
TF Records,
Pytorch,
SqLite Database

#### Deep Learning Models
>Module 1 - Inflated 3D CNN Model,
Module 2 - PySlowFast Model,
Module 3 - OpenPose + DeepSort 

#### Web Front and Backend
>Java Script, HTML, CSS,
Bootstrap,
Django Restful API, 
Docker Containers

