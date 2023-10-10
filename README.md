# aurova_controllers
This repository contains the high-level controllers that the blue robot has. These include visual servoing , kinematic and dynamic controllers

## yolo_visual_servoing
This package contains the Viki Hyco application, the controller that merges a visual servo controller with the kinematic controller. The package is based on the Velodyne LiDAR sensor and the front RGB camera of the Blue robot. The target point location is based on: [Detection and depth estimation for domestic waste in outdoor environments by sensors fusion](https://arxiv.org/abs/2211.04085).  And in the repository [Lidar_camera_fusion](https://github.com/EPVelasco/lidar-camera-fusion). This package use the [aurova_detections](https://github.com/AUROVA-LAB/aurova_detections/tree/main) metapackage 
