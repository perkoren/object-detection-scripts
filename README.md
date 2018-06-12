# object-detection-scripts
Python helper scripts for Object Detection in TF

## generate_tfrecord_2_sources.py
Currently there is no support for merging multiple TFRecords into one.
It appears that always only one TFRecord is used during training (https://github.com/tensorflow/models/issues/3031), but sometimes it is necessary to merge mutliple records coming from different sourcses (PNG or JPG files) and image definitions (Yaml, CSV or XML). This script demonstrates how to generate TF Record when there are two image sources:
* JPG files and CSV definitions (generated from Pascal VOC dataset)
* PNG files and Yaml definitions
