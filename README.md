# object-detection-scripts
Python helper scripts and Coco model files for Object Detection in TF (used in custom model re-trainging for Kotlin detection app)

## generate_tfrecord_2_sources.py
Currently there is no support for merging multiple TFRecords into one.
It appears that always only one TFRecord is used during training (https://github.com/tensorflow/models/issues/3031), but sometimes it is necessary to merge mutliple records coming from different sourcses (PNG or JPG files) and image definitions (Yaml, CSV or XML). This script demonstrates how to generate TF Record when there are two image sources:
* JPG files and CSV definitions (generated from Pascal VOC dataset)
* PNG files and Yaml definitions

## coco-model
Coco model files downloaded from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz and referenced in products.config

## xml_to_csv.py
Slight modification of the conversion script obtained from https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
It creates CSV files from XML definitions of bounding boxes of detected objects.

## generate_tfrecord.py
Script used to generate train.record and test.record files.


## products.config
This config file was used in model re-training for the Kotlin detection app. Based on a config file provided in the Coco data set.

