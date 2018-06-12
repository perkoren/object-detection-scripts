"""
Generate TF record from two source types - Yaml and CSV defined.
Remember to issue from tensorflow/models beforehand:
  # export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Usage:
  # Create train data:
  python generate_tfrecord_2_sources.py -t train --csv_input=data/train.csv  --yaml_input=data/train.yaml --labels_path=data/labels.txt --output_path=train.record

  # Create test data:
  python generate_tfrecord_2_sources.py -t test --csv_input=data/test.csv  --yaml_input=data/test.yaml --labels_path=data/labels.txt --output_path=test.record

labels.txt should contain labels for object classes - 1 item per each line

For exemplary purposes: 
CSV input -> JPG images
Yaml input -> PNG images
"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import yaml

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('yaml_input', '', 'Path to the Yaml input')
flags.DEFINE_string('labels_path', '', 'Path to text file with labels - one per line')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('t', '', 'train or test')
FLAGS = flags.FLAGS

def readLabels():
    with open(FLAGS.labels_path) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    f.close()
    return content	 
     

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(height,width,filename,encoded_image,image_format,xmins,xmaxs,ymins,ymaxs,classes_text,classes):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

def create_yaml_tf(example):

    filename = example['path']
    filename = filename.encode()

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image = fid.read()
    
    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    width, height = image.size
    image_format = 'png'.encode() 

    xmins = [] 
    xmaxs = []
    ymins = [] 
    ymaxs = []
    classes_text = [] 
    classes = [] 

    content_map = readLabels()

    for box in example['boxes']:
        xmins.append(float(box['x_min'] / width))
        xmaxs.append(float(box['x_max'] / width))
        ymins.append(float(box['y_min'] / height))
        ymaxs.append(float(box['y_max'] / height))
        classes_text.append(box['label'].encode())
        classes.append(content_map.index(box['label']) + 1)

    return create_tf_example(height,width,filename,encoded_image,image_format,xmins,xmaxs,ymins,ymaxs,classes_text,classes)

def create_csv_tf(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    content_map = readLabels()	

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(content_map.index(row['class']) + 1)

    return create_tf_example(height,width,filename,encoded_image,image_format,xmins,xmaxs,ymins,ymaxs,classes_text,classes)


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  
    path = os.path.join(os.getcwd(), FLAGS.t)
    
    #from csv
    csv_desc = pd.read_csv(FLAGS.csv_input)
    grouped = split(csv_desc, 'filename')
    for group in grouped:
        csv_tf = create_csv_tf(group, path)
        writer.write(csv_tf.SerializeToString())
    print('CSV data added to TF record')
    
    #from yaml
    yaml_desc = yaml.load(open(FLAGS.yaml_input, 'rb').read())

    len_yaml_desc = len(yaml_desc)

    for i in range(len_yaml_desc):
        yaml_desc[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(FLAGS.yaml_input), yaml_desc[i]['path']))
    
    counter = 0
    for line in yaml_desc:
        yaml_tf = create_yaml_tf(line)
        writer.write(yaml_tf.SerializeToString())

    print('Yaml data added to TF record')
    print('TF record created: ' + FLAGS.output_path)
    writer.close()
   
   
if __name__ == '__main__':
    tf.app.run()
