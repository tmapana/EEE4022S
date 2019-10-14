# Training a new model to use in my project, reference:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
# EEE4022S - Final Year Project
# Tlotliso Mapana
# MPNTLO002

import os
import matplotlib.image as mpimg
import sys
sys.path.append("/home/tlotliso/Documents/EEE4022S/Model/models-master/research")
sys.path.append("/home/tlotliso/Documents/EEE4022S/Model/models-master/research/object_detection/utils")
import tensorflow as tf
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', 'output.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_boat_tf(fileName):
	height = 427 # Image height
	width = 640 # Image width
	
	fileName = str.encode(fileName)
	
	with open(fileName, 'rb') as jpgFile:
		encoded_image_data = jpgFile.read() # Encoded image bytes
	
	image_format = b'jpeg' # b'jpeg' or b'png'

	xmins = [51.0, 346.0] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [187.0, 638.0] # List of normalized right x coordinates in bounding box
			 # (1 per box)
	ymins = [168.0, 175.0] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [272.0, 352.0] # List of normalized bottom y coordinates in bounding box
			 # (1 per box)
	classes_text = ['Boat'] # List of string class name of bounding box (1 per box)
	classes = [1] # List of integer class id of bounding box (1 per box)

	boat_tf = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(fileName),
		'image/source_id': dataset_util.bytes_feature(fileName),
		'image/encoded': dataset_util.bytes_feature(encoded_image_data),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return boat_tf
	
def main(_):
	writer = tf.io.TFRecordWriter(FLAGS.output_path)
	
	fileName = "coco_3.jpg"
	jpgBoatfile = Image.open(fileName)
	print(jpgBoatfile)
	jpgInfo = jpgBoatfile._getexif()
	
	for tag, value in jpgInfo.items():
		key = TAGS.get(tag, tag)
		print(key + " " + str(value)) 
	
	boat_tf = create_boat_tf(fileName)
	writer.write(boat_tf.SerializeToString())

	writer.close()


if __name__ == '__main__':
	tf.compat.v1.app.run()
