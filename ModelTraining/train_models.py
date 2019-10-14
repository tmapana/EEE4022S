import tensorflow as tf

def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'int_list1': tf.io.FixedLenFeature([], tf.int64),
        'float_list1': tf.io.FixedLenFeature([], tf.float32),
        'str_list1': tf.io.FixedLenFeature([], tf.string),
        # If size is different of different records, use VarLenFeature 
        'float_list2': tf.io.VarLenFeature(tf.float32)
    }
    sample = tf.io.parse_single_example(data_record, features)
    return sample

# Initialize tfrecord path
dataset1 = tf.data.TFRecordDataset(['images.tfrecord'])
dataset1 = dataset1.map(extract_fn)

dataset2 = tf.data.TFRecordDataset(['output.tfrecord'])
dataset2 = dataset2.map(extract_fn)
