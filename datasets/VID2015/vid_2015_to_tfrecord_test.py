"""
    Usage:
    python vid_2015_to_tfrecord_test.py
"""
import os
import glob
import hashlib
import numpy as np
import tensorflow as tf

from tf_detectors.datasets.VID2015.vid_2015_to_tfrecord import gen_shard
from PIL import Image



class CreateVID2015TFRecordTest(tf.test.TestCase):

    def _assertProtoEqual(self, proto_field, expectation):
      """Helper function to assert if a proto field equals some value.
      Args:
        proto_field: The protobuf field to compare.
        expectation: The expected value of the protobuf field.
      """
      proto_list = [p for p in proto_field]
      self.assertListEqual(proto_list, expectation)

    def test_gen_shard(self):
        # Make random example file
        test_dir = './tmp'
        example_path = 'attempt0'
        annotation_dir = os.path.join(test_dir, 'Annotations/')
        data_dir = os.path.join(test_dir, 'Data/VID/test')
        if os.path.exists(test_dir):
            os.system('rm -rf {}'.format(test_dir))
        os.makedirs(os.path.join(annotation_dir, example_path), exist_ok=True)
        os.makedirs(os.path.join(data_dir, example_path), exist_ok=True)

        # Video information
        frame_number = 20
        height, width = 256, 256
        expectation = {'context_features':{}, 'sequence_features':{}}
        expectation['context_features'] = {'video/folder':example_path.encode('utf8'),
                                           'video/frame_number':frame_number,
                                           'video/height':height,
                                           'video/width':width,}
        expectation['sequence_features'] = {'image/encoded':[],
                                            'image/filename':[],
                                            'image/format':['jpeg'.encode('utf8')]*frame_number,
                                            'image/key/sha256':[],
                                            'image/object/bbox/xmax':[],
                                            'image/object/bbox/xmin':[],
                                            'image/object/bbox/ymax':[],
                                            'image/object/bbox/ymin':[],
                                            'image/object/generated':[0]*frame_number,
                                            'image/object/name':['temp'.encode('utf8')]*frame_number,
                                            'image/object/occluded':[1]*frame_number,
                                            'image/sources':['TestData'.encode('utf8')]*frame_number,}

        for f in range(frame_number):
            ## make annotation files (fake annotation)
            xmax = np.random.randint(width)
            xmin = max(xmax - 50, 0)
            ymax = np.random.randint(height)
            ymin = max(ymax - 50, 0)
            filename = '%06d'%f
            annotation = "\
            <annotation>\n\
                <folder>{}</folder>\n\
                <filename>{}</filename>\n\
                <source>\n\
                    <database>TestData</database>\n\
                </source>\n\
                <size>\n\
                    <width>{}</width>\n\
                    <height>{}</height>\n\
                </size>\n\
                <object>\n\
                    <trackid>0</trackid>\n\
                    <name>temp</name>\n\
                    <bndbox>\n\
                        <xmax>{}</xmax>\n\
                        <xmin>{}</xmin>\n\
                        <ymax>{}</ymax>\n\
                        <ymin>{}</ymin>\n\
                    </bndbox>\n\
                    <occluded>1</occluded>\n\
                    <generated>0</generated>\n\
                </object>\n\
            </annotation>\n\
            ".format(example_path, filename, width, height, xmax, xmin, ymax, ymin)
            annotation_file = open(os.path.join(annotation_dir, example_path, '%06d.xml'%f), 'w')
            annotation_file.write(annotation)
            annotation_file.close()

            ## make image files (fake image)
            frame_data = np.random.rand(height, width, 3)
            frame = Image.fromarray(frame_data, 'RGB')
            save_path = os.path.join(data_dir, example_path, '%06d.JPEG'%f)
            frame.save(save_path)
            with tf.gfile.FastGFile(save_path, 'rb') as fid:
                encoded_jpg = fid.read()

            ## save sequential features
            key = hashlib.sha256(encoded_jpg).hexdigest()
            expectation['sequence_features']['image/encoded'].append(encoded_jpg)
            expectation['sequence_features']['image/filename'].append(filename.encode('utf8'))
            expectation['sequence_features']['image/key/sha256'].append(key.encode('utf8'))
            expectation['sequence_features']['image/object/bbox/xmax'].append(float(xmax)/width)
            expectation['sequence_features']['image/object/bbox/xmin'].append(float(xmin)/width)
            expectation['sequence_features']['image/object/bbox/ymax'].append(float(ymax)/height)
            expectation['sequence_features']['image/object/bbox/ymin'].append(float(ymin)/height)

        # Make temporary tfrecord file(gen_shard)
        out_filename = os.path.join(test_dir, 'test.tfrecord')
        gen_shard([example_path], annotation_dir, out_filename,
                test_dir, 'test')

        # Parse temporary tfrecord file
        data_files = tf.gfile.Glob([out_filename])
        data_queue = tf.train.string_input_producer(data_files)
        reader = tf.TFRecordReader()
        key, example = reader.read(data_queue)
        context_features = {
                'video/folder': tf.FixedLenFeature([], dtype=tf.string),
                'video/frame_number': tf.FixedLenFeature([], dtype=tf.int64),
                'video/height': tf.FixedLenFeature([], dtype=tf.int64),
                'video/width': tf.FixedLenFeature([], dtype=tf.int64),
                }
        sequence_features = {
                'image/filename': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/encoded': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/sources': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/key/sha256': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/format': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/object/bbox/xmin': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                'image/object/bbox/xmax': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                'image/object/bbox/ymin': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                'image/object/bbox/ymax': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                'image/object/name': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/object/occluded': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                'image/object/generated': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=example,
                context_features=context_features,
                sequence_features=sequence_features)

        global_step = tf.train.get_or_create_global_step()
        with tf.train.MonitoredTrainingSession(checkpoint_dir=test_dir) as sess:
            if not sess.should_stop():
                cp, sp = sess.run([context_parsed, sequence_parsed])

        # Assertion : Context features
        self.assertDictEqual(expectation['context_features'], cp)

        # Assertion : Sequence features
        self.assertListEqual(list(expectation['sequence_features'].keys()),
                             list(sp.keys()))

        for key in expectation['sequence_features'].keys():
            if key == 'image/encoded':
                continue
            self.assertListEqual(sorted(list(expectation['sequence_features'][key])),
                                 sorted(list(sp[key])))


# Convert tensor to image format and draw bouding box on it
#image = tf.image.decode_jpeg(sequence_parsed['image/encoded'][0], channels=3)
#image = tf.image.convert_image_dtype(image, tf.float32)
#image = tf.expand_dims(image, 0) # [batch, height, width, 3]
#bounding_box = [[[
#                    sequence_parsed['image/object/bbox/ymin'][0],
#                    sequence_parsed['image/object/bbox/xmin'][0],
#                    sequence_parsed['image/object/bbox/ymax'][0],
#                    sequence_parsed['image/object/bbox/xmax'][0],
#               ]]] # [batch, #_of_bb, 4]
#image_bb = tf.image.draw_bounding_boxes(image, bounding_box)
#tf.summary.image('image', image)
#tf.summary.image('image_with_bounding_box', image_bb)


# Launch tensorboard
#global_step = tf.train.get_or_create_global_step()
#summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
#summary_op = tf.summary.merge(summaries)
#with tf.train.MonitoredTrainingSession(checkpoint_dir=test_dir) as sess:
#    while not sess.should_stop():
#        sess.run(summary_op)
#        break
#os.system('tensorboard --logdir={} --port=8888'.format(test_dir))

if __name__ == '__main__':
    tf.test.main()
