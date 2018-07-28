import os
import tensorflow as tf

class VID2015InputProducer(object):
    def __init__(self):
        return

    def get_input(self,
                  base_dir,
                  file_patterns=['*.tfrecord'],
                  num_threads=4,
                  batch_size=32,
                  bucket_boundaries=[128,256,512],
                  allow_smaller_final_batch=True,
                  input_device=None,
                  num_epochs=None):
        """Get input tensors bucketed by frame number

        Returns:
            images : float32 image tensor [T, N, H, W, 3] padded to batch max frame_number T
            bboxes : float32 bounding boxes [T, N, O, 4]
            height : int32 video height
            width : int32 video widths
            frame_number : int32 the number of frames in video
        """
        queue_capacity = num_threads*batch_size*2
        data_queue = self._get_data_queue(base_dir,
                                          file_patterns,
                                          capacity=queue_capacity,
                                          num_epochs=num_epochs)
        with tf.device(input_device):
            image, bbox, height, width, frame_number = self._read_single_example(data_queue)
            tensors = [image, bbox, height, width, frame_number]
            frame_number, tensors = tf.contrib.training.bucket_by_sequence_length(
                    input_length=frame_number,
                    tensors=tensors,
                    bucket_boundaries=bucket_boundaries,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    capacity=queue_capacity,
                    #keep_input=keep_input,
                    allow_smaller_final_batch=allow_smaller_final_batch,
                    dynamic_pad=True)
            [images, bboxes, height, width, frame_number] = tensors
        return images, bboxes, height, width, frame_number

    def preprocess(self, record):
        return record

    def _get_input_filter(self):
        pass

    def _get_data_queue(self,
                        base_dir,
                        file_patterns=['*.tfrecord'],
                        capacity=2**15,
                        num_epochs=None):
        """Get a data queue for a list of record files"""
        data_files = [tf.gfile.Glob(os.path.join(base_dir,file_pattern))
                      for file_pattern in file_patterns]
        data_files = [data_file for sublist in data_files for data_file in sublist]
        data_queue = tf.train.string_input_producer(data_files,
                                                    shuffle=True,
                                                    capacity=capacity,
                                                    num_epochs=num_epochs)
        return data_queue

    def _read_single_example(self, data_queue):
        reader = tf.TFRecordReader()
        key, example = reader.read(data_queue)
        context_features = {
                'video/folder': tf.FixedLenFeature([], dtype=tf.string),
                'video/frame_number': tf.FixedLenFeature([1], dtype=tf.int64,
                                                        default_value=1),
                'video/height': tf.FixedLenFeature([], dtype=tf.int64),
                'video/width': tf.FixedLenFeature([], dtype=tf.int64),
                }
        sequence_features = {
                'image/filename': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/encoded': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/sources': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/key/sha256': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/format': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
                'image/object/name': tf.VarLenFeature(dtype=tf.string),
                'image/object/occluded': tf.VarLenFeature(dtype=tf.int64),
                'image/object/generated': tf.VarLenFeature(dtype=tf.int64),
                }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=example,
                context_features=context_features,
                sequence_features=sequence_features)

        height = context_parsed['video/height']
        width = context_parsed['video/width']
        frame_number = tf.cast(context_parsed['video/frame_number'], tf.int32)

        def _decode_image(_image):
            return tf.image.decode_jpeg(_image, channels=3)
        image = tf.map_fn(_decode_image,
                          sequence_parsed['image/encoded'],
                          dtype=tf.uint8)
        image = tf.image.convert_image_dtype(image, tf.float32) # [T, H, W, 3]
        bbox = tf.stack([tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/ymin']),
                         tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/xmin']),
                         tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/ymax']),
                         tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/xmax']),], axis=-1) # [T, O, 4]
        return image, bbox, height, width, frame_number

## Test
#test_dir='./example'
#
#global_step = tf.train.get_or_create_global_step()
#input_producer = VID2015InputProducer()
#images, bboxes, height, width, frame_number = input_producer.get_input(test_dir, batch_size=1)
#
#images = tf.squeeze(images, [0])
#bboxes = tf.squeeze(bboxes, [0])
#images = images[0]
#bboxes = bboxes[0]
#bboxes = tf.Print(bboxes, [bboxes])
#images_bb = tf.image.draw_bounding_boxes(images, bboxes)
#tf.summary.image('images', images)
#tf.summary.image('images_with_bounding_box', images_bb)
#summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
#summary_op = tf.summary.merge(summaries)
#
#with tf.train.MonitoredTrainingSession(checkpoint_dir=test_dir) as sess:
#    while not sess.should_stop():
#        sess.run(summary_op)
#        break
#    os.system('tensorboard --logdir={} --port=8888'.format(test_dir))
##        _images, _bboxes, _height, _width, _frame_number = \
##            sess.run([images, bboxes, height, width, frame_number])

