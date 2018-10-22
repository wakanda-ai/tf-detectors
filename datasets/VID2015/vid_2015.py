import os
import tensorflow as tf

def parse_tfrecord(example):
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
    new_height = 320
    new_width = 320
    image = tf.image.resize_images(image, (new_height, new_width), tf.image.ResizeMethod.BICUBIC)
    image = tf.image.convert_image_dtype(image, tf.float32) # [T, H, W, 3]
    bbox = tf.stack([tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/ymin']),
                     tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/xmin']),
                     tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/ymax']),
                     tf.sparse_tensor_to_dense(sequence_parsed['image/object/bbox/xmax']),], axis=-1) # [T, O, 4]
    return image, bbox, height, width, frame_number


class VID2015InputProducer(object):
    def __init__(self):
        return

    def get_input(self,
                  base_dir, file_patterns=['*.tfrecord'],
                  num_threads=4,
                  batch_size=32,
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

        data_files = [tf.gfile.Glob(os.path.join(base_dir,file_pattern))
                      for file_pattern in file_patterns]
        data_files = [data_file for sublist in data_files for data_file in sublist]
        dataset = tf.data.TFRecordDataset(data_files, num_parallel_reads=num_threads)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=num_threads)
        dataset = dataset.shuffle(batch_size * 100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        self.iterator = dataset.make_one_shot_iterator()
        [images, bboxes, height, width, frame_number] = self.iterator.get_next()
        return images, bboxes, height, width, frame_number


## Test
test_dir='./tfrecord'
result_dir='./example_result'

global_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
input_producer = VID2015InputProducer()
images, bboxes, height, width, frame_number = input_producer.get_input(test_dir, batch_size=1, num_threads=4)

images_shape = tf.concat([[-1], tf.shape(images)[-3:]], axis=0)
bboxes_shape = tf.concat([[-1], tf.shape(bboxes)[-2:]], axis=0)
images = tf.reshape(images, images_shape)
bboxes = tf.reshape(bboxes, bboxes_shape)
images_bb = tf.image.draw_bounding_boxes(images, bboxes)
#bboxes = tf.Print(bboxes, [bboxes])

#tf.summary.image('images', images)
tf.summary.image('images_with_bounding_box', images_bb, max_outputs=20)
summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
summary_op = tf.summary.merge(summaries)
summary_writer = tf.summary.FileWriter(result_dir)

#with tf.train.MonitoredTrainingSession(checkpoint_dir=result_dir) as sess:
import time
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        start_time = time.time()

        _images, _bboxes, _height, _width, _frame_number, _global_step = \
            sess.run([images, bboxes, height, width, frame_number, global_step])
        print(_global_step)
        print(_frame_number)
        _summary = sess.run(summary_op)
        summary_writer.add_summary(_summary, global_step=_global_step)
        print(time.time()-start_time)

#    os.system('tensorboard --logdir={} --port=8888'.format(result_dir))

