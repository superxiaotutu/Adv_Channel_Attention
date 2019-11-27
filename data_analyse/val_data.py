import tensorflow as tf
import os
import tensorflow.contrib.slim as slim

PATH_IMAGENET = "/mnt/exdata2/ImageNet_tfrecord/tfrecord"


class ImageNet_datastream:
    def __init__(self, sess, batchsize=10, imgsize=299):
        self.sess = sess
        self.val_img_batch, self.val_label_batch = self.read_and_decode(PATH_IMAGENET, batchsize, imgsize)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)

    def read_and_decode(self, path, batchsize=10, imgsize=224):
        file_path = os.path.join(path, "validation-*")
        num_samples = 50000

        dataset = self.get_record_dataset(file_path, num_samples=num_samples, num_classes=1000)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
        image, label = data_provider.get(['image', 'label'])

        image = self._fixed_sides_resize(image, output_height=imgsize, output_width=imgsize)

        img_batch, label_batch = tf.train.batch([image, label], batch_size=batchsize, allow_smaller_final_batch=True)
        label_batch = tf.one_hot(label_batch - 1, 1000)
        return img_batch, label_batch

    def _fixed_sides_resize(self, image, output_height, output_width):
        """Resize images by fixed sides.

        Args:
            image: A 3-D image `Tensor`.
            output_height: The height of the image after preprocessing.
            output_width: The width of the image after preprocessing.
        Returns:
            resized_image: A 3-D tensor containing the resized image.
        """
        output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
        output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_nearest_neighbor(
            image, [output_height, output_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    def get_record_dataset(self, record_path, reader=None, num_samples=1281167, num_classes=1000):
        """Get a tensorflow record file.

        Args:

        """
        if not reader:
            reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/class/label':
                tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1],
                                                                         dtype=tf.int64))}

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                                  format_key='image/format'),
            'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        items_to_descriptions = {
            'image': 'An image with shape image_shape.',
            'label': 'A single integer.'}
        return slim.dataset.Dataset(
            data_sources=record_path,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            num_classes=num_classes,
            items_to_descriptions=items_to_descriptions,
            labels_to_names=labels_to_names)

    def get_test_batch(self):
        image, label = self.sess.run([self.val_img_batch, self.val_label_batch])
        return image, label
