#!/usr/bin/python
#coding:utf-8

"""
YOLO implementation in Tensorflow.
"""

"""
How to restore variables from the inception v1 checkpoint:

  import tensorflow as tf
  from nets.inception_v1 import inception_v1
  from nets.inception_utils import inception_arg_scope
  slim = tf.contrib.slim
  x = tf.placeholder(tf.float32, [None, 224, 224, 3])
  with slim.arg_scope(inception_arg_scope()):
      net, endpoints = inception_v1(x,
                                    num_classes=None,
                                    is_training=True,
                                    global_pool=False)
  vts = slim.get_variables_to_restore()
  restorer = tf.train.Saver(vts)
  with tf.Session() as sess:
      restorer.restore(sess, "./inception_v1.ckpt")

You can change the size of `x'. We have modified the arch of the original
inception v1 code, and now:
    with `x' as [None, 224, 224, 3], shape of `net' will be [?, 7, 7, 1024].
    with `x' as [None, 416, 416, 3], shape of `net' will be [?, 13, 13, 1024].

And we don't want global pooling at this level yet, so set global_pool=False.
"""

import pdb
import tensorflow as tf
import cv2
import numpy as np
import math
import random
import Queue
from collections import namedtuple
from backbone.inception_v1 import inception_v1
from backbone.inception_utils import inception_arg_scope

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

tf.app.flags.DEFINE_boolean("is_training", True, "To train or not to train.")

tf.app.flags.DEFINE_string("check_point_file", "./inception_v1.ckpt",
        "Path of checkpoint file. Must come with its parent dir name, "
        "even it is in the current directory (eg, ./model.ckpt).")

tf.app.flags.DEFINE_integer("num_of_classes", 1, "Number of classes.")

# We don't do any clustering here. Just use 5 as a heuristic.
tf.app.flags.DEFINE_integer("num_of_anchor_boxes", 5,
        "Number of anchor boxes.")

# NOTE
#   1. Lable files should be put in the same directory and in the YOLO format
#   2. Empty line and lines that start with # will be ignore.
#      (But # at the end will not. Careful!)
tf.app.flags.DEFINE_string("training_file_list", "/disk1/labeled/trainall_roomonly.txt",
        "File which contains all the training images.")

# Format of this file should be:
#
#   0 person
#   1 car
#   2 xxx
#   ...
#
# Empty line and lines that start with # will be ignore.
# (But # at the end will not. Careful!)
tf.app.flags.DEFINE_string("class_name_file", "/disk1/labeled/classnames.txt",
        "File which contains id <=> classname mapping.")

tf.app.flags.DEFINE_string("train_log_dir", "/tmp/train_log",
        "Directory to save training log (and weights).")

tf.app.flags.DEFINE_integer("image_size_min", 320,
        "The minimum size of a image (i.e., image_size_min * image_size_min).")

tf.app.flags.DEFINE_integer("num_of_image_scales", 10,
        "Number of scales used to preprocess images. We want different size of "
        "input to our network to bring up its generality.")

tf.app.flags.DEFINE_integer("num_of_steps", 20000,
        "Max num of step. -1 makes it infinite.")

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

class DatasetReader():
    """ Reader to read images/labels """
    def __init__(self, training_file_list, class_name_file, num_of_classes,
                 shuffle=True):
        """
          Args:
            training_file_list: File contain paths of all training images.
            num_of_classes: Number of classes.
            class_name_file: See FLAGS.class_name_file.

        Note that training images/labels should be in the same directory.
        """
        self.num_of_classes = num_of_classes

        self._images_list = []
        with open(training_file_list, "r") as file:
            for line in file:
                line = line.strip()
                if not len(line) or line.startswith('#'): continue
                self._images_list.append(line)
        if shuffle:
            random.shuffle(self._images_list)

        self.images_queue = Queue.Queue()
        for file in self._images_list:
            self.images_queue.put(file)

        self.label_classname = {}
        with open(class_name_file, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                if not len(parts) == 2:
                    raise ValueError("Class name file incorrect format: %s" % line)
                self.label_classname[parts[0]] = parts[1]

    def next_batch(self, num=50, image_size=320):
        """ Return next batch of images.

          Args:
            num: Number of images to return.
            image_size: Size of image. If the loaded image is not in this size,
                        then it will be resized to [image_size, image_size, 3]
          Return:
            batch_xs: A batch of image, i.e., a numpy array in shape
                      [num, image_size, image_size, 3]
            batch_ys: A batch of ground truth bounding box value, in shape
                      [num, [-1, 5]].
        """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        batch_xs_filename = []
        for _ in range(num):
            batch_xs_filename.append(self.images_queue.get())
        for path in batch_xs_filename:
            self.images_queue.put(path)
        batch_ys_filename = []
        for path in batch_xs_filename:
            batch_ys_filename.append(path.replace('jpg', 'txt'))

        batch_xs = []
        batch_ys = []
        for x_path, y_path in zip(batch_xs_filename, batch_ys_filename):
            orignal_image = cv2.imread(x_path)
            height, weight, channel = orignal_image.shape
            # INTER_NEAREST - a nearest-neighbor interpolation
            # INTER_LINEAR - a bilinear interpolation (used by default)
            # INTER_AREA - resampling using pixel area relation. It may be a preferred
            #              method for image decimation, as it gives moire’-free results
            #              But when the image is zoomed, it is similar to the INTER_NEAREST
            #              method.
            # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
            # INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
            im = cv2.resize(orignal_image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
            batch_xs.append(im)

            yfile = open(y_path, "r")
            for line in yfile:
                line = line.strip()
                if not len(line) or line.startswith('#'): continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError("Invalid label format in %s: %s" % (path, line))
                label, x, y, w, h = parts
                # NOTE we don't have to scale the coordinates, because a point
                # at 0.7 in the original image would also appear at 0.7 at the
                # scaled image.
                label = int(label)
                x = float(x) # x coordinate of the box center
                y = float(y) # y coordinate of the box center
                w = float(w) # width of the box
                h = float(h) # height of the box
                # Some label at COCO dataset has 1.000 as width/height
                assert x<=1 and y<=1 and w<=1 and h<=1, \
                        ("[x,y,w,h]: %s, [_x,_y,_w,_h]: %s, y_path:%s" % (
                           str([x,y,w,h]), str([_x,_y,_w,_h]), y_path))

                batch_ys.append(label)
                batch_ys.append(x)
                batch_ys.append(y)
                batch_ys.append(w)
                batch_ys.append(h)

            # After reading labels from every single file, we add a -1 at the
            # very end as the delimiter so that we can split this batch after it
            # gets passed into a function.
            batch_ys.append(-1)

            yfile.close()

        batch_xs = np.array(batch_xs)
        assert batch_xs.shape == (num, image_size, image_size, 3), "batch_xs shape mismatch"
        batch_ys = np.array(batch_ys)

        tf.logging.info("Shape of batch_xs: " + str(batch_xs.shape))
        tf.logging.info("Shape of batch_ys: " + str(batch_ys.shape))

        return batch_xs, batch_ys

class YOLOLossCalculator():
    """ Provides methos for calculating loss """

    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        return math.exp(-np.logaddexp(0, -x))
        # return 1 / (1 + math.exp(-x))

    @staticmethod
    def _intersec_len(x1, w1, x2, w2):
        """ Calculate intersection len of two lines. """
        assert x1 >= 0 and w1 >= 0 and x2 >= 0 and w2 >= 0, "All four param should >= 0."
        if x1 < x2:
            if x2 >= x1 + w1/2:
                return 0
            return x1 + w1/2 - x2
        elif x1 > x2:
            if x1 >= x2 + w2/2:
                return 0
            return x2 + w2/2 - x1
        else:
            return min(w1, w2)

    @staticmethod
    def calculate_iou(box1, box2):
        """ Calculate box1 ^ box2 """
        assert len(box1) == 4 and len(box2) == 4, "Both box1 and box2 should be of len 4"
        xlen = YOLOLossCalculator._intersec_len(box1[0], box1[2], box2[0], box2[2])
        ylen = YOLOLossCalculator._intersec_len(box1[1], box1[3], box2[1], box2[3])
        return xlen * ylen

    @staticmethod
    def calculate_loss(output, ground_truth, num_of_anchor_boxes,
                       num_of_classes, feed_dict=None):
        """ Calculate loss using the loss from yolov1 + yolov2.

          Args:
            output: Computed output from the network. In shape
                    [batch_num,
                        image_size/32,
                            image_size/32,
                                (num_of_anchor_boxes * (5 + C))]
            ground_truth: Ground truth bounding boxes. In shape (batch_num, ).
                          In each batch, all the labels (length 5) are flattened.
                          We will split them here.
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_classes: See FLAGS.num_of_classes.
            feed_dict: You must provide this to feed _x and _y_gt.
          Return:
              loss: The final loss.
        """

        loss = 0

        assert output.shape[1] == output.shape[2], "Output tensor shape mismatch"
        feature_map_len = int(output.shape[1])
        # coordinates of top left corner and bottom right corner
        cell_info = namedtuple('cell_info',
                                ['top_left_x', 'top_left_y',
                                 'bottom_right_x', 'bottom_right_y'])
        feature_map_info = {}
        for i in range(feature_map_len):
            feature_map_info[i] = {}
            for j in range(feature_map_len):
                feature_map_info[i][j] = cell_info(top_left_x = i*32,
                                                   top_left_y = j*32,
                                                   bottom_right_x = (i+1)*32,
                                                   bottom_right_y = (j+1)*32)

        image_size = feature_map_len*32

        ground_truth_old = ground_truth.eval(feed_dict=feed_dict)
        ground_truth = []
        # Split the ground truth value with -1 as delimiter.
        # list.index(...) sucks.
        split_indexes_rhs = [i for i, j in enumerate(ground_truth_old) if j == -1]
        split_indexes_lhs = [0] # Yah!
        split_indexes_lhs.extend(map(lambda x: x+1, split_indexes_rhs[:-1]))
        split_indexes = zip(split_indexes_lhs, split_indexes_rhs)
        for x, y in split_indexes:
            ground_truth.append([])
            ground_truth[-1].extend(ground_truth_old[x:y])

        ground_truth = map(lambda arr: list(np.array(arr).reshape((-1, 5))), ground_truth)
        for labels in ground_truth:
            for idx, label in enumerate(labels):
                rescale_label = list(label[0:1])
                rescale_label.extend([x*image_size for x in label[1:5]])
                assert label[1] <= 1  and label[2] <= 1 and label[3] <= 1 and label[4] <= 1, \
                        "label[1:5]: %s" % str(label[1:5])
                # we will add a [cell_i, cell_j] to indicate which cell this
                # ground truth belong to.
                l, x, y, w, h = rescale_label
                cell_i = int(x / 32)
                cell_j = int(y / 32)
                assert cell_i < image_size/32 and cell_j < image_size/32, \
                        "cell_i&cell_j exceed: [%d, %d]" % (cell_i, cell_j)
                rescale_label.extend([cell_i, cell_j])
                labels[idx] = rescale_label

        # Turn it into np.array for easy manipulation.
        output_np = output.eval(feed_dict=feed_dict)
        output_np = output_np.reshape(output_np.shape[0], output_np.shape[1], output_np.shape[2],
                                      num_of_anchor_boxes, 5 + num_of_classes)
        for out in output_np:
            for x_idx, x in enumerate(out):
                for y_idx, y in enumerate(x):
                    assert len(y) % num_of_anchor_boxes == 0, "Mismatch when iterate through output_np"
                    for box in y:
                        # Calculate x,y,w,h relative to the center of the cell,
                        # such that:
                        #   - the top left corner of the gt will not shift
                        #     > 1*cellsize away from the cell center.
                        #   - w and h is sigmoid-ed and calculate relative to
                        #     image_size
                        x, y, w, h = box[0:4]
                        ci = feature_map_info[x_idx][y_idx]
                        center_x = (ci.top_left_x + ci.bottom_right_x) / 2
                        center_y = (ci.top_left_y + ci.bottom_right_y) / 2
                        x = center_x + YOLOLossCalculator.sigmoid(x) * 32
                        y = center_y + YOLOLossCalculator.sigmoid(y) * 32
                        w = YOLOLossCalculator.sigmoid(w) * image_size
                        h = YOLOLossCalculator.sigmoid(w) * image_size

                        box[0] = x
                        box[1] = y
                        box[2] = w
                        box[3] = h

        output_abs_np = output_np

        # compute loss for every image in this batch
        for predictions, gt in zip(output_abs_np, ground_truth):
            # shape of predictions:
            #   [image_size/32, image_size/32, num_of_anchor_boxes, (5 + num_of_classes)]
            # shape of gt (ground truth): [-1, 7]
            cell_with_objects = set()
            matched_box = set()
            # pdb.set_trace()
            for label in gt:
                l, x, y, w, h, cell_i, cell_j = label
                assert l < num_of_classes, "Label should be smaller than num_of_classes"

                l = int(l)
                cell_i = int(cell_i)
                cell_j = int(cell_j)

                cell_with_objects.add( (cell_i, cell_j) )

                # cell_predictions.shape == (num_of_anchor_boxes,)
                cell_predictions = predictions[cell_i][cell_j]
                max_iou = 0
                max_iou_pred = None
                max_iou_pred_idx = -1
                for idx, box in enumerate(cell_predictions):
                    iou = YOLOLossCalculator.calculate_iou(box[0:4], [x, y, w, h])
                    # tf.logging.info("iou: " + str(iou))
                    if iou > max_iou:
                        # tf.logging.info("@@@@@@@@@Another max iou: " + str(iou))
                        max_iou = iou
                        max_iou_pred = box
                        max_iou_pred_idx = idx
                if max_iou <= 0:
                    tf.logging.warning("Cannot find a box with intersection with the gt box. "
                                       "Default to the first one")
                    max_iou_pred = cell_predictions[0]
                    max_iou_pred_idx = 0
                else:
                    tf.logging.info("Anchor box[%d] max iou[%f]" % (max_iou_pred_idx, max_iou))

                matched_box.add((cell_i, cell_j, max_iou_pred_idx))

                # For the box that matches the ground truth box, get the loss
                # TODO add hyperparameter
                # TODO What if more than one ground truth box match the same box
                #      If we allow a single box to match more than one ground
                #      truth box, then the 
                #               + math.pow(max_iou_pred[5 + int(l)] - 1, 2))
                #      below should be removed and a more thorough treatment of
                #      the class probablities should be added.
                match_gt_loss = (math.pow(max_iou_pred[0] - x, 2)
                                + math.pow(max_iou_pred[1] - y,2)
                                + math.pow(math.sqrt(max_iou_pred[2]) - math.sqrt(w), 2)
                                + math.pow(math.sqrt(max_iou_pred[3]) - math.sqrt(h), 2)
                                + math.pow(max_iou_pred[4] - 1, 2)
                                + math.pow(max_iou_pred[5 + int(l)] - 1, 2))
                for i in range(num_of_classes):
                    if i == l: continue
                    match_gt_loss += math.pow(max_iou_pred[5 + i] - 0, 2)

                loss += match_gt_loss

            # calculate loss for the anchor box which doesnot match/is-response-for
            # any ground truth box
            # TODO currently we treat all those boxes the same, i.e., regardless
            # of whether their corresponding boxes contain or does not contain
            # an object.
            no_match_loss = 0
            for cell_i, x in enumerate(predictions):
                for cell_j, y in enumerate(x):
                    for idx, box in enumerate(y):
                        if (cell_i, cell_j, idx) not in matched_box:
                            no_match_loss += math.pow(box[4] - 0, 2)
            loss += no_match_loss

        # loss = tf.constant(loss)
        return tf.identity(loss)

# image should be of shape [None, x, x, 3], where x should multiple of 32,
# starting from 320 to 608.
def YOLOvx(images, num_of_anchor_boxes=5, num_of_classes=1, reuse=tf.AUTO_REUSE):
    """ This architecture of YOLO is not strictly the same as in those paper.
    We use inceptionv1 as a starting point, and then add necessary layers on
    top of it. Therefore, we name it `YOLO (version x)`.

    Args:
        images: Input images.
        num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
        num_of_classes: see FLAGS.num_of_classes.
        reuse: Whether or not the network and its variable should be reused.
    Return:
        net: The final YOLOvx network.
        vars_to_restore: Reference to variables that should be restored from
                         the checkpoint file.
    """
    with slim.arg_scope(inception_arg_scope()):
        net, endpoints = inception_v1(images, num_classes=None,
                                      is_training=True, global_pool=False,
                                      reuse=reuse)

    # Get inception v1 variable here, because the network arch will change
    # later on.
    vars_to_restore = slim.get_variables_to_restore()
    inception_net = net

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=trunc_normal(0.01)):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope("extra_inception_module_0", reuse=reuse):
                with tf.variable_scope("Branch_0"):
                    branch_0 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope("Branch_1"):
                    branch_1 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope("Branch_2"):
                    branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 32, [5, 5], scope='Conv2d_0b_3x3')
                with tf.variable_scope("Branch_3"):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3,
                                values=[branch_0, branch_1, branch_2, branch_3])

                # `inception_net' has 1024 channels and `net' has 160 channels.
                #
                # TODO(Yubin): in the original paper, it is not directly added
                # together. Instead, `inception_net' should be at least x times the 
                # size of `net' such that we can "pool concat" them.
                net = tf.concat(axis=3, values=[inception_net, net])

                # w = int(net.get_shape()[0])
                # h = int(net.get_shape()[1])
                # assert w == h, "Width/Height of final net layer not equal"

                # Follow the number of output in YOLOv3
                net = slim.conv2d(net,
                        num_of_anchor_boxes * (5 + num_of_classes),
                        [1, 1],
                        scope="Final_output")

    return net, vars_to_restore

def train(training_file_list, class_name_file, num_of_classes, num_of_image_scales, image_size_min,
          num_of_anchor_boxes, num_of_steps, check_point_file):
    """ Train the YOLOvx network.

      Args:
          training_file_list: See FLAGS.training_file_list.
          class_name_file: See FLAGS.class_name_file.
          num_of_classes: See FLAGS.num_of_classes.
          num_of_image_scales: See FLAGS.num_of_image_scales.
          image_size_min: See FLAGS.image_size_min.
          num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes
          num_of_steps: See FLAGS.num_of_steps.
          check_point_file: See FLAGS.check_point_file.
      Return:
          None.
    """

    # TODO we should not bundle `vars_to_restore' in io_info. A better way
    # will be to split apart the inception module and load its weights only once.
    io_info = namedtuple('io_info', ['size', 'input_tensor', 'output_tensor',
                                     'ground_truth', 'vars_to_restore'])
    variable_size_io_map = []
    variable_sizes = []
    for i in range(num_of_image_scales):
        variable_sizes.append(image_size_min + i*32)
    for size in variable_sizes:
        assert size % 32 == 0, ("Image size should be multiple of 32. "
                                "FLAGS.image_size_min may be wrong.")
        _x = tf.placeholder(tf.float32, [None, size, size, 3])
        _y, _vars_to_restore = YOLOvx(_x)
        # 7: label, x, y, w, h, cell_i, cell_j
        _y_gt = tf.placeholder(tf.float32, [None])
        variable_size_io_map.append(io_info(size=size,
                                            input_tensor=_x,
                                            output_tensor=_y,
                                            ground_truth=_y_gt,
                                            vars_to_restore=_vars_to_restore))

    # load images and labels
    reader = DatasetReader(training_file_list, class_name_file, num_of_classes)

    tf.logging.info("Before session run")
    io_map_idx = 0
    already_loaded_ckpt = False # TODO I hate this flag!
    with tf.Session() as sess:
        initializer = tf.global_variables_initializer()
        sess.run(initializer)

        for i in range(20000): #TODO FLAGS.num_of_steps
            # `size' is the size of input image, not the final feature map size
            size = variable_size_io_map[io_map_idx].size
            _x = variable_size_io_map[io_map_idx].input_tensor
            _y = variable_size_io_map[io_map_idx].output_tensor
            _y_gt = variable_size_io_map[io_map_idx].ground_truth
            _vars_to_restore = variable_size_io_map[io_map_idx].vars_to_restore
            # load initial weights of the inception module
            if not already_loaded_ckpt:
                restorer = tf.train.Saver(_vars_to_restore)
                restorer.restore(sess, check_point_file)
                already_loaded_ckpt = True

            batch_xs, batch_ys = reader.next_batch(num=50, image_size=size)

            loss = YOLOLossCalculator.calculate_loss(
                                    _y,
                                    _y_gt,
                                    num_of_anchor_boxes,
                                    num_of_classes,
                                    feed_dict = {_x: batch_xs, _y_gt: batch_ys}
                                )

            # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            trainable_vars = tf.trainable_variables()
            opt = tf.train.GradientDescentOptimizer(1e-4)
            # print(tf.gradients(loss, trainable_vars))
            grads_and_vars = opt.compute_gradients(loss, trainable_vars)
            # print(grads_and_vars)
            # # eta = opt._learning_rate
            train_step = opt.apply_gradients(grads_and_vars)
            # global_step = tf.Variable(0, trainable=False)
            # starter_learning_rate = 0.01
            # learning_rate = tf.train.exponential_decay(
            #         learning_rate = starter_learning_rate,
            #         global_step = global_step,
            #         decay_steps = 100000,
            #         decay_rate = 0.96,
            #         staircase=True
            #     )
            # train_step = (
            #         tf.train.GradientDescentOptimizer(learning_rate)
            #         .minimize(cross_entropy, global_step=global_step)
            #     )


            # _, loss_val = sess.run([train_step, loss])
            loss_val = sess.run([loss])

            tf.logging.info("LossVal: " + str(loss_val))

            if i % 100 == 0:
                io_map_idx += 1
                io_map_idx %= len(variable_size_io_map)

def main(_):

    if FLAGS.is_training:
        tf.logging.info("yolo.py started in training mode. Starting to train...")
        train(FLAGS.training_file_list,
              FLAGS.class_name_file,
              FLAGS.num_of_classes,
              FLAGS.num_of_image_scales,
              FLAGS.image_size_min,
              FLAGS.num_of_anchor_boxes,
              FLAGS.num_of_steps,
              FLAGS.check_point_file)
    else:
        tf.logging.info("What do you want to do?")

if __name__ == '__main__':
    tf.app.run()
