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
from tensorflow.python.client import timeline
import cv2
import numpy as np
import math
import random
import sys
import datetime
from time import gmtime, strftime
from inspect import currentframe, getframeinfo
import Queue
from collections import namedtuple
from backbone.inception_v1 import inception_v1
from backbone.inception_utils import inception_arg_scope

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

tf.app.flags.DEFINE_boolean("is_training", True, "To train or not to train.")

tf.app.flags.DEFINE_boolean("freeze_backbone", True,
        "Freeze the backbone network or not")

tf.app.flags.DEFINE_string("check_point_file", "./inception_v1.ckpt",
        "Path of checkpoint file. Must come with its parent dir name, "
        "even it is in the current directory (eg, ./model.ckpt).")

tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size.")

tf.app.flags.DEFINE_integer("num_of_classes", 1, "Number of classes.")

# We don't do any clustering here. Just use 5 as a heuristic.
tf.app.flags.DEFINE_integer("num_of_anchor_boxes", 5,
        "Number of anchor boxes.")

# NOTE
#   1. Lable files should be put in the same directory and in the YOLO format
#   2. Empty line and lines that start with # will be ignore.
#      (But # at the end will not. Careful!)
tf.app.flags.DEFINE_string("training_file_list",
        "/disk1/labeled/trainall_roomonly.txt",
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

# It is pure pain to deal with tensor of variable length tensors (try and screw
# up your life ;-). So we pack each cell with a fixed number of ground truth
# bounding box.
tf.app.flags.DEFINE_integer("num_of_gt_bnx_per_cell", 20,
        "Numer of ground truth bounding box.")

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

    def _get_cell_ij(self, gt_box, image_size):
        """ Scale tensor[1:5] with @image_size and pack [cell_i, cell_j]
            at the end """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        num_of_box = image_size / 32
        percent_per_box = 1.0/num_of_box
        cell_i = math.floor(gt_box[0]/percent_per_box)
        cell_j = math.floor(gt_box[1]/percent_per_box)
        return int(cell_i), int(cell_j)

    def _append_gt_box(self, gt_bnxs, box, image_size, cell_i, cell_j):
        feature_map_len = int(image_size/32)
        assert cell_i <= feature_map_len and cell_j <= feature_map_len, \
                "cell_i/cell_j exceed feature_map_len,\
                [cell_i, cell_j, feature_map_len]: %s" % \
                    str((cell_i, cell_j, feature_map_len))

        if not len(gt_bnxs):
            gt_bnxs = [[[] for i in range(feature_map_len)]
                                for i in range(feature_map_len)]

        gt_bnxs[cell_i][cell_j].append(box)
        return gt_bnxs

    def _pack_and_flatten_gt_box(self, gt_bnxs, image_size, num_of_gt_bnx_per_cell):
        feature_map_len = image_size/32
        for i in range(feature_map_len):
            for j in range(feature_map_len):
                num_of_box = len(gt_bnxs[i][j])
                assert num_of_box <= num_of_gt_bnx_per_cell, \
                        "Number of box[%d] exceed in cell[%d][%d] \
                         (with num_of_gt_bnx_per_cell[%d]). \
                         Consider incresing FLAGS.num_of_gt_bnx_per_cell." \
                         % (num_of_box, i, j, num_of_gt_bnx_per_cell)
                for k in range(num_of_gt_bnx_per_cell-num_of_box):
                    gt_bnxs[i][j].append([0,0,0,0,0])
        gt_bnxs = np.array(gt_bnxs).reshape((feature_map_len, feature_map_len, 5*num_of_gt_bnx_per_cell))
        return gt_bnxs

    def next_batch(self, batch_size=50, image_size=320, num_of_anchor_boxes=5,
                   num_of_classes=1, num_of_gt_bnx_per_cell=20):
        """ Return next batch of images.

          Args:
            batch_size: Number of images to return.
            image_size: Size of image. If the loaded image is not in this size,
                        then it will be resized to [image_size, image_size, 3].
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_classes: See FLAGS.num_of_classes.
            num_of_gt_bnx_per_cell: See FLAGS.num_of_gt_bnx_per_cell.
          Return:
            batch_xs: A batch of image, i.e., a numpy array in shape
                      [batch_size, image_size, image_size, 3]
            batch_ys: A batch of ground truth bounding box value, in shape
                      [batch_size, image_size/32, image_size/32, 5*num_of_gt_bnx_per_cell]
        """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        batch_xs_filename = []
        for _ in range(batch_size):
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
            im = cv2.resize(orignal_image, dsize=(image_size, image_size),
                            interpolation=cv2.INTER_LINEAR)
            batch_xs.append(im)

            yfile = open(y_path, "r")
            gt_bnxs = []
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
                label = float(label)
                x = float(x) # x coordinate of the box center
                y = float(y) # y coordinate of the box center
                w = float(w) # width of the box
                h = float(h) # height of the box
                # Some label at COCO dataset has 1.000 as width/height
                assert x<=1 and y<=1 and w<=1 and h<=1, \
                        ("[x,y,w,h]: %s, [_x,_y,_w,_h]: %s, y_path:%s" % (
                           str([x,y,w,h]), str([_x,_y,_w,_h]), y_path))

                box = [label, x, y, w, h]
                cell_i, cell_j = self._get_cell_ij(box, image_size)
                gt_bnxs = self._append_gt_box(gt_bnxs, box, image_size, cell_i, cell_j)
                assert cell_i<image_size/32 and cell_j<image_size/32, "cell_i/j too large"

            gt_bnxs = self._pack_and_flatten_gt_box(gt_bnxs, image_size,
                                                    num_of_gt_bnx_per_cell)
            batch_ys.append(gt_bnxs)
            yfile.close()

        batch_xs = np.array(batch_xs)
        batch_ys = np.array(batch_ys)
        assert batch_xs.shape == (batch_size, image_size, image_size, 3), \
                "batch_xs shape mismatch. shape: %s, expected: %s" \
                  % (str(batch_xs.shape), str((batch_size, image_size, image_size, 3)))
        assert batch_ys.shape == \
                (batch_size, image_size/32, image_size/32, 5*num_of_gt_bnx_per_cell), \
                "batch_ys shape mismatch. shape: %s, expected: %s" \
                  % (str(batch_ys.shape),
                     str((batch_size, image_size/32, image_size/32, 5*num_of_gt_bnx_per_cell)))

        # tf.logging.info("Shape of batch_xs: " + str(batch_xs.shape))
        # tf.logging.info("Shape of batch_ys: " + str(batch_ys.shape))

        return batch_xs, batch_ys

class YOLOLoss():
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
        # if x1 < x2:
        #     if x2 >= x1 + w1/2:
        #         return 0
        #     return x1 + w1/2 - x2
        # elif x1 > x2:
        #     if x1 >= x2 + w2/2:
        #         return 0
        #     return x2 + w2/2 - x1
        # else:
        #     return min(w1, w2)
        result = \
          tf.cond(x1 < x2, # if x1 < x2
                  lambda: tf.cond(x2 >= x1 + w1/2,
                                  lambda: 0.0,
                                  lambda: x1 + w1/2 - x2, name="_intersec_len_cond_2"),
                  lambda: tf.cond(x1 > x2, # elif x1 > x2
                                  lambda: tf.cond(x1 >= x2 + w2/2,
                                                  lambda: 0.0,
                                                  lambda: x2 + w2/2 - x1, name="_intersec_len_cond_4"),
                                  # else
                                  lambda: tf.minimum(w1, w2), name="_intersec_len_cond_3"), name="_intersec_len_cond_1")
        return result

    @staticmethod
    def _cal_iou_real_gt(box1, box2):
        xlen = YOLOLoss._intersec_len(box1[0], box1[2], box2[0], box2[2])
        ylen = YOLOLoss._intersec_len(box1[1], box1[3], box2[1], box2[3])
        return xlen * ylen

    @staticmethod
    def _cal_iou(out, gt):
        """ Calculate out ^ gt. If gt all == 0, then it is a fake one, -1 """
        result = \
            tf.cond(tf.logical_and(gt[2] == 0, gt[3] == 0),
                    lambda: 0.0,
                    lambda: YOLOLoss._cal_iou_real_gt(out, gt), name="_cal_iou_cond")
        return result

    @staticmethod
    def calculate_ground_truth_loss(idx, gt_boxes, op_boxes):
        # TODO
        num_of_anchor_boxes = 5
        num_of_classes = 1
        num_of_gt_bnx_per_cell = 20

        l = gt_boxes[idx][0]
        x = gt_boxes[idx][1]
        y = gt_boxes[idx][2]
        w = gt_boxes[idx][3]
        h = gt_boxes[idx][4]

        ious_list = []
        for i in range(num_of_anchor_boxes):
            out = op_boxes[i]
            iou = YOLOLoss._cal_iou(out[0:4], tf.stack([x,y,w,h]))
            idx_tensor = tf.cast(tf.constant(i), tf.float32)
            ious_list.append(tf.stack([iou, idx_tensor]))
        ious_list_tensors = tf.stack(ious_list)
        ious_tp_max = tf.reduce_max(ious_list_tensors, 0)
        idx = tf.cast(ious_tp_max[1], tf.int32)
        max_box = op_boxes[i]

        match_gt_loss = (tf.pow(max_box[0]-x, 2)
                        + tf.pow(max_box[1]-y, 2)
                        # TODO gradient of tf.sqrt will probably be -NaN
                        # + tf.pow(tf.sqrt(max_box[2])-tf.sqrt(w), 2)
                        # + tf.pow(tf.sqrt(max_box[3])-tf.sqrt(h), 2)
                        + tf.pow(max_box[2]-w, 2)
                        + tf.pow(max_box[3]-h, 2)
                        + tf.pow(max_box[4]-1, 2)
                        + tf.pow(max_box[5+tf.cast(l, tf.int32)] - 0, 2))
        # for i range(num_of_classes):
            # if i == l: continue
            # match_gt_loss += math.pow(max_iou_pred[5 + i] - 0, 2)
            # pass

    @staticmethod
    def calculate_loss_loop(idx, boxes, losses):
        # TODO
        num_of_anchor_boxes = 5
        num_of_classes = 1
        num_of_gt_bnx_per_cell = 20

        box = boxes[idx]

        split_num = num_of_anchor_boxes*(5+num_of_classes)
        op_boxes = box[0:split_num]
        op_boxes = tf.reshape(op_boxes, [num_of_anchor_boxes, 5+num_of_classes])
        ground_truth = box[split_num:]
        ground_truth = tf.reshape(ground_truth, [num_of_gt_bnx_per_cell, 5])
        # ground_truth = tf.Print(ground_truth, [ground_truth], "ground_truth print: ")

        gt_losses = []
        for i in range(num_of_gt_bnx_per_cell):
            l = ground_truth[i][0]
            x = ground_truth[i][1]
            y = ground_truth[i][2]
            w = ground_truth[i][3]
            h = ground_truth[i][4]

            ious_list = []
            for j in range(num_of_anchor_boxes):
                out = op_boxes[j]
                iou = YOLOLoss._cal_iou(out[0:4], tf.stack([x,y,w,h]))
                idx_tensor = tf.cast(tf.constant(j), tf.float32)
                ious_list.append(tf.stack([iou, idx_tensor]))
            ious_list_tensors = tf.stack(ious_list)
            ious_tp_max = tf.reduce_max(ious_list_tensors, 0)
            index = tf.cast(ious_tp_max[1], tf.int32)
            max_box = op_boxes[index]
            max_box = op_boxes[0]

            match_gt_loss = (tf.pow(max_box[0]-x, 2)
                            + tf.pow(max_box[1]-y, 2)
                            # TODO gradient of tf.sqrt will probably be -NaN
                            # + tf.pow(tf.sqrt(max_box[2])-tf.sqrt(w), 2)
                            # + tf.pow(tf.sqrt(max_box[3])-tf.sqrt(h), 2)
                            + tf.pow(max_box[2]-w, 2)
                            + tf.pow(max_box[3]-h, 2)
                            + tf.pow(max_box[4]-1, 2)
                            + tf.pow(max_box[5+tf.cast(l, tf.int32)] - 0, 2))
                            # TODO other classes penalized too
                            # + tf.pow(max_box[5+tf.cast(l, tf.int32)] - 0, 2))
            # for i range(num_of_classes):
                # if i == l: continue
                # match_gt_loss += math.pow(max_iou_pred[5 + i] - 0, 2)
                # pass
            gt_losses.append(match_gt_loss)


        loss = tf.reduce_sum(tf.stack(gt_losses))
        losses_ = tf.concat([losses[0:idx], tf.stack([tf.stack([loss])]), losses[idx+1:]], axis=0)
        return (idx+1, boxes, losses_)

    @staticmethod
    def calculate_loss(output, ground_truth, batch_size, num_of_anchor_boxes,
                       num_of_classes, num_of_gt_bnx_per_cell):
        """ Calculate loss using the loss from yolov1 + yolov2.

          Args:
            output: Computed output from the network. In shape
                    [batch_size,
                        image_size/32,
                            image_size/32,
                                (num_of_anchor_boxes * (5 + num_of_classes))]
            ground_truth: Ground truth bounding boxes.
                          In shape [batch_size,
                                      image_size/32,
                                        image_size/32,
                                          5 * num_of_gt_bnx_per_cell].
            batch_size: See FLAGS.batch_size.
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_classes: See FLAGS.num_of_classes.
            num_of_gt_bnx_per_cell: See FLAGS.num_of_gt_bnx_per_cell

          Return:
              loss: The final loss (after tf.reduce_mean()).
        """

        # shape of output:
        #   [?, ?, ?, num_of_anchor_boxes * (5 + num_of_classes))
        # shape of ground_truth:
        #   [?, ?, ?, 5 * num_of_gt_bnx_per_cell]

        # concat ground_truth and output to make a tensor of shape 
        #   (?, ?, ?, num_of_anchor_boxes*(5+num_of_classes) + 5*num_of_gt_bnx_per_cell)
        output_and_ground_truth = tf.concat([output, ground_truth], axis=3,
                                            name="output_and_ground_truth_concat")

        shape = tf.shape(output_and_ground_truth)
        batch_size = shape[0]
        row_num = shape[1]
        # flatten dimension
        output_and_ground_truth = \
                tf.reshape(
                    output_and_ground_truth,
                    [-1, num_of_anchor_boxes*(5+num_of_classes) + 5*num_of_gt_bnx_per_cell]
                )
        losses = tf.reduce_sum(tf.zeros_like(output_and_ground_truth), axis=-1, keepdims=True)
        _0, output_and_ground_truth, final_losses = \
                tf.while_loop(
                        lambda idx, boxes, _1 : idx < batch_size*row_num*row_num,
                        YOLOLoss.calculate_loss_loop,
                        [tf.constant(0), output_and_ground_truth, losses])

        loss = tf.reduce_mean(final_losses)

        return loss

# image should be of shape [None, x, x, 3], where x should multiple of 32,
# starting from 320 to 608.
def YOLOvx(images, num_of_anchor_boxes=5, num_of_classes=1,
           reuse=tf.AUTO_REUSE, freeze_backbone=True):
    """ This architecture of YOLO is not strictly the same as in those paper.
    We use inceptionv1 as a starting point, and then add necessary layers on
    top of it. Therefore, we name it `YOLO (version x)`.

    Args:
        images: Input images.
        num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
        num_of_classes: see FLAGS.num_of_classes.
        reuse: Whether or not the network weights should be reused.
        freeze_backbone: Whether or not to free the inception net backbone.
    Return:
        net: The final YOLOvx network.
        vars_to_restore: Reference to variables that should be restored from
                         the checkpoint file.
    """
    with slim.arg_scope(inception_arg_scope()):
        net, endpoints = inception_v1(images, num_classes=None,
                                      is_training=not freeze_backbone,
                                      global_pool=False,
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

                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                # `inception_net' has 1024 channels and `net' has 160 channels.
                #
                # TODO: in the original paper, it is not directly added together.
                # Instead, `inception_net' should be at least x times the size
                # of `net' such that we can "pool concat" them.
                net = tf.concat(axis=3, values=[inception_net, net])


                # Follow the number of output in YOLOv3
                net = slim.conv2d(net,
                        num_of_anchor_boxes * (5 + num_of_classes),
                        [1, 1],
                        scope="Final_output")

    return net, vars_to_restore

def train(training_file_list, class_name_file, batch_size, num_of_classes,
          num_of_image_scales, image_size_min, num_of_anchor_boxes,
          num_of_gt_bnx_per_cell, num_of_steps, check_point_file,
          freeze_backbone=True):
    """ Train the YOLOvx network.

      Args:
          training_file_list: See FLAGS.training_file_list.
          class_name_file: See FLAGS.class_name_file.
          batch_size: See FLAGS.batch_size.
          num_of_classes: See FLAGS.num_of_classes.
          num_of_image_scales: See FLAGS.num_of_image_scales.
          image_size_min: See FLAGS.image_size_min.
          num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
          num_of_gt_bnx_per_cell: See FLAGS.num_of_gt_bnx_per_cell.
          num_of_steps: See FLAGS.num_of_steps.
          check_point_file: See FLAGS.check_point_file.
          freeze_backbone: See FLAGS.freeze_backbone.
      Return:
          None.
    """

    variable_sizes = []
    for i in range(num_of_image_scales):
        variable_sizes.append(image_size_min + i*32)

    tf.logging.info("Building tensorflow graph...")

    # Build YOLOvx only once.
    # Because we have variable size of input, w/h of image are both None (but
    # note that they will eventually have a shape)
    _x = tf.placeholder(tf.float32, [None, None, None, 3])
    _y, _vars_to_restore = YOLOvx(_x,
                                  num_of_anchor_boxes=num_of_anchor_boxes,
                                  num_of_classes=num_of_classes,
                                  reuse=False,
                                  freeze_backbone=True)
    _y_gt = tf.placeholder(tf.float32, [None, None, None, 5*num_of_gt_bnx_per_cell])

    loss = YOLOLoss.calculate_loss(output = _y,
                                   ground_truth = _y_gt,
                                   batch_size=batch_size,
                                   num_of_anchor_boxes=num_of_anchor_boxes,
                                   num_of_classes=num_of_classes,
                                   num_of_gt_bnx_per_cell=num_of_gt_bnx_per_cell)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    tf.logging.info("All network loss/train_step built! Yah!")

    # load images and labels
    reader = DatasetReader(training_file_list, class_name_file, num_of_classes)

    initializer = tf.global_variables_initializer()

    run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False))

    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("/tmp/yolotraining", sess.graph)

    # TODO If run here, will the checkpoint be loaded below?
    sess.run(initializer)

    restorer = tf.train.Saver(_vars_to_restore)
    restorer.restore(sess, check_point_file)
    tf.logging.info("Checkpoint restored!")

    for idx in range(num_of_steps):
        # Change size every 100 steps.
        # `size' is the size of input image, not the final feature map size.
        image_size = variable_sizes[(idx / 100) % len(variable_sizes)]

        batch_xs, batch_ys = reader.next_batch(
                                batch_size=batch_size,
                                image_size=image_size,
                                num_of_anchor_boxes=num_of_anchor_boxes,
                                num_of_classes=num_of_classes,
                                num_of_gt_bnx_per_cell=num_of_gt_bnx_per_cell
                            )

        sys.stdout.write("Running train_step[%d]..." % idx)
        sys.stdout.flush()
        start_time = datetime.datetime.now()
        _, loss_val = sess.run([train_step, loss],
                               feed_dict={_x:batch_xs, _y_gt:batch_ys})
        # merged_summary_trained, _2, loss_val = sess.run(
                            # [merged_summary, train_step, loss],
                            # feed_dict={_x: batch_xs, _y_gt: batch_ys},
                            # # options=run_option,
                            # # run_metadata=run_metadata,
                           # )
        # train_writer.add_summary(merged_summary_trained, idx)
        elapsed_time = datetime.datetime.now() - start_time

        # create timeline object and write run metadata
#        tl = timeline.Timeline(run_metadata.step_stats)
#        chrome_trace_format = tl.generate_chrome_trace_format()
#        with open("/tmp/yolo-trace/chrome_trace_format_%d" % idx, "w+") as f:
#            f.write(chrome_trace_format)

        print("Elapsed time: %s, LossVal: %s" % (str(elapsed_time), str(loss_val)))

def main(_):

    if FLAGS.is_training:
        tf.logging.info("yolo.py started in training mode. Starting to train...")
        train(
            training_file_list    = FLAGS.training_file_list,
              class_name_file     = FLAGS.class_name_file,
              batch_size          = FLAGS.batch_size,
              num_of_classes      = FLAGS.num_of_classes,
              num_of_image_scales = FLAGS.num_of_image_scales,
              image_size_min      = FLAGS.image_size_min,
              num_of_anchor_boxes = FLAGS.num_of_anchor_boxes,
              num_of_gt_bnx_per_cell = FLAGS.num_of_gt_bnx_per_cell,
              num_of_steps        = FLAGS.num_of_steps,
              check_point_file    = FLAGS.check_point_file,
              freeze_backbone     = FLAGS.freeze_backbone)
    else:
        tf.logging.info("What do you want to do?")

if __name__ == '__main__':
    tf.app.run()
