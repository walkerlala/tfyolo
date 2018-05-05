#!/usr/bin/python
#coding:utf-8

"""
YOLO implementation in Tensorflow.
"""

"""
How to restore variables from the inception v1 checkpoint:

  # import tensorflow as tf
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

from collections import namedtuple
from inspect import currentframe, getframeinfo
from time import gmtime, strftime
import Queue
import cv2
import datetime
import math
import numpy as np
import os
import pdb
import random
import sys
import tensorflow as tf
from tensorflow.python.client import timeline
from backbone.inception_v1 import inception_v1
from backbone.inception_utils import inception_arg_scope

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

tf.app.flags.DEFINE_boolean("is_training", True, "To train or not to train.")

tf.app.flags.DEFINE_boolean("freeze_backbone", False,
        "Freeze the backbone network or not")

tf.app.flags.DEFINE_string("checkpoint_file", "./inception_v1.ckpt",
        "Path of checkpoint file. Must come with its parent dir name, "
        "even it is in the current directory (eg, ./model.ckpt).")
tf.app.flags.DEFINE_boolean("restore_all_variables", False,
        "Whether or not to restore all variables. Default to False, which "
        "means restore only variables for the backbone network")

tf.app.flags.DEFINE_string("train_ckpt_dir", "/disk1/yolo_train_dir",
        "Path to save checkpoints")
tf.app.flags.DEFINE_string("train_log_dir", "/disk1/yolotraining/",
        "Path to save tfevent (for tensorboard)")

tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size.")

tf.app.flags.DEFINE_integer("num_of_classes", 1, "Number of classes.")

tf.app.flags.DEFINE_float("infer_threshold", 0.6, "Objectness threshold")

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
                   num_of_classes=1, num_of_gt_bnx_per_cell=20, normalize_image=True):
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
            if normalize_image:
                # TODO I don't know what exactly the 2nd parameter for
                im = cv2.normalize(np.asarray(im, np.float32), np.array([]),
                                   alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
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
                assert w>0 and h > 0, \
                        ("w&h must > 0. w&h: %s, y_path:%s" % (str([w,h]), y_path))
                if x==0 or y==0:
                    print("WARNING: x|y == 0, x&y: %s, y_path:%s" % (str[x, y], y_path))

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

        return batch_xs, batch_ys

class YOLOLoss():
    """ Provides methos for calculating loss """

    def __init__(self, batch_size, num_of_anchor_boxes,
                 num_of_classes, num_of_gt_bnx_per_cell):
        """
          Args:
            (see their difinition in FLAGS)
        """
        # NOTE, though we can know batch_size when building the network/loss,
        # but we don't use it anywhere when building then network/loss, because,
        # who know we wont' have a variable size batch input?
        self.batch_size = batch_size
        self.num_of_anchor_boxes = num_of_anchor_boxes
        self.num_of_classes = num_of_classes
        self.num_of_gt_bnx_per_cell = num_of_gt_bnx_per_cell

    @staticmethod
    def sigmoid(x):
        return math.exp(-np.logaddexp(0, -x))
        # return 1 / (1 + math.exp(-x))

    @staticmethod
    def bbox_iou_corner_xy(bboxes1, bboxes2):
        """
        Args:
            bboxes1: shape (batch_size, total_bboxes1, 4)
                with x1, y1, x2, y2 point order.
            bboxes2: shape (batch_size, total_bboxes2, 4)
                with x1, y1, x2, y2 point order.

            p1 *-----
               |     |
               |_____* p2

        Returns:
            Tensor with shape (batch_size, total_bboxes1, total_bboxes2)
            with the IoU (intersection over union) of bboxes1[:, i] and
            bboxes2[:, j] in [:, i, j].
        """

        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=2)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=2)

        xI1 = tf.maximum(x11, tf.transpose(x21, perm=[0,2,1]))
        xI2 = tf.minimum(x12, tf.transpose(x22, perm=[0,2,1]))

        yI1 = tf.minimum(y11, tf.transpose(y21, perm=[0,2,1]))
        yI2 = tf.maximum(y12, tf.transpose(y22, perm=[0,2,1]))

        inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI1 - yI2), 0)

        bboxes1_area = (x12 - x11) * (y11 - y12)
        bboxes2_area = (x22 - x21) * (y21 - y22)

        union = (bboxes1_area
                 + tf.transpose(bboxes2_area, perm=[0,2,1])) - inter_area

        return inter_area / (union+0.0001)

    @staticmethod
    def bbox_iou_center_xy(bboxes1, bboxes2):
        """ same as `bbox_overlap_iou_v1', except that we have
            center_x, center_y, w, h instead of x1, y1, x2, y2.

          Args:
            bboxes1: [d1, d21, 4].
            bboxes2: [d1, d22, 4].

          Return:
             [d1, d21, d22, 1]
        """

        x11, y11, w11, h11 = tf.split(bboxes1, 4, axis=2)
        x21, y21, w21, h21 = tf.split(bboxes2, 4, axis=2)

        xi1 = tf.maximum(x11, tf.transpose(x21, perm=[0,2,1]))
        xi2 = tf.minimum(x11, tf.transpose(x21, perm=[0,2,1]))

        yi1 = tf.maximum(y11, tf.transpose(y21, perm=[0,2,1]))
        yi2 = tf.minimum(y11, tf.transpose(y21, perm=[0,2,1]))

        wi = w11/2.0 + tf.transpose(w21/2.0, perm=[0,2,1])
        hi = h11/2.0 + tf.transpose(h21/2.0, perm=[0,2,1])

        inter_area = tf.maximum(wi - (xi1 - xi2), 0) \
                      * tf.maximum(hi - (yi1 - yi2), 0)

        bboxes1_area = w11 * h11
        bboxes2_area = w21 * h21

        union = (bboxes1_area
                 + tf.transpose(bboxes2_area, perm=[0,2,1])) - inter_area

        # some invalid boxes should have iou of 0 instead of NaN
        # If inter_area is 0, then this result will be 0; if inter_area is
        # not 0, then union is not too, therefore adding a epsilon is OK.
        return inter_area / (union+0.0001)

    def concat_tranpose_broadcast(self, gt_boxes_padded, op_boxes):
        """ A concat operation with transpose semantic.

            Args:
              gt_boxes_padded: [-1, num_of_gt_bnx_per_cell, 5+num_of_classes].
              op_boxes: [-1, num_of_anchor_boxes, 5+num_of_classes].

            Return:
              [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 2, 5+num_of_classes]
        """
        num_of_gt_bnx_per_cell = self.num_of_gt_bnx_per_cell
        num_of_anchor_boxes = self.num_of_anchor_boxes
        num_of_classes = self.num_of_classes

        gt_tile = tf.tile(gt_boxes_padded, [1,1,num_of_anchor_boxes])
        gt_reshape = tf.reshape(
                        gt_tile,
                        [-1, num_of_anchor_boxes * num_of_gt_bnx_per_cell, 5+num_of_classes]
                    )

        op_tile = tf.tile(op_boxes, [1, num_of_gt_bnx_per_cell, 1])
        gt_op = tf.concat([gt_reshape, op_tile], axis=-1)
        return tf.reshape(
              gt_op,
              [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 2, 5+num_of_classes]
            )

    def bbox_iou_center_xy_pad_original(self, gt_boxes, op_boxes):
        """
          Args:
           gt_boxes: [-1, num_of_gt_bnx_per_cell, 5]
           op_boxes: [-1, num_of_anchor_boxes, 5+num_of_classes]

          Return:
            ious: [num_of_gt_bnx_per_cell, num_of_anchor_boxes, 3, 5+num_of_classes],
                  with
                    [:, :, :, 0, 0] the iou, and [:, :, :, 0, :] padded to 5+num_of_classes
                    [:, :, :, 1, :] the ground_truth, padded to 5+num_of_classes
                    [:, :, :, 2, :] the output
        """
        num_of_anchor_boxes = self.num_of_anchor_boxes
        num_of_classes = self.num_of_classes
        num_of_gt_bnx_per_cell = self.num_of_gt_bnx_per_cell

        ious_0 = YOLOLoss.bbox_iou_center_xy(gt_boxes[:, :, 1:5], op_boxes[:, :, 0:4])
        # [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 1]
        ious_1 = tf.expand_dims(ious_0, axis=-1)
        # [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 5+num_of_classes]
        ious_paddings = [[0,0], [0,0], [0,0], [0, num_of_classes+4]]
        ious_2 = tf.pad(ious_1, ious_paddings)
        # [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 1, 5+num_of_classes]
        ious = tf.expand_dims(ious_2, -2)

        # [-1, num_of_gt_bnx_per_cell, 5+num_of_classes]
        gt_paddings = [[0,0], [0,0], [0, num_of_classes]]
        gt_boxes_padded = tf.pad(gt_boxes, paddings=gt_paddings)

        # [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 2, 5+num_of_classes]
        op_gt_bundle = self.concat_tranpose_broadcast(
                        gt_boxes_padded,
                        op_boxes
                    )
        bundle = tf.concat([ious, op_gt_bundle], -2)
        return bundle

    def calculate_loss_inner(self, op_and_gt_batch):

        num_of_anchor_boxes = self.num_of_anchor_boxes
        num_of_classes = self.num_of_classes
        num_of_gt_bnx_per_cell = self.num_of_gt_bnx_per_cell

        split_num = num_of_anchor_boxes*(5+num_of_classes)
        op_boxes = tf.reshape(op_and_gt_batch[..., 0:split_num],
                              [-1, num_of_anchor_boxes, 5+num_of_classes])
        gt_boxes = tf.reshape(op_and_gt_batch[..., split_num:],
                              [-1, num_of_gt_bnx_per_cell, 5])
        # There are many some fake ground truth (i.e., [0,0,0,0,0]) in gt_boxes, 
        # but we can't naively remove it here. Instead, we use tf.ceil() below
        # to calculate loss. That way, if it is a fake gt box with [0,0,0,0,0]
        # tf.ceil() will return 0, doing no harm.
        #
        # Note that to make it "no harm" for those real gt box, a real gt box
        # should contain coordinates < 1 (if > 1, you can also scale it to
        # <1 before using tf.ceil())

        # get each gt's max iou and do some necessary padding
        ious = self.bbox_iou_center_xy_pad_original(gt_boxes, op_boxes)
        # ious of shape
        #   [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 3, 5+num_of_classes],
        # with
        #   [:, :, :, 0, 0] the iou, and [:, :, :, 0, :] padded to 5+num_of_classes
        #   [:, :, :, 1, :] the ground_truth, padded to 5+num_of_classes
        #   [:, :, :, 2, :] the output
        # We reduce along the 2 axis here, that is, find the max ious for
        # each ground truth box.
        ious_max = tf.reduce_max(ious, axis=2, keepdims=False)

        # TODO gradient of tf.sqrt will probably be -NaN, so we use pow for all
        cooridniates_se = tf.pow(ious_max[:, :, 2, 0:4] - ious_max[:, :, 1, 1:5], 2)
        cooridniates_se_ceil = tf.ceil(ious_max[:, :, 1, 1:5])
        cooridniates_se_real_gt = cooridniates_se * cooridniates_se_ceil
        coordinate_loss = tf.reduce_sum(cooridniates_se_real_gt)

        objectness_se = tf.pow(ious_max[:, :, 2, 4] - 1, 2)
        objectness_se_ceil = tf.ceil(ious_max[:, :, 1, 2])
        objectness_se_real_gt = objectness_se * objectness_se_ceil
        objectness_loss = tf.reduce_sum(objectness_se_real_gt)

        #TODO now we have only one class so we are hard-code that the class
        # is at position 0
        classness_se = tf.pow(ious_max[:, :, 2, 0] - 1, 2)
        classness_se_ceil = tf.ceil(ious_max[:, :, 1, 2])
        classness_se_real_gt = classness_se * classness_se_ceil
        classness_loss = tf.reduce_sum(classness_se_real_gt)

        all_loss = coordinate_loss + objectness_loss + classness_loss

        return all_loss

    def calculate_loss(self, output, ground_truth):
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
          Return:
              loss: The final loss (after tf.reduce_mean()).
        """

        num_of_anchor_boxes = self.num_of_anchor_boxes
        num_of_classes = self.num_of_classes
        num_of_gt_bnx_per_cell = self.num_of_gt_bnx_per_cell

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

        with tf.variable_scope("calculate_loss_inner_scope"):
            loss = self.calculate_loss_inner(output_and_ground_truth)
        return loss

# image should be of shape [None, x, x, 3], where x should multiple of 32,
# starting from 320 to 608.
def YOLOvx(images, num_of_anchor_boxes, num_of_classes, freeze_backbone,
           reuse=tf.AUTO_REUSE):
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

    tf.summary.image("images_input", images, max_outputs=5)

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

    tf.summary.histogram("all_weights", tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS))
    tf.summary.histogram("all_bias", tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES))
    tf.summary.histogram("all_activations", tf.contrib.layers.summarize_collection(tf.GraphKeys.ACTIVATIONS))
    # tf.summary.histogram("all_variables", tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    tf.summary.histogram("all_global_step", tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_STEP))

    # TODO now output is relative to the whole image
    return net, vars_to_restore

def validation(output, images, num_of_anchor_boxes, num_of_classes,
          infer_threshold=0.6):
    """
        Args:
         output: YOLOvx network output, shape
                 [None, None, None, num_of_anchor_boxes * (5+num_of_classes)]
         images: input images, shape [None, None, None, 3]
         infer_threshold: See FLAGS.infer_threshold.
         num_of_anchor_boxes:
         num_of_classes:

        Return:

    """

    with tf.variable_scope("validation_scope"):
        image_shape = tf.shape(images)
        batch_size = image_shape[0]
        output_shape = tf.shape(output)
        output_row_num = output_shape[1]

        # scale output
        # NOTE(TODO) these coordinates of these box are all relative to the whole image
        output = tf.reshape(
                output,
                [batch_size, output_row_num * output_row_num * num_of_anchor_boxes, 5+num_of_classes]
            )
        # output_bool = tf.equal(output, tf.constant([0.0]))
        # output_bool_sum = tf.reduce_sum(tf.cast(output_bool, tf.int32))
        # output = tf.Print(output, [output_bool_sum], "output_bool_sum: ", summarize=10000)

        # get P(class) = P(object) * P(class|object)
        p_class = output[:, :, 5:] * tf.expand_dims(output[:, :, 4], -1)
        output = tf.concat([output[:, :, 0:5], p_class], axis=-1)

        # mask all the bounding boxes whose objectness value is not greater than
        # threshold.
        output_idx = output[..., 4]
        mask = tf.cast(tf.greater(output_idx, infer_threshold), tf.int32)
        mask = tf.expand_dims(tf.cast(mask, output.dtype), -1)
        masked_output = output * mask
        # TODO now we just draw all the box, regardless its classes.
        boxes_origin = masked_output[..., 0:4]
        boxes = tf.concat([
                boxes_origin[..., 1:2] - boxes_origin[..., 3:4]/2,
                boxes_origin[..., 0:1] - boxes_origin[..., 2:3]/2,
                boxes_origin[..., 1:2] + boxes_origin[..., 3:4]/2,
                boxes_origin[..., 0:1] + boxes_origin[..., 2:3]/2],
                axis=-1
            )
        # boxes = tf.Print(boxes, [boxes], "boxes", summarize=10000)

        result = tf.image.draw_bounding_boxes(images, boxes, name="predict_on_images")
        return result

def validation_all_boxes(output, images, num_of_anchor_boxes, num_of_classes):
    """
        DUPLICATE FROM validation()
    """

    with tf.variable_scope("validation_all_boxes_scope"):
        image_shape = tf.shape(images)
        batch_size = image_shape[0]
        output_shape = tf.shape(output)
        output_row_num = output_shape[1]

        # scale output
        # NOTE(TODO) these coordinates of these box are all relative to the whole image
        output = tf.reshape(
                output,
                [batch_size, output_row_num * output_row_num * num_of_anchor_boxes, 5+num_of_classes]
            )
        # output_bool = tf.equal(output, tf.constant([0.0]))
        # output_bool_sum = tf.reduce_sum(tf.cast(output_bool, tf.int32))
        # output = tf.Print(output, [output_bool_sum], "output_bool_sum: ", summarize=10000)

        # get P(class) = P(object) * P(class|object)
        p_class = output[:, :, 5:] * tf.expand_dims(output[:, :, 4], -1)
        output = tf.concat([output[:, :, 0:5], p_class], axis=-1)

        # mask all the bounding boxes whose objectness value is not greater than
        # threshold.
        # output_idx = output[..., 4]
        # mask = tf.cast(tf.greater(output_idx, infer_threshold), tf.int32)
        # mask = tf.expand_dims(tf.cast(mask, output.dtype), -1)
        # masked_output = output * mask
        # TODO now we just draw all the box, regardless its classes.
        boxes_origin = output[..., 0:4]
        boxes = tf.concat([
                boxes_origin[..., 1:2] - boxes_origin[..., 3:4]/2,
                boxes_origin[..., 0:1] - boxes_origin[..., 2:3]/2,
                boxes_origin[..., 1:2] + boxes_origin[..., 3:4]/2,
                boxes_origin[..., 0:1] + boxes_origin[..., 2:3]/2],
                axis=-1
            )
        # boxes = tf.Print(boxes, [boxes], "boxes", summarize=10000)

        result = tf.image.draw_bounding_boxes(images, boxes, name="predict_on_images_all_boxes")
        return result

def build_images_with_ground_truth(images, ground_truth, num_of_gt_bnx_per_cell):
    """Put ground truth boxes on images.

      Args:
        images: [batch_size, image_size, image_size, 3]
        ground_truth: [batch_size, num_of_gt_bnx_per_cell*feature_map_len*feature_map_len, 5]

      Return:
          images with boxes on it.
    """
    with tf.variable_scope("build_image_scope"):
        feature_map_len = tf.shape(images)[1]/32
        ground_truth = tf.reshape(
                            ground_truth,
                            [-1, num_of_gt_bnx_per_cell*feature_map_len*feature_map_len, 5]
                        )
        x = ground_truth[..., 1:2]
        y = ground_truth[..., 2:3]
        w = ground_truth[..., 3:4]
        h = ground_truth[..., 4:5]
        boxes = tf.concat([
                    y - h/2, # ymin
                    x - w/2, # xmin
                    y + h/2, # ymax
                    x + w/2  # xmax
                ],
                axis=-1
            )
        result = tf.image.draw_bounding_boxes(images, boxes, name="ground_truth_on_images")
        return result

def train(training_file_list, class_name_file, batch_size, num_of_classes,
          num_of_image_scales, image_size_min, num_of_anchor_boxes,
          num_of_gt_bnx_per_cell, num_of_steps, checkpoint_file,
          restore_all_variables, train_ckpt_dir, train_log_dir, freeze_backbone,
          infer_threshold):
    """ Train the YOLOvx network.

      Args:
          See definitions for all args in FLAGS definition above.

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
    _y, backbone_vars = YOLOvx(_x,
                               num_of_anchor_boxes=num_of_anchor_boxes,
                               num_of_classes=num_of_classes,
                               freeze_backbone=freeze_backbone,
                               reuse=tf.AUTO_REUSE)
    _y_gt = tf.placeholder(tf.float32, [None, None, None, 5*num_of_gt_bnx_per_cell])

    images_with_grouth_boxes = build_images_with_ground_truth(
                                    _x,
                                    _y_gt,
                                    num_of_gt_bnx_per_cell=num_of_gt_bnx_per_cell
                                )
    tf.summary.image("images_with_grouth_boxes", images_with_grouth_boxes)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # Not all var in the backbone graph are trainable.
    all_vars.extend(backbone_vars)
    all_vars = list(set(all_vars))

    losscal = YOLOLoss(
                  batch_size=batch_size,
                  num_of_anchor_boxes=num_of_anchor_boxes,
                  num_of_classes=num_of_classes,
                  num_of_gt_bnx_per_cell=num_of_gt_bnx_per_cell
                )
    loss = losscal.calculate_loss(output = _y, ground_truth = _y_gt)
    tf.summary.scalar("finalloss", loss)
    global_step = tf.Variable(0, name='self_global_step', trainable=False)
    all_vars.append(global_step)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss, global_step=global_step)

    validation_images = validation(output=_y, images=_x,
                                   num_of_anchor_boxes=num_of_anchor_boxes,
                                   num_of_classes=num_of_classes,
                                   infer_threshold=infer_threshold)
    tf.summary.image("images_validation", validation_images, max_outputs=5)

    validation_images_all_boxes = validation_all_boxes(output=_y, images=_x,
                                   num_of_anchor_boxes=num_of_anchor_boxes,
                                   num_of_classes=num_of_classes)
    tf.summary.image("images_validation_all_boxes", validation_images_all_boxes)
    tf.logging.info("All network loss/train_step built! Yah!")

    # load images and labels
    reader = DatasetReader(training_file_list, class_name_file, num_of_classes)

    initializer = tf.global_variables_initializer()

    # run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    run_option = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False))

    merged_summary = tf.summary.merge_all()
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    elif not os.path.isdir(train_log_dir):
        print("{} already exists and is not a dir. Exit.".format(train_log_dir))
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    sess.run(initializer)

    if restore_all_variables:
        vars_to_restore = all_vars
    else:
        vars_to_restore = backbone_vars
    restorer = tf.train.Saver(vars_to_restore)
    restorer.restore(sess, checkpoint_file)
    # value of `global_step' is restored as well
    tf.logging.info("Checkpoint restored!")
    saver = tf.train.Saver(all_vars)

    idx = sess.run(global_step)
    while idx != num_of_steps:
        # Change size every 100 steps.
        # `size' is the size of input image, not the final feature map size.
        image_size = variable_sizes[(idx / 100) % len(variable_sizes)]
        if idx % 100 == 0 and idx:
            print("Switching to another image size: %d" % image_size)

        batch_xs, batch_ys = reader.next_batch(
                                batch_size=batch_size,
                                image_size=image_size,
                                num_of_anchor_boxes=num_of_anchor_boxes,
                                num_of_classes=num_of_classes,
                                num_of_gt_bnx_per_cell=num_of_gt_bnx_per_cell
                            )

        sys.stdout.write("Running train_step[{}]...".format(idx))
        sys.stdout.flush()
        start_time = datetime.datetime.now()
        train_summary, loss_val,_1,_2,_3,_4 = sess.run(
                        [merged_summary, loss, train_step,
                         validation_images, images_with_grouth_boxes,
                         validation_images_all_boxes],
                        feed_dict={_x: batch_xs, _y_gt: batch_ys},
                        options=run_option,
                        # run_metadata=run_metadata,
                        )
        train_writer.add_summary(train_summary, idx)
        elapsed_time = datetime.datetime.now() - start_time
        sys.stdout.write("Elapsed time: {}, LossVal: {:10.10f} | ".format(elapsed_time, loss_val))

        print("Validating this batch....")

        if idx % 500  == 0:
            ckpt_name = os.path.join(train_ckpt_dir, "model-%d.ckpt" % idx)
            if not os.path.exists(train_ckpt_dir):
                os.makedirs(train_ckpt_dir)
            elif not os.path.isdir(train_ckpt_dir):
                print("{} is not a directory.".format(train_ckpt_dir))
                return -1
            saver.save(sess, ckpt_name, global_step=global_step)

        idx += 1

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
              checkpoint_file    = FLAGS.checkpoint_file,
              restore_all_variables = FLAGS.restore_all_variables,
              train_ckpt_dir      = FLAGS.train_ckpt_dir,
              train_log_dir       = FLAGS.train_log_dir,
              freeze_backbone     = FLAGS.freeze_backbone,
              infer_threshold     = FLAGS.infer_threshold)
    else:
        tf.logging.info("What do you want to do?")

if __name__ == '__main__':
    tf.app.run()
