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
import sys
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
# up your life ;-). So we pack the tensor of ground truth bnx to make it 100.
# Those fake ground truth bnx will all be 0 in it.
tf.app.flags.DEFINE_integer("num_of_gt_bnx_fixed", 100,
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

    def _scale_gt_and_pack_cell_ij(self, gt_box, image_size):
        """ Scale tensor[1:5] with @image_size and pack [cell_i, cell_j]
            at the end """
        gt_box_0_1 = gt_box[0:1]
        scaled_gt_box_1_5 = map(lambda x: x * image_size, gt_box[1:5])
        cell_i = map(lambda x: int(x/32), scaled_gt_box_1_5[0:1])
        cell_j = map(lambda y: int(y/32), scaled_gt_box_1_5[1:2])

        gt_box_0_1.extend(scaled_gt_box_1_5)
        gt_box_0_1.extend(cell_i)
        gt_box_0_1.extend(cell_j)
        return gt_box_0_1

    def next_batch(self, batch_size=50, image_size=320, num_of_anchor_boxes=5,
                   num_of_classes=1, num_of_gt_bnx=100):
        """ Return next batch of images.

          Args:
            batch_size: Number of images to return.
            image_size: Size of image. If the loaded image is not in this size,
                        then it will be resized to [image_size, image_size, 3].
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_classes: See FLAGS.num_of_classes.
            num_of_gt_bnx: Every `batch_ys` returned has a fixed number of ground
                           truth bounding box (to make the training easier). If
                           the actual ground truth does not has enough ground truth
                           bounding box, we pack `batch_ys` to make it fixed-length.
          Return:
            batch_xs: A batch of image, i.e., a numpy array in shape
                      [batch_size, image_size, image_size, 3]
            batch_ys: A batch of ground truth bounding box value, in shape
                      [batch_size, num_of_gt_bnx, 7] (that is, 5.. & cell_i & cell_j)
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

                _box = self._scale_gt_and_pack_cell_ij(
                                    [label, x, y, w, h],
                                    image_size
                                )
                gt_bnxs.append(_box)

            assert len(gt_bnxs) < num_of_gt_bnx, "num_of_gt_bnx_fixed too small!"
            num = num_of_gt_bnx - len(gt_bnxs)
            for i in range(num):
                # cell_i & cell_j are both set to -1.0 to prevent matching.
                gt_bnxs.append([0,0,0,0,0,-1.0,-1.0])
            batch_ys.append(gt_bnxs)

            yfile.close()

        batch_xs = np.array(batch_xs)
        batch_ys = np.array(batch_ys)
        assert batch_xs.shape == (batch_size, image_size, image_size, 3), \
                "batch_xs shape mismatch. shape: %s, expected: %s" \
                    % (str(batch_xs.shape), str((batch_size, image_size, image_size, 3)))
        assert batch_ys.shape == (batch_size, num_of_gt_bnx, 7), \
                "batch_ys shape mismatch. shape: %s, expected: %s" \
                    % (str(batch_ys.shape), str((batch_size, image_size, image_size, 3)))

        tf.logging.info("Shape of batch_xs: " + str(batch_xs.shape))
        tf.logging.info("Shape of batch_ys: " + str(batch_ys.shape))

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
                                  lambda: x1 + w1/2 - x2),
                  lambda: tf.cond(x1 > x2, # elif x1 > x2
                                  lambda: tf.cond(x1 >= x2 + w2/2,
                                                  lambda: 0.0,
                                                  lambda: x2 + w2/2 - x1),
                                  # else
                                  lambda: tf.minimum(w1, w2)))
        return result

    @staticmethod
    def _cal_iou(box1, box2):
        """ Calculate box1 ^ box2 """
        xlen = YOLOLoss._intersec_len(box1[0], box1[2], box2[0], box2[2])
        ylen = YOLOLoss._intersec_len(box1[1], box1[3], box2[1], box2[3])
        return xlen * ylen

    @staticmethod
    def _cal_cell_square_error_matched(gt_box, output, feature_map_len,
                                       num_of_anchor_boxes,
                                       num_of_classes):

        l = gt_box[0]
        x = gt_box[1]
        y = gt_box[2]
        w = gt_box[3]
        h = gt_box[4]
        ious_list = []
        for i in range(num_of_anchor_boxes):
            out = output[i]
            iou = YOLOLoss._cal_iou(out[0:4], tf.stack([x,y,w,h]))
            idx_tensor = tf.cast(tf.constant(i), tf.float32)
            ious_list.append(tf.stack([iou, idx_tensor]))
        ious_list_tensors = tf.stack(ious_list)
        # return a tf.Tensor of [idx, iou]
        ious_tp_max = tf.reduce_max(ious_list_tensors, 0)
        idx = tf.cast(ious_tp_max[1], tf.int32)
        max_box = output[idx]

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
        # match_gt_loss = tf.Print(match_gt_loss,
                                # [match_gt_loss, l, x, y, w, h,
                                 # max_box[0], max_box[1], max_box[2], max_box[3], max_box[4]],
                                # "match_gt_loss & l & x & y & w & h | mb[0], mb[1], mb[2], mb[3], mb[4]: ")
        return match_gt_loss
        # we just don't care the boxes that don't match...now...
        # For the box that matches the ground truth box, get the loss
        # TODO add hyperparameter
        # TODO What if more than one ground truth box match the same box
        #      If we allow a single box to match more than one ground
        #      truth box, then the 
        #               + math.pow(max_iou_pred[5 + int(l)] - 1, 2))
        #      below should be removed and a more thorough treatment of
        #      the class probablities should be added.

    @staticmethod
    def _cal_cell_square_error(gt_box, output, cell_i, cell_j, feature_map_len,
                               num_of_anchor_boxes, num_of_classes):
        # l = gt_box[0]
        # x = gt_box[1]
        # y = gt_box[2]
        # w = gt_box[3]
        # h = gt_box[4]
        # gt_cell_i = gt_box[5]
        # gt_cell_j = gt_box[6]
        result = tf.cond(tf.logical_and(tf.abs(tf.cast(cell_i, tf.float32) - gt_box[5]) < 0.0001,
                                        tf.abs(tf.cast(cell_j, tf.float32) - gt_box[6]) < 0.0001),
                         lambda: YOLOLoss._cal_cell_square_error_matched(
                                               gt_box = gt_box,
                                               output = output,
                                               feature_map_len = feature_map_len,
                                               num_of_anchor_boxes = num_of_anchor_boxes,
                                               num_of_classes = num_of_classes
                                          ),
                         lambda: tf.constant(0.0))
        # result = tf.Print(result, [l, x, y, w, h, gt_cell_i, gt_cell_j, cell_i, cell_j],
                            # "[l, x, y, w, h, gt_cell_i, gt_cell_j, cell_i, cell_j]: ")
        return result

    @staticmethod
    def _cal_square_error(tensor, feature_map_len, num_of_gt_bnx,
                          num_of_anchor_boxes, num_of_classes):
        """ The ground truth box may not appear in the same cell as the
            output. There is a cell_i/cell_j informatioin in @gt_box, and
            we know which cell the output belong to by @cell_i and @cell_j.
        """

        cell_ij = tensor[0:1][0]
        output = tensor[1:1+num_of_anchor_boxes * (5+num_of_classes)]
        output = tf.reshape(output, [num_of_anchor_boxes, (5+num_of_classes)])
        ground_truth = tensor[1+num_of_anchor_boxes * (5+num_of_classes):]
        ground_truth = tf.reshape(ground_truth, [num_of_gt_bnx, 7])

        losses = \
            tf.map_fn(lambda gt_box:
                YOLOLoss._cal_cell_square_error(
                            gt_box = gt_box,
                            output = output,
                            cell_i = tf.cast(cell_ij, tf.int32) / feature_map_len,
                            cell_j = tf.cast(cell_ij, tf.int32) % feature_map_len,
                            feature_map_len = feature_map_len,
                            num_of_anchor_boxes = num_of_anchor_boxes,
                            num_of_classes = num_of_classes
                         )
            ,ground_truth)
        losses = tf.reduce_mean(losses)

        return losses

    @staticmethod
    def _cal_image_loss(output_and_gt, feature_map_len, num_of_gt_bnx,
                        num_of_anchor_boxes, num_of_classes):
        """ Same as calculate_loss(), except that now @output_and_gt only
            contains info for one image, and has shape
            (feature_map_len, feature_map_len, ?).

            The first "num_of_classes * (5+num_of_classes)" element in the last
            dimension are from output, and the rest are from ground_truth.

          Args:
            output_and_gt: Tensor of shape (feature_map_len, feature_map_len, ?)
            feature_map_len: Number of cells in a column/row of that image.
            num_of_gt_bnx: See FLAGS.num_of_gt_bnx.
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_classes: See FLAGS.num_of_classes.

          Return: Loss for this image.
        """

        # We can use
        #   for i in range(0, feature_map_len):
        #       for j in range(0, feature_map_len):
        #           tensor = output_and_gt[i][j]
        #           ...
        # to iterate into `output_and_gt' and manipulate every single tensor.
        # But this for loop is expensive when building the network, because
        # it run feature_map_len*feature_map_len time. If turned into a lambda,
        # it only run once when building the network. So, all we have to do is
        # to get (i,j) when using lambda. To do that, we pack
        # (i*feature_map_len+j) into the last dimension of `output_and_gt'.
        indexes = tf.reshape(
                        tf.cast(tf.constant(range(feature_map_len*feature_map_len)), tf.float32),
                        [feature_map_len, feature_map_len, 1]
                    )
        output_and_gt = tf.concat([indexes, output_and_gt], axis=-1)
        losses = \
            tf.map_fn(lambda xdim:
                tf.map_fn(lambda ts:
                    YOLOLoss._cal_square_error(
                        tensor=ts,
                        feature_map_len=feature_map_len,
                        num_of_gt_bnx=num_of_gt_bnx,
                        num_of_anchor_boxes=num_of_anchor_boxes,
                        num_of_classes=num_of_classes)
                ,xdim)
            ,output_and_gt)
#        _losses = []
#        for i in range(0, feature_map_len):
#            for j in range(0, feature_map_len):
#                ts = output_and_gt[i][j]
#                _losses.append(
#                    YOLOLoss._cal_square_error(
#                                tensor = ts,
#                                cell_i = i,
#                                cell_j = j,
#                                feature_map_len = feature_map_len,
#                                num_of_gt_bnx = num_of_gt_bnx,
#                                num_of_anchor_boxes = num_of_anchor_boxes,
#                                num_of_classes = num_of_classes))
#        losses = tf.stack(_losses)
        return tf.reduce_sum(losses)

    @staticmethod
    def _scale_box_relative_to_cell_center(box_and_incremental_nums,
                                           image_size,
                                           feature_map_len,
                                           num_of_anchor_boxes,
                                           num_of_classes):
        """ Calculate x,y,w,h relative to the center of the cell,
            such that:
              - the top left corner of the gt will not shift
                > 1*cellsize away from the cell center.
              - w and h is sigmoid-ed and calculate relative to image_size

          Args:
            box: A tensor of length num_of_anchor_boxes * (5 + num_of_classes)
                (which means we have to reshape it)
            feature_map_len: Number of cell in the one row/column of the image.
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_classes: See FLAGS.num_of_classes.
          Return:
            A tensor with all the coordinates scaled.
        """

        box = box_and_incremental_nums[0]
        incremental_nums = box_and_incremental_nums[1]

        # incremental_nums[0] should equals incremental_nums[1]
        num = incremental_nums[0]-1
        cell_i = tf.cast(num, tf.int32) % feature_map_len
        cell_j = tf.cast(num, tf.int32) / feature_map_len
        top_left_x = cell_i * 32
        top_left_y = cell_j * 32
        bottom_right_x = (cell_i + 1) * 32
        bottom_right_y = (cell_j + 1) * 32
        center_x = (top_left_x + bottom_right_x) / 2
        center_y = (top_left_y + bottom_right_y) / 2

        def _scale_box(bx, image_size, center_x, center_y):
            _x = bx[0]
            _y = bx[1]
            _w = bx[2]
            _h = bx[3]

            # TODO why should these cast
            x = tf.cast(center_x, tf.float32) + tf.sigmoid(_x) * 32
            y = tf.cast(center_y, tf.float32) + tf.sigmoid(_y) * 32
            w = tf.sigmoid(_w) * tf.cast(image_size, tf.float32)
            h = tf.sigmoid(_h) * tf.cast(image_size, tf.float32)

            p1 = tf.stack([x, y, w, h])
            p2 = bx[4:5+num_of_classes]
            return tf.concat([p1, p2], axis=0)
        #--------- END -----

        # tf.logging.info("##### Shape of box: %s" % str(box.shape))
        boxes = tf.reshape(box, [num_of_anchor_boxes, 5 + num_of_classes])
        boxes = \
            tf.map_fn(lambda bx:
                _scale_box(bx, image_size, center_x, center_y)
            ,boxes)
        boxes = tf.reshape(box, [-1])
        return boxes

    @staticmethod
    def calculate_loss(output, ground_truth, batch_size, num_of_anchor_boxes,
                       num_of_classes, num_of_gt_bnx):
        """ Calculate loss using the loss from yolov1 + yolov2.

          Args:
            output: Computed output from the network. In shape
                    [batch_size,
                        image_size/32,
                            image_size/32,
                                (num_of_anchor_boxes * (5 + num_of_classes))]
            ground_truth: Ground truth bounding boxes.
                          In shape [batch_size, num_of_gt_bnx, 7].
            batch_size: See FLAGS.batch_size.
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_classes: See FLAGS.num_of_classes.
            num_of_gt_bnx: See FLAGS.num_of_gt_bnx_fixed.

          Return:
              loss: The final loss (after tf.reduce_mean()).
        """

        # static assert
        assert output.shape[1] == output.shape[2], "Output tensor shape mismatch"

        # don't use tf.shape() here because it is dynamic. We are sure about its
        # output's shape, so we don't have to use that.
        feature_map_len = output.shape[1]
        image_size = feature_map_len*32

        # Flatten ground_truth to [batch_size, num_of_gt_bnx * 7] and stack the 1
        # dimension, making it
        #    [batch_size, image_size/32, image_size/32, num_of_gt_bnx * 7]
        # That way we can bundle it with `output`.
        def _stack_it(tensor, length):
            return tf.stack([tensor for i in range(length)])
        #-------------------
        ground_truth = tf.reshape(ground_truth, [batch_size, num_of_gt_bnx * 7])
        ground_truth = \
            tf.map_fn(lambda piece_in_batch:
                _stack_it(_stack_it(piece_in_batch, feature_map_len), feature_map_len)
            ,ground_truth)

        # Scale `output`
        #
        # lambda blackhole!
        all_zeros = tf.ones(tf.shape(output))
        incremental_nums = \
            tf.map_fn(lambda piece_in_batch:
                tf.map_fn(lambda xdim:
                    tf.map_fn(lambda ydim:
                        tf.scan(lambda box1, box2: # use tf.scan
                            tf.add(box1, box2)
                        ,ydim)
                    ,xdim)
                ,piece_in_batch)
            ,all_zeros)

        output_pack = tf.stack([output, incremental_nums], axis=3)

        # NOTE we have one fewer inner loop here. That is because we want to
        # take the [output, incremental_nums] bundle.
        output_scaled = \
            tf.map_fn(lambda piece_in_batch:
                tf.map_fn(lambda xdim:
                    tf.map_fn(lambda box_and_incremental_nums:
                        YOLOLoss._scale_box_relative_to_cell_center(
                                    box_and_incremental_nums,
                                    image_size,
                                    feature_map_len,
                                    num_of_anchor_boxes,
                                    num_of_classes
                                 )
                    ,xdim)
                ,piece_in_batch)
            ,output_pack)

        output_shape = tf.shape(output)
        output = tf.reshape(output_scaled, [tf.shape(output)[0], # dynamic shape
                                            output.shape[1],     # static shape
                                            output.shape[2],     # static shape
                                            num_of_anchor_boxes * (5 + num_of_classes)])
        # shape of output:
        #   [?, feature_map_len, feature_map_len, num_of_anchor_boxes * (5 + num_of_classes))
        # shape of ground_truth:
        #   [?, feature_map_len, feature_map_len, num_of_gt_bnx * 7]

        # concat ground_truth and output to make a tensor of shape 
        #   (?, feature_map_len, feature_map_len, ?)
        #
        # (Let's assume 30 == num_of_anchor_boxes * (5+num_of_classes) )
        # At the very end dimension, the 30 tensors from output starts first,
        # followed with those tensors from ground_truth.
        #
        # TODO check that why we have to cast
        output = tf.cast(output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        output_and_ground_truth = tf.concat([output, ground_truth], axis=3)

        losses = \
            tf.map_fn(lambda piece_output_and_gt:
                YOLOLoss._cal_image_loss(
                            output_and_gt = piece_output_and_gt,
                            feature_map_len = feature_map_len,
                            num_of_gt_bnx = num_of_gt_bnx,
                            num_of_anchor_boxes = num_of_anchor_boxes,
                            num_of_classes = num_of_classes)
            ,output_and_ground_truth)

        loss = tf.reduce_mean(losses)

        return loss

# image should be of shape [None, x, x, 3], where x should multiple of 32,
# starting from 320 to 608.
def YOLOvx(images, num_of_anchor_boxes=5, num_of_classes=1, reuse=tf.AUTO_REUSE,
           freeze_backbone=True):
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

def train(training_file_list, class_name_file, batch_size, num_of_classes,
          num_of_image_scales, image_size_min, num_of_anchor_boxes,
          num_of_gt_bnx_fixed, num_of_steps, check_point_file,
          freeze_backbone=True):
    """ Train the YOLOvx network.

      Args:
          training_file_list: See FLAGS.training_file_list.
          class_name_file: See FLAGS.class_name_file.
          batch_size: See FLAGS.batch_size.
          num_of_classes: See FLAGS.num_of_classes.
          num_of_image_scales: See FLAGS.num_of_image_scales.
          image_size_min: See FLAGS.image_size_min.
          num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes
          num_of_steps: See FLAGS.num_of_steps.
          check_point_file: See FLAGS.check_point_file.
          freeze_backbone: See FLAGS.freeze_backbone.
      Return:
          None.
    """

    # TODO we should not bundle `vars_to_restore' in io_info. A better way
    # will be to split apart the inception module and load its weights only once.
    io_info = namedtuple('io_info', ['size', 'input_tensor', 'output_tensor',
                                     'ground_truth', 'vars_to_restore', 
                                     'loss', 'train_step'])
    variable_size_io_map = []
    variable_sizes = []
    for i in range(num_of_image_scales):
        variable_sizes.append(image_size_min + i*32)

    tf.logging.info("Building tensorflow graph...")

    # with tf.device('/device:GPU:0'):
    for idx, size in enumerate(variable_sizes):
        assert size % 32 == 0, ("Image size should be multiple of 32. "
                                "FLAGS.image_size_min may be wrong.")
        _x = tf.placeholder(tf.float32, [None, size, size, 3])
        _y, _vars_to_restore = YOLOvx(_x)
        # 7: label, x, y, w, h, cell_i, cell_j
        _y_gt = tf.placeholder(tf.float32, [None, num_of_gt_bnx_fixed, 7])

        loss = YOLOLoss.calculate_loss(
                                _y,
                                _y_gt,
                                batch_size=batch_size,
                                num_of_anchor_boxes=num_of_anchor_boxes,
                                num_of_classes=num_of_classes,
                                num_of_gt_bnx=num_of_gt_bnx_fixed
                            )

        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        variable_size_io_map.append(io_info(size=size,
                                            input_tensor=_x,
                                            output_tensor=_y,
                                            ground_truth=_y_gt,
                                            vars_to_restore=_vars_to_restore,
                                            loss=loss,
                                            train_step=train_step))

        tf.logging.info("\rFinished building network loss/train_step: %d" % idx)

    tf.logging.info("All network loss/train_step built! Yah!")

    # load images and labels
    reader = DatasetReader(training_file_list, class_name_file, num_of_classes)

    initializer = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=True))

    # TODO If run here, will the checkpoint be loaded below?
    sess.run(initializer)

    io_map_idx = 0
    already_loaded_ckpt = False # TODO I hate this flag!
    for i in range(20000): #TODO FLAGS.num_of_steps
        # `size' is the size of input image, not the final feature map size
        size = variable_size_io_map[io_map_idx].size
        # _x = variable_size_io_map[io_map_idx].input_tensor
        # _y = variable_size_io_map[io_map_idx].output_tensor
        # _y_gt = variable_size_io_map[io_map_idx].ground_truth
        _vars_to_restore = variable_size_io_map[io_map_idx].vars_to_restore
        loss = variable_size_io_map[io_map_idx].loss
        train_step = variable_size_io_map[io_map_idx].train_step

        # load initial weights of the inception module
        if not already_loaded_ckpt:
            restorer = tf.train.Saver(_vars_to_restore)
            restorer.restore(sess, check_point_file)
            tf.logging.info("Checkpoint restored!")
            already_loaded_ckpt = True

        batch_xs, batch_ys = reader.next_batch(
                                batch_size=batch_size,
                                image_size=size,
                                num_of_anchor_boxes=num_of_anchor_boxes,
                                num_of_classes=num_of_classes,
                                num_of_gt_bnx=num_of_gt_bnx_fixed
                            )

        sys.stdout.write("Running train_step[%d]..." % i)
        sys.stdout.flush()
        # sess.run(initializer)
        _, loss_val = sess.run([train_step, loss],
                               feed_dict={_x: batch_xs, _y_gt: batch_ys})
        # loss_val = sess.run([loss],
                            # feed_dict={_x: batch_xs, _y_gt: batch_ys})

        tf.logging.info("LossVal: " + str(loss_val))

        if i % 100 == 0:
            io_map_idx += 1
            io_map_idx %= len(variable_size_io_map)

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
              num_of_gt_bnx_fixed = FLAGS.num_of_gt_bnx_fixed,
              num_of_steps        = FLAGS.num_of_steps,
              check_point_file    = FLAGS.check_point_file,
              freeze_backbone     = FLAGS.freeze_backbone)
    else:
        tf.logging.info("What do you want to do?")

if __name__ == '__main__':
    tf.app.run()
