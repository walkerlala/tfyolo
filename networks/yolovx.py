#coding:utf-8

"""
Defining the structure of YOLOvx.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import pdb
import tensorflow as tf
from backbone.inception_v1 import inception_v1
from backbone.vgg import vgg_16
from backbone.vgg import vgg_arg_scope
from backbone.inception_utils import inception_arg_scope

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

FLAGS = tf.app.flags.FLAGS

def backbone_network(images, backbone_arch, reuse):

    freeze_backbone = FLAGS.freeze_backbone

    if backbone_arch == 'inception_v1':
        with slim.arg_scope(inception_arg_scope()):
            backbone_network, endpoints = inception_v1(
                                            images,
                                            num_classes=None,
                                            is_training=not freeze_backbone,
                                            global_pool=False,
                                            reuse=reuse
                                        )
        vars_to_restore = slim.get_variables_to_restore()

    else:
        with slim.arg_scope(vgg_arg_scope()):
            backbone_network, endpoints = vgg_16(
                                           images,
                                           num_classes=None,
                                           is_training=not freeze_backbone,
                                           global_pool=False,
                                           fc_conv_padding='SAME',
                                           spatial_squeeze=False,
                                           reuse=reuse)
        vars_to_restore = slim.get_variables_to_restore()
        # vgg_16_vars_to_restore = list(set(vgg_16_vars_to_restore)-set(inceptioin_vars_to_restore))

    return backbone_network, vars_to_restore

# image should be of shape [None, x, x, 3], where x should multiple of 32,
# starting from 320 to 608.
def YOLOvx(images, backbone_arch, num_of_anchor_boxes, num_of_classes,
           freeze_backbone, reuse=tf.AUTO_REUSE):
    """ This architecture of YOLO is not strictly the same as in those paper.
    We use inceptionv1 as a starting point, and then add necessary layers on
    top of it. Therefore, we name it `YOLO (version x)`.

    Args:
        images: Input images, of shape [None, None, None, 3].
        backbone_arch:
        num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
        num_of_classes: see FLAGS.num_of_classes.
        reuse: Whether or not the network weights should be reused when
            building another YOLOvx in the same program.
        freeze_backbone: Whether or not to freeze the inception net backbone.

    Return:
        net: The final YOLOvx network.
        vars_to_restore: Reference to variables (of the inception net backbone)
            that should be restored from the checkpoint file.
    """

    backbone, vars_to_restore = backbone_network(images, backbone_arch, reuse=reuse)

    with slim.arg_scope([slim.conv2d],
                        weights_initializer=trunc_normal(0.01),
                        activation_fn=tf.nn.sigmoid,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': True,
                            'decay':0.95,
                            'epsilon': 0.001,
                            'updates_collections': tf.GraphKeys.UPDATE_OPS,
                            'fused': None}):

        backbone = slim.conv2d(backbone, 2048, [2,2], scope='backbone_refined')

        # Follow the number of output in YOLOv3
        # Use sigmoid the constraint output to [0, 1]
        net = slim.conv2d(
                backbone,
                num_of_anchor_boxes * (5 + num_of_classes),
                [1, 1],
                scope="Final_output"
            )
    return net, vars_to_restore

#    with slim.arg_scope([slim.conv2d],
#                        weights_initializer=trunc_normal(0.01),
#                        normalizer_fn=slim.batch_norm,
#                        normalizer_params={
#                            'is_training': True,
#                            'decay':0.95,
#                            'epsilon': 0.001,
#                            'updates_collections': tf.GraphKeys.UPDATE_OPS,
#                            'fused': None}):
#        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
#                             stride=1, padding='SAME'):
#            with tf.variable_scope("extra_inception_module_0", reuse=reuse):
#                with tf.variable_scope("Branch_0"):
#                    branch_0 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
#                with tf.variable_scope("Branch_1"):
#                    branch_1 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
#                    branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Conv2d_0b_3x3')
#                with tf.variable_scope("Branch_2"):
#                    branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
#                    branch_2 = slim.conv2d(branch_2, 32, [5, 5], scope='Conv2d_0b_3x3')
#                with tf.variable_scope("Branch_3"):
#                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
#                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
#
#                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
#
#                # `inception_net' has 1024 channels and `net' has 160 channels.
#                #
#                # TODO: in the original paper, it is not directly added together.
#                # Instead, `inception_net' should be at least x times the size
#                # of `net' such that we can "pool concat" them.
#                net = tf.concat(axis=3, values=[inception_net, net])
#
#                # Follow the number of output in YOLOv3
#                # Use sigmoid the constraint output to [0, 1]
#                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.sigmoid):
#                    net = slim.conv2d(
#                            net,
#                            num_of_anchor_boxes * (5 + num_of_classes),
#                            [1, 1],
#                            scope="Final_output"
#                        )

    # tf.summary.histogram("all_weights", tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS))
    # tf.summary.histogram("all_bias", tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES))
    # tf.summary.histogram("all_activations", tf.contrib.layers.summarize_collection(tf.GraphKeys.ACTIVATIONS))
    # # tf.summary.histogram("all_variables", tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    # tf.summary.histogram("all_global_step", tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_STEP))

    # NOTE all the values output in `net` are relative to the cell size, not the
    # whole image
    # return net, vars_to_restore

class YOLOLoss():
    """ Provides methods for calculating loss """

    def __init__(self, batch_size, num_of_anchor_boxes, num_of_classes,
                 num_of_gt_bnx_per_cell, global_step):
        """
          Args:
            (see their difinition in FLAGS)
            global_step: A tensor, used to switch between loss function
        """
        # NOTE, though we can know batch_size when building the network/loss,
        # but we are not using it anywhere when building then network/loss,
        # because, maybe we will have a variable-sized batch input, who know?
        self.__batch_size = batch_size
        self.num_of_anchor_boxes = num_of_anchor_boxes
        self.num_of_classes = num_of_classes
        self.num_of_gt_bnx_per_cell = num_of_gt_bnx_per_cell
        self.global_step = global_step

    @staticmethod
    def sigmoid(x):
        return math.exp(-np.logaddexp(0, -x))
        # return 1 / (1 + math.exp(-x))

    @staticmethod
    def bbox_iou_corner_xy(bboxes1, bboxes2):
        """
        Args:
          bboxes1: shape (total_bboxes1, 4)
              with x1, y1, x2, y2 point order.
          bboxes2: shape (total_bboxes2, 4)
              with x1, y1, x2, y2 point order.

            p1 *-----
               |     |
               |_____* p2

        Returns:
            Tensor with shape (total_bboxes1, total_bboxes2)
            with the IoU (intersection over union) of bboxes1[i] and
            bboxes2[j] in [i, j].
        """

        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=2)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=2)

        xI1 = tf.maximum(x11, tf.transpose(x21))
        xI2 = tf.minimum(x12, tf.transpose(x22))

        yI1 = tf.minimum(y11, tf.transpose(y21))
        yI2 = tf.maximum(y12, tf.transpose(y22))

        inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI1 - yI2), 0)

        bboxes1_area = (x12 - x11) * (y11 - y12)
        bboxes2_area = (x22 - x21) * (y21 - y22)

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

        return inter_area / (union+0.0001)

    @staticmethod
    def bbox_iou_center_xy_batch(bboxes1, bboxes2):
        """ Same as `bbox_overlap_iou_v1', except that:

          1. it use center_x, center_y, w, h instead of x1, y1, x2, y2.
          2. it operate on a batch rather than a piece of batch, and it
             retains the batch structure (i.e., it operate piece-wise
             level between batches)! This is IMPORTANT. It means that
             we will do bbox_iou_center_xy_single() with piece 1 of
             bboxes1 against piece 1 of bboxes2, and piece 2 of bboxes1
             against piece 2 of bboxes2, but never piece 1 of bboxes1
             against piece 2 of bboxes2.

          Args:
            bboxes1: [batch_size, d21, 4].
            bboxes2: [batch_size, d22, 4].

          Return:
             [batch_size, d21, d22, 1]
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

    def concat_tranpose_broadcast_batch(self, gt_boxes_padded, op_boxes):
        """ A concat operation with broadcasting semantic. NOTE that it also
            operates on batch level and retains the batch structure, as
            described in `bbox_iou_center_xy_batch()` above.

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
        """ Concat gt_boxes and op_boxes with broadcasting semantic, as shown
            in `concat_tranpose_broadcast_batch()', and finally concat the
            corresponding `ious` **at the beginning**.

          Args:
           gt_boxes: [-1, num_of_gt_bnx_per_cell, 5]
           op_boxes: [-1, num_of_anchor_boxes, 5+num_of_classes]

          Return:
            bundle: [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes, 3, 5+num_of_classes],
                    with
                    [:, :, :, 0, 0] the iou, and [:, :, :, 0, :] padded to 5+num_of_classes
                    [:, :, :, 1, :] the ground_truth, padded to 5+num_of_classes
                    [:, :, :, 2, :] the output
        """
        num_of_anchor_boxes = self.num_of_anchor_boxes
        num_of_classes = self.num_of_classes
        num_of_gt_bnx_per_cell = self.num_of_gt_bnx_per_cell

        ious = YOLOLoss.bbox_iou_center_xy_batch(
                            gt_boxes[:, :, 1:5],
                            op_boxes[:, :, 0:4]
                          )
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
        op_gt_bundle = self.concat_tranpose_broadcast_batch(
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
        # Note that to make it "no harm", a real gt box should contain coordinates
        # < 1 such that tf.ceil() return 1.

        # [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes]
        ious = YOLOLoss.bbox_iou_center_xy_batch(
                                gt_boxes[:, :, 1:5],
                                op_boxes[:, :, 0:4]
                            )
        # [-1, num_of_gt_bnx_per_cell, 1]
        values, idx = tf.nn.top_k(ious, k=1)
        # [-1, num_of_gt_bnx_per_cell] (just squeeze the last dimension)
        idx = tf.reshape(idx, [-1, num_of_gt_bnx_per_cell])

        # Get tuples of (gt, op) with the max iou (i.e., those who match)
        # [-1, num_of_gt_bnx_per_cell, num_of_anchor_boxes]
        one_hot_idx = tf.one_hot(idx, depth=num_of_anchor_boxes)
        full_idx = tf.where(tf.equal(1.0, one_hot_idx))
        gt_idx = full_idx[:, 0:2]
        op_idx = full_idx[:, 0:3:2]
        # [?, 5]
        gt_boxes_max = tf.gather_nd(gt_boxes, gt_idx)
        # [?, 5+num_of_classes]
        op_boxes_max = tf.gather_nd(op_boxes, op_idx)
        # [?, 5+num_of_classes + 5]
        # NOTE the order in which they are concatenated!
        iou_max_boxes = tf.concat([op_boxes_max, gt_boxes_max], axis=1)

        # Get those op boxes which were never matched by any gt box
        mask = tf.reduce_max(one_hot_idx, axis=-2)
        op_never_max_indices = tf.where(tf.equal(0.0, mask))
        never_max_op_boxes = tf.gather_nd(op_boxes, op_never_max_indices)

        #coordinate loss
        _5_nc = 5+num_of_classes
        cooridniates_se_wh = tf.sqrt(
            tf.pow(iou_max_boxes[:, 2:4]-iou_max_boxes[:, _5_nc+3:_5_nc+5], 2)+0.0001
        )
        cooridniates_se_xy = tf.pow(iou_max_boxes[:, 0:2]-iou_max_boxes[:, _5_nc+1:_5_nc+3], 2)
        cooridniates_se = tf.concat([cooridniates_se_xy, cooridniates_se_wh], axis=-1)
        cooridniates_se_ceil = tf.ceil(iou_max_boxes[:, _5_nc+1:])
        cooridniates_se_real_gt = cooridniates_se * cooridniates_se_ceil
        cooridniates_loss = tf.reduce_sum(cooridniates_se_real_gt)

        # objectness loss
        #
        objectness_se = tf.pow(iou_max_boxes[:, 4]-1, 2)
        objectness_se_ceil = tf.ceil(iou_max_boxes[:, _5_nc+2])
        objectness_se_real_gt = objectness_se * objectness_se_ceil
        objectness_loss = tf.reduce_sum(objectness_se_real_gt)
        # usually all fake gt boxes will consume a single anchor boxes which
        # should contain non-object. So here we calculate the non-objectness
        # loss for that mis-included-should-be-not-matched boxes.
        non_objectness_se = tf.pow(iou_max_boxes[:, 4]-0, 2)
        non_objectness_se_ceil = 1 - objectness_se_ceil # select those boxes
        non_objectness_se_real_gt = non_objectness_se * non_objectness_se_ceil
        non_objectness_loss = tf.reduce_sum(non_objectness_se_real_gt)
        non_objectness_loss = non_objectness_loss / num_of_gt_bnx_per_cell

        #classness loss (TODO)

        # NOTE usually never_max_op_boxes should be empty
        never_max_loss = tf.reduce_sum(tf.pow(never_max_op_boxes[:, 4]-0, 2))
        # assert_op = tf.Assert(never_max_loss==0.0, [never_max_op_boxes])

        # with tf.control_dependencies([assert_op]):
        all_loss = (cooridniates_loss +
                    5*objectness_loss +
                    1*non_objectness_loss +
                    never_max_loss)
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

