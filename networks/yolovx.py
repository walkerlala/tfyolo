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
from backbone.inception_utils import inception_arg_scope
from backbone.vgg import vgg_16
from backbone.vgg import vgg_arg_scope
from backbone.resnet_v2 import resnet_v2_101
from backbone.resnet_v2 import resnet_arg_scope
from backbone.vx import vx
from backbone.vx import vx_arg_scope

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
    elif backbone_arch == 'vx':
        with slim.arg_scope(vx_arg_scope()):
            backbone_network, endpoints = vx(
                                            images,
                                            is_training=not freeze_backbone,
                                            reuse=reuse
                                        )
            vars_to_restore = slim.get_variables_to_restore()
    elif backbone_arch == 'vgg_16':
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
    elif backbone_arch == 'resnet_v2':
        with slim.arg_scope(resnet_arg_scope()):
            backbone_network, endpoints = resnet_v2_101(
                                            images,
                                            num_classes=None,
                                            is_training=not freeze_backbone,
                                            global_pool=False,
                                            spatial_squeeze=False,
                                            reuse=reuse)
        vars_to_restore = slim.get_variables_to_restore()
    else:
        raise ValueError("No backbone network {} available.".format(backbone_arch))

    return backbone_network, vars_to_restore

def shortcut_from_input(images):
    """ """
    with tf.variable_scope("shortcut_from_input", "shortcut_from_input", [images]):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=trunc_normal(0.01),
                            activation_fn=tf.nn.leaky_relu):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                out = slim.conv2d(images, 32, [3, 3], scope="first_conv2d")
                out = slim.max_pool2d(out, [3, 3], stride=2, scope="first_max_pool2d")
                out = slim.conv2d(out, 64, [3, 3], scope="second_conv2d")
                out = slim.max_pool2d(out, [3, 3], stride=4, scope="second_max_pool2d")
                out = slim.conv2d(out, 128, [3, 3], scope="third_conv2d")
                out = slim.max_pool2d(out, [3, 3], stride=4, scope="third_max_pool2d")
                return out

# image should be of shape [None, x, x, 3], where x should multiple of 32,
# starting from 320 to 608.
def YOLOvx(images, backbone_arch, num_anchor_boxes, num_classes=1,
           freeze_backbone=True, reuse=tf.AUTO_REUSE):
    """ This architecture of YOLO is not strictly the same as in those papers.
    We use some backbone networks as a starting point, and then add necessary
    layers on top of it. Therefore, we name it `YOLOvx (version x)`.

    Args:
        images: Input images, of shape [None, None, None, 3].
        backbone_arch: backbone network to use.
        num_anchor_boxes: See FLAGS.num_anchor_boxes.
        num_classes: see FLAGS.num_classes.
        reuse: Whether or not the network weights should be reused when
            building another YOLOvx in the same program.
        freeze_backbone: Whether or not to freeze the inception net backbone.

    Return:
        net: The final YOLOvx network.
        vars_to_restore: Reference to variables in the backbone network that
            should be restored from the checkpoint file.
    """

    backbone, vars_to_restore = backbone_network(images, backbone_arch, reuse=reuse)
    # shortcut = shortcut_from_input(images)

    with slim.arg_scope([slim.conv2d],
                        weights_initializer=trunc_normal(0.01),
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': True,
                            'decay':0.95,
                            'epsilon': 0.001,
                            'updates_collections': tf.GraphKeys.UPDATE_OPS,
                            'fused': None}):

        backbone = slim.conv2d(backbone, 1024, [3, 3], scope='backbone_refined_1')
        backbone = slim.conv2d(backbone, 1024, [3, 3], scope='backbone_refined_2')
        backbone = slim.conv2d(backbone, 64, [1, 1], scope='backbone_refined_1x1')
        backbone = slim.conv2d(backbone, 1024, [3, 3], scope='backbone_refined_3')
        backbone = slim.conv2d(backbone, 1024, [3, 3], scope='backbone_refined_4')
        # backbone = tf.concat([backbone, shortcut], axis=-1, name="backbone_with_shortcut")

        net = slim.conv2d(
                backbone,
                num_anchor_boxes * (5 + num_classes),
                [3, 3],
                scope="bboxes_full"
            )
    return net, vars_to_restore

class YOLOLoss():
    """ Provides methods for calculating loss """

    def __init__(self, batch_size, num_anchor_boxes, num_classes, num_gt_bnx,
                 global_step):
        """
          Args:
            (see their difinition in FLAGS)
            global_step: A tensor, used to switch between loss function
        """
        # NOTE, though we can know batch_size when building the network/loss,
        # but we are not using it anywhere when building then network/loss,
        # because, maybe we will have a variable-sized batch input, who know?
        self.__batch_size = batch_size
        self.num_anchor_boxes = num_anchor_boxes
        self.num_classes = num_classes
        self.num_gt_bnx = num_gt_bnx
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
        """ Same as `bbox_iou_corner_xy()', except that:

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

    @staticmethod
    def bbox_iou_center_xy_flat(bboxes1, bboxes2):
        """ Compute iou between bboxes1 and bboxes2 one by one.

        By "one by one", it means iou between bboxes1[0] and bboxes2[0], bboxes1[1]
        and bboxes2[1]...

          Args:
              bboxes1: of shape (-1, 4)
              bboxes2: of shape (-1, 4)
          Return:
              iou: of shape (-1,)
        """
        x11, y11, w11, h11 = tf.split(bboxes1, 4, axis=1)
        x21, y21, w21, h21 = tf.split(bboxes2, 4, axis=1)

        xi1 = tf.maximum(x11, x21)
        xi2 = tf.minimum(x11, x21)

        yi1 = tf.maximum(y11, y21)
        yi2 = tf.minimum(y11, y21)

        wi = w11/2.0 + w21/2.0
        hi = h11/2.0 + h21/2.0

        inter_area = tf.maximum(wi - (xi1 - xi2), 0) \
                      * tf.maximum(hi - (yi1 - yi2), 0)

        bboxes1_area = w11 * h11
        bboxes2_area = w21 * h21

        union = (bboxes1_area + bboxes2_area) - inter_area

        # some invalid boxes should have iou of 0 instead of NaN
        # If inter_area is 0, then this result will be 0; if inter_area is
        # not 0, then union is not too, therefore adding a epsilon is OK.
        return tf.squeeze(inter_area / (union+0.0001))

    def calculate_loss(self, output, ground_truth):
        """ Calculate loss using the loss from yolov1 + yolov2.

          Args:
            output: Computed output from the network. In shape
                [batch_size,
                    image_size/32,
                        image_size/32,
                            num_anchor_boxes * (5 + num_classes)]
            ground_truth: Ground truth bounding boxes. In shape
                [batch_size,
                    image_size/32,
                        image_size/32,
                            num_gt_bnx * 5]
          Return:
              loss: The final loss (after tf.reduce_mean()).
        """

        num_anchor_boxes = self.num_anchor_boxes
        num_classes = self.num_classes
        num_gt_bnx = self.num_gt_bnx

        # concat ground_truth and output to make a tensor of shape 
        #   (?, ?, ?, num_anchor_boxes*(5+num_classes) + 5*num_gt_bnx)
        output_and_ground_truth = tf.concat([output, ground_truth], axis=3)

        # flatten dimension
        op_and_gt_batch = \
                tf.reshape(
                    output_and_ground_truth,
                    [-1, num_anchor_boxes*(5+num_classes) + 5*num_gt_bnx]
                )

        split_num = num_anchor_boxes*(5+num_classes)
        op_boxes = tf.reshape(op_and_gt_batch[..., 0:split_num],
                              [-1, num_anchor_boxes, 5+num_classes])
        # op_boxes = tf.Print(op_boxes, [op_boxes], "!!!op_boxes: ", summarize=10000)
        gt_boxes = tf.reshape(op_and_gt_batch[..., split_num:],
                              [-1, num_gt_bnx, 5])
        # There are many some fake ground truth (i.e., [0,0,0,0,0]) in gt_boxes, 
        # but we can't naively remove it here. Instead, we use tf.boolean_mask()
        # to mask the relevant tensors out.

        # [-1, num_gt_bnx, num_anchor_boxes]
        ious = YOLOLoss.bbox_iou_center_xy_batch(
                                gt_boxes[:, :, 1:5],
                                op_boxes[:, :, 0:4]
                            )
        # [-1, num_gt_bnx, 1]
        values, idx = tf.nn.top_k(ious, k=1)
        # [-1, num_gt_bnx] (just squeeze the last dimension)
        idx = tf.reshape(idx, [-1, num_gt_bnx])

        # Get tuples of (gt, op) with the max iou (i.e., those who match)
        #
        # [-1, num_gt_bnx, num_anchor_boxes]
        one_hot_idx = tf.one_hot(idx, depth=num_anchor_boxes)
        full_idx = tf.where(tf.equal(1.0, one_hot_idx))
        gt_idx = full_idx[:, 0:2]
        op_idx = full_idx[:, 0:3:2]
        # [?, 5]
        gt_boxes_max = tf.gather_nd(gt_boxes, gt_idx)
        # [?, 5+num_classes]
        op_boxes_max = tf.gather_nd(op_boxes, op_idx)
        # [?, 5+num_classes + 5]
        # NOTE the order in which they are concatenated!
        iou_max_boxes_raw = tf.concat([op_boxes_max, gt_boxes_max], axis=1)
        # mask out fake gt_op pair
        nonzero_mask = tf.reduce_any(tf.not_equal(0.0, gt_boxes_max), axis=1)
        iou_max_boxes = tf.boolean_mask(iou_max_boxes_raw, nonzero_mask)
        iou_max_op = iou_max_boxes[..., 0:5+num_classes]
        iou_max_gt = iou_max_boxes[..., 5+num_classes: ]
        #iou_max_boxes = tf.Print(
        #        iou_max_boxes,
        #        [iou_max_boxes],
        #        "###iou_max_boxes: ",
        #        summarize=1000
        #    )

        # Compute the real iou one by one
        iou_flat = YOLOLoss.bbox_iou_center_xy_flat(
                    iou_max_op[..., 0:4],
                    iou_max_gt[..., 1:5]
                   )

        # Get op boxes which are never matched by any non-fake gt boxes.
        nonzero_mask = tf.cast(
                           tf.reduce_any(tf.not_equal(0.0, gt_boxes), axis=2),
                           tf.float32
                       )[..., None]
        filtered_one_hot = one_hot_idx * nonzero_mask
        # [-1, num_anchor_boxes]
        active_op = tf.sign(tf.reduce_sum(filtered_one_hot, axis=1))
        nonactive_op = 1 - active_op
        nonactive_op_idx = tf.where(tf.equal(1.0, nonactive_op))
        op_never_matched = tf.gather_nd(op_boxes, nonactive_op_idx)

        #coordinate loss
        cooridniates_se_wh = tf.square(
                        tf.sqrt(iou_max_op[:, 2:4])-tf.sqrt(iou_max_gt[:, 3:5])
                    )
        cooridniates_se_xy = tf.square(iou_max_op[:, 0:2]-iou_max_gt[:, 1:3])
        #cooridniates_se_xy = tf.Print(
        #        cooridniates_se_xy,
        #        [tf.shape(iou_max_op[:, 2:4]),
        #         tf.shape(iou_max_gt[:, 3:5]),
        #         tf.shape(iou_max_op[:, 0:2]),
        #         tf.shape(iou_max_gt[:, 1:3])],
        #        "Shape info: "
        #    )
        cooridniates_se = tf.concat([cooridniates_se_xy, cooridniates_se_wh], axis=-1)
        cooridniates_loss = tf.reduce_sum(cooridniates_se)

        # objectness loss
        # We minus the output confidence by 1 instead of the real iou, which is
        # used in the papers. Our method work well (presumably), but you are
        # free to use whichever you like.
        # objectness_se = tf.square(iou_max_op[:, 4] - iou_flat)
        objectness_se = tf.square(iou_max_op[:, 4] - 1)
        #objectness_se = tf.Print(
        #        objectness_se,
        #        [iou_max_op[:, 4]],
        #        "######objectness output: ",
        #        summarize=10000
        #    )
        objectness_loss = tf.reduce_sum(objectness_se)

        #classness loss (TODO)

        # nonobjectness loss
        nonobjectness_se = tf.square(op_never_matched[:, 4]-0)
        #nonobjectness_se = tf.Print(
        #        nonobjectness_se,
        #        [op_never_matched[:, 4]],
        #        "!!!!!!nonobjectness_se output: ",
        #        summarize=10000
        #    )
        nonobjectness_loss = tf.reduce_sum(nonobjectness_se)
        # nonobjectness_loss is too large and will overwhelm the whole network,
        # so we take its ln. Even that, most "nonobject boxes" have a objectness
        # lower than 0.5, so we think it is OK that we kind of "ignore" it.
        nonobjectness_loss = tf.log(nonobjectness_loss)
        #objectness_loss = tf.Print(
        #        objectness_loss,
        #        [cooridniates_loss, objectness_loss, nonobjectness_loss],
        #        "cl & ol & nol: "
        #    )

        tf.summary.scalar("cooridniates_loss", cooridniates_loss)
        tf.summary.scalar("objectness_loss", objectness_loss)
        tf.summary.scalar("nonobjectness_loss", nonobjectness_loss)
        loss = 0.5*cooridniates_loss + 1*objectness_loss + 1*nonobjectness_loss
        return loss

