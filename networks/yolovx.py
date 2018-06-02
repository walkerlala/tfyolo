#coding:utf-8

"""
Defining the structure of YOLOvx.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import math
import pdb
import tensorflow as tf
from tensorflow.python.framework import ops
from backbone.inception_v1 import inception_v1
from backbone.vgg import vgg_16
from backbone.vgg import vgg_arg_scope
from backbone.inception_utils import inception_arg_scope

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
random_normal = lambda stddev: tf.random_normal_initializer(0.0, stddev)

FLAGS = tf.app.flags.FLAGS

def sigmoid_0(x):
    """When x==0, sigmoid_0(x) == 0 """
    return math.exp(-np.logaddexp(0, -x)) - 0.5
    # return 1 / (1 + math.exp(-x))

def fit_anchor_boxes(op_boxes, num_of_anchor_boxes):
    """ All anchor boxes will be located like

        +----------+     +-----------+
        |  .    .  |     |  .      . |
        |    .     |  or |  .      . |
        |  .    .  |     |  .      . |
        +----------+     +-----------+
          5 anchors        6 anchors

        All anchor boxes of the same shape.

        op_boxes of shape:
            [num_of_anchor_boxes, 5+num_of_classes]
    """
    # op_boxes = copy.deepcopy(_op_boxes)
    def is_even(x):
        return x%2 == 0
    def get_coord(idx, num):
        """Get x/y coordinate of anchor"""
        # idx start from 0
        if not is_even(num) and idx == int(num/2.0):
            return 0.5
        if not is_even(num):
            num -= 1
        if idx > int(num/2.0):
            idx -= 1
        piece_len = 1.0 / num
        return piece_len * idx + piece_len/2.0

    num = int(math.sqrt(num_of_anchor_boxes))
    result_opboxes = []
    for idx, obox in enumerate(op_boxes):
        if not is_even(num_of_anchor_boxes) and idx == int(num_of_anchor_boxes/2)+1:
            xcoord = 0.5
            ycoord = 0.5
        else:
            xidx = idx%num # start from 0
            yidx = idx/num # start from 0
            xcoord = get_coord(xidx, num)
            ycoord = get_coord(yidx, num)

        fit_xcoord = min(sigmoid_0(obox[0]) + xcoord, 0.999)
        obox[0] = fit_xcoord
        fit_ycoord = min(sigmoid_0(obox[1]) + ycoord, 0.999)
        obox[1] = fit_ycoord
        #TODO w/h fixed
        fit_w = max(min(sigmoid_0(np.exp(obox[2])), 0.999), 0.01)
        obox[2] = fit_w
        fit_h = max(min(sigmoid_0(np.exp(obox[3])), 0.999), 0.01)
        obox[3] = fit_h
        # obox[4] = sigmoid_0(obox[4])
        obox[4] = obox[4]
        # print("objectness is: " + str(obox[4]))
        #TODO for classes
        result_opboxes.append(obox)
    return np.array(result_opboxes)

def py_fit_anchor_boxes(net, num_of_anchor_boxes, num_of_classes):
    """Translate output of the network with predefined anchor boxes """

    def _py_func_identity_grad(op, grad):
        # grad = tf.Print(grad, [grad], "+++grad: ", summarize=1000)
        # return tf.ones_like(op.inputs[0]) * grad
        # return op.inputs[0] * grad
        return grad

    def fit_anchor_boxes_all(net):
        original_shape = np.shape(net)
        net = np.reshape(net, [-1, num_of_anchor_boxes, 5+num_of_classes])
        for idx, op_boxes in enumerate(net):
            net[idx] = fit_anchor_boxes(op_boxes, num_of_anchor_boxes)
        return np.reshape(net, original_shape)

    with tf.name_scope("pyfunc_fit_anchor", "net_fit_anchor", [net]):
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
        tf.RegisterGradient(rnd_name)(_py_func_identity_grad)
        default_graph = tf.get_default_graph()
        with default_graph.gradient_override_map({"PyFunc": rnd_name}):
            net = tf.py_func(fit_anchor_boxes_all, [net], [tf.float32], stateful=True)
    return net[0]

def backbone_network(images, backbone_arch, reuse):
    """Which backbone network to use"""

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

    return backbone_network, vars_to_restore

def shortcut_network(images):
    """Create a shortcut directly from input.  The input was scaled down 32 times."""

    def inception_unit(net, scale=2, count=0):
        if not hasattr(shortcut_network, "_count"):
            shortcut_network._count = 0
        shortcut_network._count += 1
        with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d],
                stride=1, padding='SAME'):
            with slim.arg_scope(
                    [slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_initializer=trunc_normal(0.01),
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={
                        'is_training': True,
                        'decay':0.95,
                        'epsilon': 0.001,
                        'updates_collections': tf.GraphKeys.UPDATE_OPS,
                        'fused': None}):
                with tf.variable_scope("shortcut_from_images_{}".format(shortcut_network._count)):
                    net = slim.conv2d(net, 16, [3,3], scope='conv2d_0')
                    net = slim.conv2d(net, 16, [3,3], stride=scale, scope='scope_2')
                    return net

                    with tf.variable_scope("Branch_0"):
                        branch_0 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope("Branch_1"):
                        branch_1 = slim.conv2d(net, 16, [1, 1], stride=1, scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Conv2d_0b_3x3')
                    # with tf.variable_scope("Branch_2"):
                        # branch_2 = slim.conv2d(net, 16, [1, 1], stride=1, scope='Conv2d_0a_1x1')
                        # branch_2 = slim.conv2d(branch_2, 32, [5, 5], scope='Conv2d_0b_3x3')
                    with tf.variable_scope("Branch_3"):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    net = tf.concat(
                            axis=3, values=[branch_0,
                                            branch_1,
                                            #branch_2,
                                            branch_3])
        return net
    # -- END --
    net = inception_unit(images, scale=4)
    net = inception_unit(net, scale=4)
    net = inception_unit(net, scale=2)
    return net

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
    shortcut_from_input = shortcut_network(images)

    with slim.arg_scope([slim.conv2d],
                        stride=1, padding='SAME',
                        weights_initializer=trunc_normal(0.01),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': True,
                            'decay':0.95,
                            'epsilon': 0.001,
                            'updates_collections': tf.GraphKeys.UPDATE_OPS,
                            'fused': None}):

        backbone = slim.conv2d(backbone, 1024, [3,3], scope='backbone_refined_0')
        backbone = slim.conv2d(backbone, 1024, [3,3], scope='backbone_refined_1')
        # backbone = slim.conv2d(backbone, 2048, [3,3], scope='backbone_refined_1')
        #backbone = tf.concat(
        #            axis=3,
        #            values=[backbone, shortcut_from_input],
        #            name="backbone_shortcut")

        with slim.arg_scope([slim.conv2d], weights_initializer=random_normal(0.1)):
            net = slim.conv2d(
                    backbone,
                    num_of_anchor_boxes * (5 + num_of_classes),
                    [1, 1],
                    scope="final_output"
                )

    net = py_fit_anchor_boxes(net, num_of_anchor_boxes, num_of_classes)
    return net, vars_to_restore

    # tf.summary.histogram("all_weights", tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS))
    # tf.summary.histogram("all_bias", tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES))
    # tf.summary.histogram("all_activations", tf.contrib.layers.summarize_collection(tf.GraphKeys.ACTIVATIONS))
    # # tf.summary.histogram("all_variables", tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    # tf.summary.histogram("all_global_step", tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_STEP))

    # NOTE all the values output in `net` are relative to the cell size, not the
    # whole image
    # return net, vars_to_restore

def cal_iou(coor1, coor2):
    """Calculate iou of two boxes coor1 & coor2,
       which contain [x, y, w, h]
    """
    assert len(coor1)==4 and len(coor2)==4, \
            "len(coor2/coor2) != 4"
    w2 = (coor1[2]+coor2[2])/2.0
    max_x = max(coor1[0], coor2[0])
    min_x = min(coor1[0], coor2[0])
    h2 = (coor1[3]+coor2[3])/2.0
    max_y = max(coor1[1], coor2[1])
    min_y = min(coor1[1], coor2[1])
    x_distance = max(0, w2-(max_x-min_x))
    y_distance = max(0, h2-(max_y-min_y))
    inner_area = x_distance * y_distance

    box1_area = coor1[2] * coor1[3]
    box2_area = coor2[2] * coor2[3]
    union = box1_area + box2_area

    return inner_area / (union+0.001)

def max_iou_with_op_boxes(gbox, op_boxes):
    """Get idx of the obox which has the max iou with @gbox"""
    gbox_coord = gbox[1:]
    max_iou = 0.0
    max_iou_idx = 0
    for idx, obox in enumerate(op_boxes):
        obox_coord = obox[0:4]
        iou = cal_iou(gbox_coord, obox_coord)
        if iou >= max_iou:
            max_iou_idx = idx
    return max_iou_idx

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

    # use tf.py_func to iterate into it
    def py_cal_loss(self, ts):
        """ ts of shape
            [-1, num_of_anchor_boxes*(5+num_of_classes) + 5*num_of_gt_bnx_per_cell]
        """
        num_of_anchor_boxes = self.num_of_anchor_boxes
        num_of_classes = self.num_of_classes
        num_of_gt_bnx_per_cell = self.num_of_gt_bnx_per_cell

        def py_cal_loss_onecell(cell):
            """ cell of shape
                [num_of_anchor_boxes*(5+num_of_classes) + 5*num_of_gt_bnx_per_cell]
            """
            split_num = num_of_anchor_boxes*(5+num_of_classes)
            op_boxes = np.reshape(
                           cell[0:split_num],
                           [num_of_anchor_boxes, 5+num_of_classes]
                       )
            gt_boxes = np.reshape(
                           cell[split_num:],
                           [num_of_gt_bnx_per_cell, 5]
                       )
            max_oboxes = set()
            gt_op_matched_pairs = []
            for g_idx, gbox in enumerate(gt_boxes):
                if not np.any(gbox): #fake gt box
                    continue
                o_idx = max_iou_with_op_boxes(gbox, op_boxes)
                max_oboxes.add(o_idx)
                gt_op_matched_pairs.append( (g_idx, o_idx) )
            # calculate coordinate loss & objectness confidence loss
            coor_loss = 0.0
            objectness_loss = 0.0
            for tp in gt_op_matched_pairs:
                g_idx = tp[0]
                o_idx = tp[1]
                gbox_coord = gt_boxes[g_idx][1:]
                obox_coord = op_boxes[o_idx][0:4]
                coor_loss += (math.pow(gbox_coord[0] - obox_coord[0], 2) +
                              math.pow(gbox_coord[1] - obox_coord[1], 2) +
                              math.pow(math.sqrt(gbox_coord[2]) - math.sqrt(obox_coord[2]), 2) +
                              math.pow(math.sqrt(gbox_coord[3]) - math.sqrt(obox_coord[3]), 2))
                obox_obj = op_boxes[o_idx][4]
                # print("###########################obox_obj before cal loss: " + str(obox_obj))
                # iou = cal_iou(gbox_coord, obox_coord)
                # objectness_loss += math.pow(obox_obj-iou, 2)
                objectness_loss += math.pow(obox_obj - 1, 2)

            # calculate noobjectness confidence loss
            noobjectness_loss = 0.0
            all_op_idx = set(range(len(op_boxes)))
            for o_idx in all_op_idx-max_oboxes:
                obox_obj = op_boxes[o_idx][4]
                noobjectness_loss += math.pow(obox_obj, 2)

            # calculate classness loss (TODO)

            loss = 0.5 * coor_loss + 10 * objectness_loss + 0.5 * noobjectness_loss
            return np.float32(loss)
        # ----- END DEF py_cal_loss_onecell ----

        total_loss = np.float32(0.0)
        # Docs say `ts' is not guaranteed to be a copy,
        # so we make a deepcopy here.
        # ts = copy.deepcopy(ts)
        for cell in ts:
            total_loss += py_cal_loss_onecell(cell)
        return total_loss

    @staticmethod
    def _py_func_identity_grad(op, grad):
        # we are passing gradient from the last output (the loss) of this
        # network, which is 1. (Uncomment the line above to see the output)
        # return tf.ones_like(op.inputs[0]) * grad
        return op.inputs[0] * grad

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
              loss: The final loss.
        """

        num_of_anchor_boxes = self.num_of_anchor_boxes
        num_of_classes = self.num_of_classes
        num_of_gt_bnx_per_cell = self.num_of_gt_bnx_per_cell

        # concat ground_truth and output to make a tensor of shape 
        #   (?, ?, ?, num_of_anchor_boxes*(5+num_of_classes) + 5*num_of_gt_bnx_per_cell)
        output_and_ground_truth = tf.concat([output, ground_truth], axis=3,
                                            name="output_and_ground_truth_concat")
        # flatten dimension
        op_and_gt_batch = \
            tf.reshape(
                output_and_ground_truth,
                [-1, num_of_anchor_boxes*(5+num_of_classes) + 5*num_of_gt_bnx_per_cell]
            )
        with tf.name_scope("pyfunc", "MyLoss", [op_and_gt_batch]):
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
            tf.RegisterGradient(rnd_name)(YOLOLoss._py_func_identity_grad)
            default_graph = tf.get_default_graph()
            with default_graph.gradient_override_map({"PyFunc": rnd_name}):
                loss_out = tf.py_func(
                               self.py_cal_loss,
                               [op_and_gt_batch],
                               [tf.float32],
                               stateful=True,
                               name="pyfunction"
                            )
        return loss_out[0]
