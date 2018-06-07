#!/usr/bin/python
#coding:utf-8

import datetime
import os
import pdb
import sys
import cv2
import tensorflow as tf
from utils.dataset import DatasetReader
from utils.dataset import ImageHandler
from utils.metrics import map_batch
from networks.yolovx import YOLOvx
from networks.yolovx import YOLOLoss
from networks.yolovx import DotcountLoss
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

tf.app.flags.DEFINE_boolean("train", False, "To train or not to train.")
tf.app.flags.DEFINE_boolean("dotcount", False, "To train the Dotcount model.")
tf.app.flags.DEFINE_boolean("test", False, "To test or not to test.")
tf.app.flags.DEFINE_boolean("multiple_images", False, "Predict for multiple images.")

tf.app.flags.DEFINE_string("infile", "Image.jpg", "The image to predict.")
tf.app.flags.DEFINE_alias("test_files_list", "infile")
tf.app.flags.DEFINE_string("outfile", "Prediction.jpg", "Output path of the predictions.")
tf.app.flags.DEFINE_alias("outdir", "outfile")

tf.app.flags.DEFINE_boolean("freeze_backbone", False,
        "Freeze the backbone network or not")

# TODO mark some args as co-exist, and backbone_arch should have limited choice
tf.app.flags.DEFINE_boolean("use_checkpoint", True, "To use or not to use checkpoint.")
tf.app.flags.DEFINE_string("checkpoint", "./vgg_16.ckpt",
        "Path of checkpoint file. Must come with its parent dir name, "
        "even it is in the current directory (eg, ./model.ckpt).")
tf.app.flags.DEFINE_boolean("restore_all_variables", False,
        "Whether or not to restore all variables. Default to False, which "
        "means restore only variables for the backbone network")
tf.app.flags.DEFINE_string("backbone_arch", "vgg_16",
        "Avaliable backbone architecture are 'vgg_16' and 'inception_v1'")
tf.app.flags.DEFINE_alias("backbone", "backbone_arch")

tf.app.flags.DEFINE_string("train_ckpt_dir", "/disk1/yolockpts/",
        "Path to save checkpoints")
tf.app.flags.DEFINE_string("train_log_dir", "/disk1/yolotraining/",
        "Path to save tfevent (for tensorboard)")

tf.app.flags.DEFINE_float("starter_learning_rate", 1e-2, "Starter learning rate")

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size.")

tf.app.flags.DEFINE_integer("num_of_classes", 1, "Number of classes.")

tf.app.flags.DEFINE_float("infer_threshold", 0.6, "Objectness threshold")

# NOTE We don't do any clustering for this. Just use 5 as a heuristic.
tf.app.flags.DEFINE_integer("num_of_anchor_boxes", 5,
        "Number of anchor boxes.")

tf.app.flags.DEFINE_integer("summary_steps", 100, "Write summary ever X steps")

# NOTE
#   1. Lable files should be put in the same directory and in the YOLO format
#   2. Empty line and lines that start with # will be ignore.
#      (But # at the end will not. Careful!)
tf.app.flags.DEFINE_string("train_files_list",
        "/disk1/labeled/roomonly_train.txt",
        "File which contains all images for training.")
tf.app.flags.DEFINE_string("eval_files_list",
        "/disk1/labeled/roomonly_valid.txt",
        "File which contains all images for evaluation.")

# Format of this file should be:
#
#   0 person
#   1 car
#   2 xxx
#   ...
#
# Empty line and lines that start with # will be ignore.
# (But # at the end will not. Careful!)
# TODO we are now ignoring class information (predict only for persons)
tf.app.flags.DEFINE_string("class_name_file", "/disk1/labeled/classnames.txt",
        "File which contains id <=> classname mapping.")

tf.app.flags.DEFINE_integer("image_size_min", 320,
        "The minimum size of a image (i.e., image_size_min * image_size_min).")

tf.app.flags.DEFINE_integer("num_of_image_scales", 10,
        "Number of scales used to preprocess images. We want different size of "
        "input to our network to bring up its generality.")

# It is pure pain to deal with tensor of variable length tensors (try and screw
# up your life ;-). So we pack each cell with a fixed number of ground truth
# bounding box (even though there is not that many ground truth bounding box
# at that cell). See code at "utils/dataset.py"
tf.app.flags.DEFINE_integer("num_of_gt_bnx_per_cell", 20,
        "Numer of ground truth bounding box per feature map cell. "
        "If there are not enough ground truth bouding boxes, "
        "some number of fake boxes will be padded.")

tf.app.flags.DEFINE_integer("num_of_steps", 20000,
        "Max num of step. -1 makes it infinite.")

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

def scale_output(output, outscale, num_of_anchor_boxes, num_of_classes=None,
                 only_xy=False):
    """ Scale x/y coordinates to be relative to the whole image.

        Args:
          output: YOLOvx network output, shape
            [None, None, None, num_of_anchor_boxes*X],
            where X is "(5+num_of_classes)" or "3", depending on whether
            @only_xy is False or True.
          outscale: [None, None, 3], the first dimension is against the
            second dimension of @output, the second dimension is against
            the third dimension of @output (that is, we do it across the
            whole batch).
          num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
          num_of_classes: See FLAGS.num_of_classes.
          only_xy: Whether @output only include x/y coordinates.

        Return:
          output: the scaled output.
    """
    xy_pad_num = 1 if only_xy else 3+num_of_classes

    # [None, None, 2]
    outscale_add_part = outscale[..., 0:2]
    # [None, None, 1]
    outscale_div_part = outscale[..., 2:3]
    outscale_div_part = tf.tile(outscale_div_part, [1,1,2])

    outscale_add_shape = tf.shape(outscale_add_part)
    outscale_add_pad = tf.zeros(
               [outscale_add_shape[0], outscale_add_shape[1], xy_pad_num],
               dtype=tf.float32
            )
    # [None, None, 5+num_of_classes]
    outscale_add = tf.concat([outscale_add_part, outscale_add_pad], axis=-1)
    # [None, None, 1, 5+num_of_classes]
    outscale_add = tf.expand_dims(outscale_add, axis=-2)
    # [None, None, num_of_anchor_boxes, 5+num_of_classes]
    outscale_add = tf.tile(outscale_add, [1,1,num_of_anchor_boxes,1])

    outscale_div_shape = tf.shape(outscale_div_part)
    outscale_div_pad = tf.ones(
               [outscale_div_shape[0], outscale_div_shape[1], xy_pad_num],
               dtype=tf.float32
            )
    outscale_div = tf.concat([outscale_div_part, outscale_div_pad], axis=-1)
    outscale_div = tf.expand_dims(outscale_div, axis=-2)
    outscale_div = tf.tile(outscale_div, [1,1,num_of_anchor_boxes,1])

    output_shape = tf.shape(output)
    output = tf.reshape(
                output,
                [output_shape[0], output_shape[1], output_shape[2],
                 num_of_anchor_boxes, 2+xy_pad_num])

    output = output / outscale_div + outscale_add

    output = tf.reshape(output,
                    [output_shape[0], output_shape[1], output_shape[2], -1])
    return output

def scale_ground_truth(ground_truth, outscale, num_of_gt_bnx_per_cell):
    """Similar to scale_output(), except that it operates on ground_truth labels
    instead of output labels"""

    # [None, None, 2]
    outscale_add_part = outscale[..., 0:2]
    # [None, None, 1]
    outscale_div_part = outscale[..., 2:3]
    outscale_div_part = tf.tile(outscale_div_part, [1,1,2])

    # the first element in a ground truth label is class-id, following with
    # x,y,w,h coordinates.
    outscale_add_shape = tf.shape(outscale_add_part)
    outscale_add_pad_lhs = tf.zeros(
               [outscale_add_shape[0], outscale_add_shape[1], 1],
               dtype=tf.float32
            )
    outscale_add_pad_rhs = tf.zeros(
               [outscale_add_shape[0], outscale_add_shape[1], 2],
               dtype=tf.float32
            )
    # [None, None, 5]
    outscale_add = tf.concat(
            [outscale_add_pad_lhs, outscale_add_part, outscale_add_pad_rhs],
            axis=-1
            )
    # [None, None, 1, 5]
    outscale_add = tf.expand_dims(outscale_add, axis=-2)
    # [None, None, num_of_gt_bnx_per_cell, 5]
    outscale_add = tf.tile(outscale_add, [1,1,num_of_gt_bnx_per_cell,1])

    outscale_div_shape = tf.shape(outscale_div_part)
    outscale_div_pad_lhs = tf.ones(
               [outscale_div_shape[0], outscale_div_shape[1], 1],
               dtype=tf.float32
            )
    outscale_div_pad_rhs = tf.ones(
               [outscale_div_shape[0], outscale_div_shape[1], 2],
               dtype=tf.float32
            )
    outscale_div = tf.concat(
            [outscale_div_pad_lhs, outscale_div_part, outscale_div_pad_rhs],
            axis=-1
            )
    outscale_div = tf.expand_dims(outscale_div, axis=-2)
    outscale_div = tf.tile(outscale_div, [1,1,num_of_gt_bnx_per_cell,1])

    ground_truth_shape = tf.shape(ground_truth)
    ground_truth = tf.reshape(
            ground_truth,
            [ground_truth_shape[0], ground_truth_shape[1], ground_truth_shape[2],
             num_of_gt_bnx_per_cell, 5])

    ground_truth = ground_truth / outscale_div + outscale_add
    ground_truth = tf.reshape(ground_truth,
            [ground_truth_shape[0], ground_truth_shape[1], ground_truth_shape[2], -1])
    return ground_truth

def non_max_suppression_single(output_piece):
    """Perform non max suppression on a single image. """
    # non-max suppression
    selected_indices = tf.image.non_max_suppression(
                        output_piece[...,0:4],
                        output_piece[..., 4],
                        max_output_size=10000,
                        iou_threshold=0.5
                    )
    # mask non-selected box
    one_hot = tf.one_hot(
                selected_indices,
                tf.shape(output_piece)[0],
                dtype=output_piece.dtype
              )
    mask = tf.reduce_sum(one_hot, axis=0)
    output_piece = output_piece * mask[..., None]
    return output_piece

def non_max_suppression_batch(output):
    """Perform non max suppression on a batch of images
    (not really images, but output of the neural network)."""
    result = tf.map_fn(
                lambda output_piece: non_max_suppression_single(output_piece),
                output
             )
    return result

def validation(output, images, num_of_anchor_boxes, num_of_classes=0,
          infer_threshold=0.6, only_xy=False):
    """
        Args:
         output: output of YOLOvx network, shape:
            [-1, -1, -1, num_of_anchor_boxes * (5+num_of_classes)]
            or
            [-1, -1, -1, num_of_anchor_boxes * 3]
            depending on whether @only_xy is False or not.

            Note, the x/y coordinates from the original neural network output are
            relative to the the corresponding cell. Here we expect them to be
            already scaled to relative to the whole image (i.e., used scale_output())

         images: input images, shape [None, None, None, 3]
         infer_threshold: See FLAGS.infer_threshold.
         num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
         num_of_classes: See FLAGS.num_of_classes.
         only_xy: Whether @output only include x/y coordinates.

        Return:
         result: A copy of @images with prediction bounding box on it.
    """
    with tf.variable_scope("validation_scope"):
        output_shape = tf.shape(output)
        output_row_num = output_shape[1]

        # pad output with w/h of 0.1 and classness of 0 if it only contains x/y
        if only_xy:
            output = tf.reshape(
                      output,
                      [-1, num_of_anchor_boxes*output_row_num*output_row_num, 3]
                    )
            shape = tf.shape(output)
            output_pad_wh = tf.ones([shape[0], shape[1], 2], dtype=tf.float32)
            output_pad_wh = output_pad_wh * 0.1
            if num_of_classes:
                output_pad_cls = tf.zeros([shape[0], shape[1], num_of_classes],
                                          dtype=tf.float32)
                output = tf.concat([outut[..., 0:2], output_pad_wh,
                                    outut[..., 2:3], output_pad_cls], axis=-1)
            else:
                output = tf.concat(
                            [output[..., 0:2], output_pad_wh, output[..., 2:3]],
                            axis=-1
                         )

        output = tf.reshape(
                    output,
                    [-1,
                     num_of_anchor_boxes*output_row_num*output_row_num,
                     5+num_of_classes]
                )

        # get P(class) = P(object) * P(class|object)
        # p_class = output[:, :, 5:] * tf.expand_dims(output[:, :, 4], -1)
        # output = tf.concat([output[:, :, 0:5], p_class], axis=-1)

        # mask all bounding boxes whose objectness values are less than threshold.
        output_idx = output[..., 4]
        mask = tf.cast(tf.greater(output_idx, infer_threshold), tf.int32)
        mask = tf.expand_dims(tf.cast(mask, output.dtype), -1)
        masked_output = output * mask
        # NOTE now we just draw all the box, regardless of its classes.
        boxes_x = masked_output[..., 0:1]
        boxes_y = masked_output[..., 1:2]
        boxes_w = masked_output[..., 2:3]
        boxes_h = masked_output[..., 3:4]
        output_rhs = masked_output[..., 4:]
        output = tf.concat([
                boxes_y - boxes_h/2, # ymin
                boxes_x - boxes_w/2, # xmin
                boxes_y + boxes_h/2, # ymax
                boxes_x + boxes_w/2, # xmax
                output_rhs],
                axis=-1
            )

        # We don't want non max suppression when evaluating the dotcount model
        # (the dotcount model should not rely on it)
        if not only_xy:
            output = non_max_suppression_batch(output)

        result = tf.image.draw_bounding_boxes(images, output, name="predict_on_images")
        return result

def build_images_with_bboxes(*args, **kwargs):
    return validation(*args, **kwargs)

def build_images_with_ground_truth(images, ground_truth, num_of_gt_bnx_per_cell):
    """Put ground truth boxes on images.

      Args:
        images: [batch_size, image_size, image_size, 3]
        ground_truth: [batch_size, num_of_gt_bnx_per_cell*feature_map_len*feature_map_len, 5]

      Return:
          result: Images with ground truth bounding boxes on it.
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

def fit_anchor_boxes(output, num_of_anchor_boxes, anchors):
    """Fit the output from the neural network to pre-defined anchor boxes.

      Args:
        output: Computed output from the network. In shape
                [batch_size,
                    image_size/32,
                        image_size/32,
                            (num_of_anchor_boxes * (5 + num_of_classes))]
        num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
        anchors: Pre-defined anchor boxes.
      Return:
        result: Output after fixed with the corresponding anchor boxes."""

    def _fit_single(anchor, split):
        """Fit a single anchor box"""
        assert len(anchor) == 2, "Incorrect anchor length"
        xs = split[..., 0:1]
        ys = split[..., 1:2]
        ws = split[..., 2:3]
        hs = split[..., 3:4]
        obj = split[..., 4:5]
        left = split[..., 5:]

        fit_xs = tf.minimum(tf.sigmoid(xs)+anchor[0], 0.999)
        fit_ys = tf.minimum(tf.sigmoid(ys)+anchor[1], 0.999)
        #w/h is fixed in the current implementation...
        fit_ws = tf.maximum(tf.minimum(tf.sigmoid(ws), 0.999), 0.01)
        fit_hs = tf.maximum(tf.minimum(tf.sigmoid(hs), 0.999), 0.01)
        fit_obj = tf.sigmoid(obj)

        return tf.concat([fit_xs, fit_ys, fit_ws, fit_hs, fit_obj, left], axis=-1)
    # -- END --
    splits = tf.split(output, num_of_anchor_boxes, axis=-1)
    fit = []
    for anchor, split in zip(anchors, splits):
        fit.append(_fit_single(anchor, split))
    result = tf.concat(fit, axis=-1)
    return result

def train():
    """ Train the YOLOvx network. """

    variable_sizes = []
    for i in range(FLAGS.num_of_image_scales):
        variable_sizes.append(FLAGS.image_size_min + i*32)

    tf.logging.info("Building tensorflow graph...")

    # Build YOLOvx only once.
    # Because we have variable size of input, w/h of image are both None (but
    # note that they will eventually have a shape)
    _x = tf.placeholder(tf.float32, [None, None, None, 3])
    _y, vars_to_restore = YOLOvx(
                            _x,
                            backbone_arch=FLAGS.backbone_arch,
                            num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                            num_of_classes=FLAGS.num_of_classes,
                            freeze_backbone=FLAGS.freeze_backbone,
                            reuse=tf.AUTO_REUSE
                          )
    # TODO should not be hard-coded.
    anchors = [(0.25, 0.75), (0.75, 0.75), (0.5, 0.5), (0.25, 0.25), (0.75, 0.25)]
    _y = fit_anchor_boxes(_y, FLAGS.num_of_anchor_boxes, anchors)

    _y_gt = tf.placeholder(
                tf.float32,
                [None, None, None, 5*FLAGS.num_of_gt_bnx_per_cell]
            )

    global_step = tf.Variable(0, name='self_global_step',
                              trainable=False, dtype=tf.int32)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # Not all var in the backbone graph are trainable.
    all_vars.extend(vars_to_restore)
    all_vars = list(set(all_vars))
    all_vars.append(global_step)

    losscal = YOLOLoss(
                  batch_size=FLAGS.batch_size,
                  num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                  num_of_classes=FLAGS.num_of_classes,
                  num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell,
                  global_step=global_step
                )
    loss = losscal.calculate_loss(output = _y, ground_truth = _y_gt)
    tf.summary.scalar("finalloss", loss)

#    starter_learning_rate = 1e-2
#    learning_rate = tf.train.exponential_decay(
#            learning_rate=starter_learning_rate,
#            global_step=global_step,
#            decay_steps=500,
#            decay_rate=0.95,
#            staircase=True
#        )
    learning_rate = FLAGS.starter_learning_rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)

    # scale x/y coordinates of output of the neural network to be relative of
    # the whole image.
    output_scale_placeholder = tf.placeholder(tf.float32, [None, None, 3])
    y_scaled = scale_output(
                    _y,
                    output_scale_placeholder,
                    num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                    num_of_classes=FLASG.num_of_classes
               )
    validation_images = validation(
                            output=y_scaled,
                            images=_x,
                            num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                            num_of_classes=FLAGS.num_of_classes,
                            infer_threshold=FLAGS.infer_threshold
                        )
    tf.summary.image("validation_images", validation_images, max_outputs=3)

    gt_scaled = scale_ground_truth(
                    _y_gt,
                    output_scale_placeholder,
                    num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell
                )
    images_with_grouth_boxes = build_images_with_ground_truth(
                                    _x,
                                    gt_scaled,
                                    num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell
                                )
    tf.summary.image("images_with_grouth_boxes", images_with_grouth_boxes, max_outputs=3)

    tf.logging.info("All network loss/train_step built! Yah!")

    # load images and labels
    reader = DatasetReader(FLAGS.train_files_list, FLAGS.class_name_file)

    initializer = tf.global_variables_initializer()

    run_option = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False))

    merged_summary = tf.summary.merge_all()
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    elif not os.path.isdir(FLAGS.train_log_dir):
        print("{} already exists and is not a dir. Exit.".format(FLAGS.train_log_dir))
        exit(1)
    train_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)

    sess.run(initializer)

    if FLAGS.use_checkpoint:
        if FLAGS.restore_all_variables:
            restorer = tf.train.Saver(all_vars)
        else:
            restorer = tf.train.Saver(vars_to_restore)
        restorer = restorer.restore(sess, FLAGS.checkpoint)
        tf.logging.info("checkpoint restored!")
    # value of `global_step' is restored as well
    saver = tf.train.Saver(all_vars)

    idx = sess.run(global_step)
    while idx != FLAGS.num_of_steps:
        # Change size every 100 steps.
        # `size' is the size of input image, not the final feature map size.
        image_size = variable_sizes[(idx / 100) % len(variable_sizes)]
        if idx % 100 == 0 and idx:
            print("Switching to another image size: %d" % image_size)

        (batch_xs,
         batch_ys,
         outscale) = reader.next_batch(
                       batch_size=FLAGS.batch_size,
                       image_size=image_size,
                       num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell,
                       infinite=True
                     )

        sys.stdout.write("Running train_step[{}]...".format(idx))
        sys.stdout.flush()
        start_time = datetime.datetime.now()
        train_summary, loss_val,_1,_2 = \
            sess.run(
                [merged_summary, loss, train_step, validation_images],
                feed_dict={
                    _x: batch_xs,
                    _y_gt: batch_ys,
                    output_scale_placeholder: outscale},
                options=run_option,
                # run_metadata=run_metadata,
            )
        # validate per `summary_steps' iterations
        if idx % FLAGS.summary_steps == 0:
            train_writer.add_summary(train_summary, idx)

        elapsed_time = datetime.datetime.now() - start_time
        sys.stdout.write(
          "Elapsed time: {}, LossVal: {:10.10f}\n".format(elapsed_time, loss_val)
        )

        # NOTE by now, global_step is always == idx+1, because we have do
        # `train_step`...
        if (idx+1) % 500  == 0:
            ckpt_name = os.path.join(FLAGS.train_ckpt_dir, "model.ckpt")
            if not os.path.exists(FLAGS.train_ckpt_dir):
                os.makedirs(FLAGS.train_ckpt_dir)
            elif not os.path.isdir(FLAGS.train_ckpt_dir):
                print("{} is not a directory.".format(FLAGS.train_ckpt_dir))
                return -1
            saver.save(sess, ckpt_name, global_step=global_step)

        idx += 1
    print("End of training.")

def pad_anchor_boxes(output, num_of_anchor_boxes, anchors):
    """Pad the output from the neural network, which only has the confidence
    value, with x/y coordinates of pre-defined anchor boxes.

      Args:
        output: Computed output from the network. In shape
                [batch_size,
                    image_size/32,
                        image_size/32,
                            num_of_anchor_boxes]
        num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
        anchors: Pre-defined anchor boxes.
      Return:
        result: Output padded with x/y coordinates of corresponding anchor boxes.
    """
    splits = tf.split(output, num_of_anchor_boxes, axis=-1)
    fit = []
    for anchor, split in zip(anchors, splits):
        split_x = tf.ones_like(split) * anchor[0]
        split_y = tf.ones_like(split) * anchor[1]
        obj = tf.sigmoid(split)
        fit.append(tf.concat([split_x, split_y, obj], axis=-1))
    result = tf.concat(fit, axis=-1)
    return result

def train_dot_count():
    """Train the YOLOvx-dotcount network."""

    variable_sizes = []
    for i in range(FLAGS.num_of_image_scales):
        variable_sizes.append(FLAGS.image_size_min + i*32)

    tf.logging.info("Building tensorflow graph...")

    _x = tf.placeholder(tf.float32, [None, None, None, 3])
    _y, vars_to_restore = YOLOvx(
                            _x,
                            backbone_arch=FLAGS.backbone_arch,
                            num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                            freeze_backbone=FLAGS.freeze_backbone,
                            only_confidence=True,
                            reuse=tf.AUTO_REUSE
                          )
    # TODO should not be hard-coded.
    anchors = [(0.25, 0.75), (0.75, 0.75), (0.5, 0.5), (0.25, 0.25), (0.75, 0.25)]
    _y = pad_anchor_boxes(_y, FLAGS.num_of_anchor_boxes, anchors)

    _y_gt = tf.placeholder(
                tf.float32,
                [None, None, None, 5*FLAGS.num_of_gt_bnx_per_cell]
            )

    global_step = tf.Variable(0, name='self_global_step',
                              trainable=False, dtype=tf.int32)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_vars.extend(vars_to_restore)
    all_vars = list(set(all_vars))
    all_vars.append(global_step)

    losscal = DotcountLoss(
                  batch_size=FLAGS.batch_size,
                  num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                  num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell,
                  global_step=global_step
                )
    loss = losscal.calculate_loss(output = _y, ground_truth = _y_gt)
    tf.summary.scalar("finalloss", loss)

#    starter_learning_rate = 1e-2
#    learning_rate = tf.train.exponential_decay(
#            learning_rate=starter_learning_rate,
#            global_step=global_step,
#            decay_steps=500,
#            decay_rate=0.95,
#            staircase=True
#        )
    learning_rate = FLAGS.starter_learning_rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)

    # scale x/y coordinates of output of the neural network to be relative of
    # the whole image.
    output_scale_placeholder = tf.placeholder(tf.float32, [None, None, 3])
    y_scaled = scale_output(
                    _y,
                    output_scale_placeholder,
                    num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                    only_xy=True
               )
    validation_images = validation(
                            output=y_scaled,
                            images=_x,
                            num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                            infer_threshold=FLAGS.infer_threshold,
                            only_xy=True
                        )
    tf.summary.image("validation_images", validation_images, max_outputs=3)

    gt_scaled = scale_ground_truth(
                    _y_gt,
                    output_scale_placeholder,
                    num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell
                )
    images_with_grouth_boxes = build_images_with_ground_truth(
                                    _x,
                                    gt_scaled,
                                    num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell
                                )
    tf.summary.image("images_with_grouth_boxes", images_with_grouth_boxes, max_outputs=3)

    tf.logging.info("All network loss/train_step built! Yah!")

    # load images and labels
    reader = DatasetReader(FLAGS.train_files_list, FLAGS.class_name_file)

    initializer = tf.global_variables_initializer()

    run_option = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False))

    merged_summary = tf.summary.merge_all()
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    elif not os.path.isdir(FLAGS.train_log_dir):
        print("{} already exists and is not a dir. Exit.".format(FLAGS.train_log_dir))
        exit(1)
    train_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)

    sess.run(initializer)

    if FLAGS.use_checkpoint:
        if FLAGS.restore_all_variables:
            restorer = tf.train.Saver(all_vars)
        else:
            restorer = tf.train.Saver(vars_to_restore)
        restorer = restorer.restore(sess, FLAGS.checkpoint)
        tf.logging.info("checkpoint restored!")
    saver = tf.train.Saver(all_vars)

    idx = sess.run(global_step)
    while idx != FLAGS.num_of_steps:
        # Change size every 100 steps.
        # `size' is the size of input image, not the final feature map size.
        image_size = variable_sizes[(idx / 100) % len(variable_sizes)]
        if idx % 100 == 0 and idx:
            print("Switching to another image size: %d" % image_size)

        (batch_xs,
         batch_ys,
         outscale) = reader.next_batch(
                       batch_size=FLAGS.batch_size,
                       image_size=image_size,
                       num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell,
                       infinite=True
                     )

        sys.stdout.write("Running train_step[{}]...".format(idx))
        sys.stdout.flush()
        start_time = datetime.datetime.now()
        train_summary, loss_val,_1,_2 = \
            sess.run(
                [merged_summary, loss, train_step, validation_images],
                feed_dict={
                    _x: batch_xs,
                    _y_gt: batch_ys,
                    output_scale_placeholder: outscale},
                options=run_option,
            )
        # validate per `summary_steps' iterations
        if idx % FLAGS.summary_steps == 0:
            train_writer.add_summary(train_summary, idx)

        elapsed_time = datetime.datetime.now() - start_time
        sys.stdout.write(
          "Elapsed time: {}, LossVal: {:10.10f}\n".format(elapsed_time, loss_val)
        )

        # NOTE by now, global_step is always == idx+1, because we have do
        # `train_step`...
        if (idx+1) % 500  == 0:
            ckpt_name = os.path.join(FLAGS.train_ckpt_dir, "model.ckpt")
            if not os.path.exists(FLAGS.train_ckpt_dir):
                os.makedirs(FLAGS.train_ckpt_dir)
            elif not os.path.isdir(FLAGS.train_ckpt_dir):
                print("{} is not a directory.".format(FLAGS.train_ckpt_dir))
                return -1
            saver.save(sess, ckpt_name, global_step=global_step)

        idx += 1
    print("End of training.")

def test():
    """Test the YOLOvx network"""

    tf.logging.info("Building tensorflow graph...")

    _x = tf.placeholder(tf.float32, [None, None, None, 3])
    _y, vars_to_restore = YOLOvx(
                            _x,
                            backbone_arch=FLAGS.backbone_arch,
                            num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                            num_of_classes=FLAGS.num_of_classes
                          )
    # TODO should not be hard-coded.
    anchors = [(0.25, 0.75), (0.75, 0.75), (0.5, 0.5), (0.25, 0.25), (0.75, 0.25)]
    _y = fit_anchor_boxes(_y, FLAGS.num_of_anchor_boxes, anchors)

    output_scale_placeholder = tf.placeholder(tf.float32, [None, None, 3])
    y_scaled = scale_output(
                    _y,
                    output_scale_placeholder,
                    num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                    num_of_classes=FLAGS.num_of_classes
               )
    images_with_bboxes = build_images_with_bboxes(
                            output=y_scaled,
                            images=_x,
                            num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                            num_of_classes=FLAGS.num_of_classes,
                            infer_threshold=FLAGS.infer_threshold
                         )
    global_step = tf.Variable(0, name='self_global_step',
                              trainable=False, dtype=tf.int32)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_vars.extend(vars_to_restore)
    all_vars = list(set(all_vars))
    all_vars.append(global_step)
    tf.logging.info("All network loss/train_step built! Yah!")

    image_handler = ImageHandler(FLAGS.multiple_images, FLAGS.infile)

    initializer = tf.global_variables_initializer()

    if FLAGS.multiple_images:
        if not os.path.exists(FLAGS.outdir):
            os.makedirs(FLAGS.outdir)

    run_option = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False))
    sess.run(initializer)

    restorer = tf.train.Saver(all_vars)
    restorer = restorer.restore(sess, FLAGS.checkpoint)
    tf.logging.info("checkpoint restored!")

    idx = 1
    while True:
        (batch_xs, batch_xs_scale_info,
            batch_xs_names, outscale) = image_handler.next_batch(FLAGS.batch_size)
        if not len(batch_xs): break
        sys.stdout.write("Testing batch[{}]...".format(idx))
        idx += 1
        sys.stdout.flush()
        start_time = datetime.datetime.now()
        final_images = sess.run(images_with_bboxes,
                                feed_dict={
                                    _x: batch_xs,
                                    output_scale_placeholder: outscale
                                },
                                options=run_option)
        elapsed_time = datetime.datetime.now() - start_time
        sys.stdout.write(
          "Prediction time: {} | Writing images...".format(elapsed_time)
        )

        image_handler.write_batch(
                        final_images,
                        batch_xs_scale_info,
                        batch_xs_names,
                        FLAGS.multiple_images,
                        FLAGS.outdir,
                        FLAGS.outfile
                      )

        sys.stdout.write("\n")
        sys.stdout.flush()

def eval():
    """Evaluate the current model (compute mAP and the like)."""

    print("eval() not implemented currently because we don't need it")
    return
    tf.logging.info("Building tensorflow graph...")

    _x = tf.placeholder(tf.float32, [None, None, None, 3])
    _y, vars_to_restore = YOLOvx(
                            _x,
                            backbone_arch=FLAGS.backbone_arch,
                            num_of_anchor_boxes=FLAGS.num_of_anchor_boxes,
                            num_of_classes=FLAGS.num_of_classes
                          )
    # TODO should not be hard-coded.
    anchors = [(0.25, 0.75), (0.75, 0.75), (0.5, 0.5), (0.25, 0.25), (0.75, 0.25)]
    _y = fit_anchor_boxes(_y, FLAGS.num_of_anchor_boxes, anchors)
    _y_gt = tf.placeholder(
                tf.float32,
                [None, None, None, 5*FLAGS.num_of_gt_bnx_per_cell]
            )

    # restore all variables
    global_step = tf.Variable(0, name='self_global_step',
                              trainable=False, dtype=tf.int32)
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_vars.extend(vars_to_restore)
    all_vars = list(set(all_vars))
    all_vars.append(global_step)
    initializer = tf.global_variables_initializer()
    run_option = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False))
    sess.run(initializer)
    restorer = tf.train.Saver(all_vars)
    restorer = restorer.restore(sess, FLAGS.checkpoint)
    tf.logging.info("checkpoint restored!")

    reader = DatasetReader(FLAGS.eval_files_list, FLAGS.class_name_file)

    image_size = 320
    idx = 1
    op_batch = None
    gt_batch = None
    while True:
        (batch_xs,
         batch_ys,
         outscale) = reader.next_batch(
                       batch_size=FLAGS.batch_size,
                       image_size=image_size,
                       num_of_gt_bnx_per_cell=FLAGS.num_of_gt_bnx_per_cell,
                       infinite=False
                     )
        if not batch_xs: break

        sys.stdout.write("Running eval_step[{}]...".format(idx))
        sys.stdout.flush()
        idx += 1
        start_time = datetime.datetime.now()
        op, gt = sess.run([_y, _gt], feed_dict={_x: batch_xs}, options=run_option)
        elapsed_time = datetime.datetime.now() - start_time
        sys.stdout.write("Prediction time: {}\n".format(elapsed_time))
        sys.stdout.flush()
        if not op_batch:
            op_batch = op
            gt_batch = gt
        else:
            op_batch = np.concatenate([op_batch, op], axis=0)
            gt_batch = np.concatenate([gt_batch, gt], axis=0)
    # calculating mAP
    mAP = map_batch(op_batch, gt_batch, FLAGS.infer_threshold)
    print("mAP: {}".format(mAP))

def main(_):

    if FLAGS.train and FLAGS.test:
        print("ERROR: FLAGS.train & FLAGS.test are both set to True.")
        exit()
    if not FLAGS.train and not FLAGS.test:
        print("ERROR: FLAGS.train & FLAGS.test are both set to False.")

    if FLAGS.train:
        tf.logging.info("Started in training mode. Starting to train...")
        if not FLAGS.dotcount:
            train()
        else:
            train_dot_count()
    elif FLAGS.test:
        tf.logging.info("Started in testing mode...")
        test()

if __name__ == '__main__':
    tf.app.run()
