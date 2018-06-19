#!/usr/bin/python
#coding:utf-8

import datetime
import os
import pdb
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from anchors import anchors_def
from utils.dataset import DatasetReader
from utils.dataset import ImageHandler
from utils.freeze_graph import freeze_graph
from utils.visualization_utils import draw_bounding_box_on_image_array
from networks.yolovx import YOLOvx
from networks.yolovx import YOLOLoss

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

#NOTE!!! tf.app.flags will not warn you for non-existed argument!
tf.app.flags.DEFINE_boolean("train", False, "To train or not.")
tf.app.flags.DEFINE_boolean("test", False, "To test/predict or not.")
tf.app.flags.DEFINE_boolean("evaluate", False, "To evaluate or not.")
tf.app.flags.DEFINE_boolean("multiple_images", False,
        "Predict for multiple images.")

tf.app.flags.DEFINE_string("infile", "./examples/image.jpg", "The image to predict.")
tf.app.flags.DEFINE_alias("test_files_list", "infile")
tf.app.flags.DEFINE_string("outfile", "./predictions/prediction.jpg",
        "Output path of the predictions.")
tf.app.flags.DEFINE_alias("outdir", "outfile")
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

tf.app.flags.DEFINE_boolean("freeze_backbone", False,
        "Freeze the backbone network or not")

tf.app.flags.DEFINE_boolean("use_checkpoint", True,
        "To use or not to use checkpoint.")
tf.app.flags.DEFINE_string("checkpoint", "./vgg_16.ckpt",
        "Path of checkpoint file. Must come with its parent dir name, "
        "even it is in the current directory (eg, ./model.ckpt).")
tf.app.flags.DEFINE_boolean("restore_all_variables", False,
        "Whether or not to restore all variables. Default to False, which "
        "means restore only variables for the backbone network")
tf.app.flags.DEFINE_string("backbone_arch", "inception_v1",
        "The backbone network architecture to use. "
        "Available backbones are  'inception_v1', 'vgg_16', 'resnet_v2', 'inception_v2'. ")
tf.app.flags.DEFINE_alias("backbone", "backbone_arch")

tf.app.flags.DEFINE_string("train_ckpt_dir", "/disk1/yolockpts/",
        "Path to save checkpoints")
tf.app.flags.DEFINE_string("train_log_dir", "/disk1/yolotraining/",
        "Path to save tfevent (for tensorboard)")

tf.app.flags.DEFINE_boolean("exponential_decay", False, "")
tf.app.flags.DEFINE_float("starter_learning_rate", 1e-2, "")
tf.app.flags.DEFINE_float("decay_steps", 10000, "")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "")

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size.")

tf.app.flags.DEFINE_integer("num_classes", 1, "Number of classes.")

tf.app.flags.DEFINE_float("infer_threshold", 0.6, "Objectness threshold")

# NOTE We don't do any clustering for this. Just use 5 as a heuristic.
tf.app.flags.DEFINE_integer("num_anchor_boxes", 5,
        "Number of anchor boxes.")

tf.app.flags.DEFINE_integer("summary_steps", 100, "Write summary ever X steps")

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

tf.app.flags.DEFINE_integer("num_image_scales", 1,
        "Number of scales used to preprocess images. We want different size of "
        "input to our network to bring up its generality.")

# It is pure pain to deal with tensor of variable length tensors (try and screw
# up your life ;-). So we pack each cell with a fixed number of ground truth
# bounding box (even though there is not that many ground truth bounding box
# at that cell). See code at "utils/dataset.py"
tf.app.flags.DEFINE_integer("num_gt_bnx_per_cell", 20,
        "Numer of ground truth bounding box per feature map cell. "
        "If there are not enough ground truth bouding boxes, "
        "some number of fake boxes will be padded.")
tf.app.flags.DEFINE_alias("num_gt_bnx", "num_gt_bnx_per_cell")

tf.app.flags.DEFINE_integer("num_steps", 20000,
        "Max num of step. -1 makes it infinite.")

tf.app.flags.DEFINE_boolean("only_export_tflite", False,
        "Only export the tflite model")

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

def scale_output(output, outscale, num_anchor_boxes, num_classes=1):
    """ Scale x/y coordinates to be relative to the whole image.

        Args:
          output: YOLOvx network output, shape
            [None, None, None, num_anchor_boxes*X],
            where X is "(5+num_classes)" or "3", depending on whether
            @only_xy is False or True.
          outscale: [None, None, 3], the first dimension is against the
            second dimension of @output, the second dimension is against
            the third dimension of @output (that is, we do it across the
            whole batch).
          num_anchor_boxes: See FLAGS.num_anchor_boxes.
          num_classes: See FLAGS.num_classes.

        Return:
          output: the scaled output.
    """
    # [None, None, 2]
    outscale_add_part = outscale[..., 0:2]
    # [None, None, 1]
    outscale_div_part = outscale[..., 2:3]
    outscale_div_part = tf.tile(outscale_div_part, [1,1,2])

    outscale_add_shape = tf.shape(outscale_add_part)
    outscale_add_pad = tf.zeros(
               [outscale_add_shape[0], outscale_add_shape[1], 3+num_classes],
               dtype=tf.float32
            )
    # [None, None, 5+num_classes]
    outscale_add = tf.concat([outscale_add_part, outscale_add_pad], axis=-1)
    # [None, None, 1, 5+num_classes]
    outscale_add = tf.expand_dims(outscale_add, axis=-2)
    # [None, None, num_anchor_boxes, 5+num_classes]
    outscale_add = tf.tile(outscale_add, [1,1,num_anchor_boxes,1])

    outscale_div_shape = tf.shape(outscale_div_part)
    outscale_div_pad = tf.ones(
               [outscale_div_shape[0], outscale_div_shape[1], 3+num_classes],
               dtype=tf.float32
            )
    outscale_div = tf.concat([outscale_div_part, outscale_div_pad], axis=-1)
    outscale_div = tf.expand_dims(outscale_div, axis=-2)
    outscale_div = tf.tile(outscale_div, [1,1,num_anchor_boxes,1])

    output_shape = tf.shape(output)
    output = tf.reshape(
                output,
                [output_shape[0], output_shape[1], output_shape[2],
                 num_anchor_boxes, 5+num_classes])

    output = output / outscale_div + outscale_add

    output = tf.reshape(output,
                    [output_shape[0], output_shape[1], output_shape[2], -1])
    return output

def scale_ground_truth(ground_truth, outscale, num_gt_bnx):
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
    # [None, None, num_gt_bnx, 5]
    outscale_add = tf.tile(outscale_add, [1,1,num_gt_bnx,1])

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
    outscale_div = tf.tile(outscale_div, [1, 1, num_gt_bnx, 1])

    ground_truth_shape = tf.shape(ground_truth)
    ground_truth = tf.reshape(
            ground_truth,
            [ground_truth_shape[0], ground_truth_shape[1],
             ground_truth_shape[2], num_gt_bnx, 5]
        )

    ground_truth = ground_truth / outscale_div + outscale_add
    ground_truth = tf.reshape(
            ground_truth,
            [ground_truth_shape[0], ground_truth_shape[1],
             ground_truth_shape[2], -1]
        )
    return ground_truth

def draw_bounding_boxes(images, output, class_names):
    """Draw bouding boxes of 'output' on 'images'.

      Args:
        images: [None, None, None, 3]
        output: [None, None, 5+num_classes]
        class_names: list of names.
      Return:
        result: Images with bounding boxes on it.
    """
    colors = ['red', 'blue', 'green']

    # define as inner function to share variables
    def draw_bounding_boxes_fn(images, output):
        for img, ops in zip(images, output):
            for op in ops:
                ymin, xmin, ymax, xmax = op[0:4]
                cls_id = np.argmax(op[5:])
                color = colors[cls_id % len(colors)]
                #TODO test len(class_names)
                name = [ class_names[cls_id] ]
                draw_bounding_box_on_image_array(
                        img,
                        ymin,
                        xmin,
                        ymax,
                        xmax,
                        color=color,
                        display_str_list=name,
                        use_normalized_coordinates=True
                    )

        return images
    # --- END ---
    result = tf.py_func(draw_bounding_boxes_fn, [images, output], tf.float32)
    return result

def non_max_suppression_single(output_piece):
    """Perform non max suppression on a single image.

    Note that we perform non max suppression regardless of classness, that is,
    if bounding boxes of two class have a high IoU, one will be suppressed.
    """
    # non-max suppression
    selected_indices = tf.image.non_max_suppression(
                        output_piece[...,0:4],
                        output_piece[..., 4],
                        max_output_size=10000,
                        iou_threshold=0.6
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

def validation(output, images, num_anchor_boxes, num_classes=1,
        class_names=['person'], infer_threshold=0.6):
    """
      Args:
       output: output of YOLOvx network, shape:
          [-1, -1, -1, num_anchor_boxes * (5+num_classes)]

          Note, the x/y coordinates from the original neural network output are
          relative to the the corresponding cell. Here we expect them to be
          already scaled to relative to the whole image (i.e., using
          scale_output())

       images: input images, shape [None, None, None, 3]
       infer_threshold: See FLAGS.infer_threshold.
       num_anchor_boxes: See FLAGS.num_anchor_boxes.
       num_classes: See FLAGS.num_classes.
       class_names: List of class names.
       infer_threshold: See FLAGS.infer_threshold.

      Return:
       result: A copy of @images with prediction bounding box on it.
    """
    with tf.variable_scope("validation_scope"):
        output_shape = tf.shape(output)
        output_row_num = output_shape[1]

        output = tf.reshape(
                    output,
                    [-1, num_anchor_boxes*(output_row_num**2), 5+num_classes]
                )

        # mask all bounding boxes whose objectness values are less than
        # threshold.
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

        output = non_max_suppression_batch(output)

        #TODO
        #Using tf.py_func to draw bounding boxes on images is too costly
        #probably because of the data-transfer between CPU and GPU.
        # result = draw_bounding_boxes(images, output, class_names)
        result = tf.image.draw_bounding_boxes(images, output)
        return result

def build_images_with_bboxes(*args, **kwargs):
    return validation(*args, **kwargs)

def get_num_bnx_from_output(output, num_anchor_boxes, num_classes=1,
                            infer_threshold=0.65, name="output_num_array"):
    """Get number of output bounding box.

      Args:
          output: Output from the neural network.
          infer_threshold: See FLAGS.infer_threshold.
          name: name of the output tensor.
      Return:
          num_array: A number in an array, such as [5]. We return an array
            instead of a number because toco has only --output_arrays argument.
    """
    output_shape = tf.shape(output)
    output_row_num = output_shape[1]

    output = tf.reshape(
                output,
                [-1, num_anchor_boxes*(output_row_num**2), 5+num_classes]
            )

    # mask all bounding boxes whose objectness values are less than
    # threshold.
    output_idx = output[..., 4]
    mask = tf.cast(tf.greater(output_idx, infer_threshold), tf.int32)
    num = tf.reduce_sum(mask)
    num_array = tf.stack([num], name=name)
    return num_array

def build_images_with_ground_truth(ground_truth, images, num_gt_bnx,
        class_names=['person']):
    """Put ground truth boxes on images.

      Args:
        ground_truth: [batch_size,
                            feature_map_len,
                                feature_map_len, num_gt_bnx*5]
        images: [batch_size, image_size, image_size, 3]
        num_gt_bnx: See FLAGS.num_gt_bnx.
        class_names: List of class names.

      Return:
          result: Images with ground truth bounding boxes on it.
    """
    with tf.variable_scope("build_image_scope"):
        feature_map_len = tf.shape(images)[1]/32
        ground_truth = tf.reshape(
                            ground_truth,
                            [-1, num_gt_bnx*(feature_map_len**2), 5]
                        )
        cls = ground_truth[..., 0:1]
        x = ground_truth[..., 1:2]
        y = ground_truth[..., 2:3]
        w = ground_truth[..., 3:4]
        h = ground_truth[..., 4:5]
        #pad boxes with a fake confidence value and cls value so that it has
        #the same shape as in validation().
        boxes = tf.concat([
                    y - h/2, # ymin
                    x - w/2, # xmin
                    y + h/2, # ymax
                    x + w/2, # xmax
                    cls,
                    cls
                ],
                axis=-1
            )

        # result = draw_bounding_boxes(images, boxes, class_names)
        result = tf.image.draw_bounding_boxes(images, boxes)
        return result

def fit_anchor_boxes(output, num_anchor_boxes, anchors):
    """Fit the output from the neural network to pre-defined anchor boxes.

      Args:
        output: Computed output from the network. In shape
                [batch_size,
                    image_size/32,
                        image_size/32,
                            (num_anchor_boxes * (5 + num_classes))]
        num_anchor_boxes: See FLAGS.num_anchor_boxes.
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

        return tf.concat(
                  [fit_xs, fit_ys, fit_ws, fit_hs, fit_obj, left],
                  axis=-1
               )
    # -- END --
    splits = tf.split(output, num_anchor_boxes, axis=-1)
    fit = []
    for anchor, split in zip(anchors, splits):
        fit.append(_fit_single(anchor, split))
    result = tf.concat(fit, axis=-1)
    return result

def train():
    """ Train the YOLOvx network. """

    variable_sizes = []
    for i in range(FLAGS.num_image_scales):
        variable_sizes.append(FLAGS.image_size_min + i*32)

    tf.logging.info("Building tensorflow graph...")

    # Build YOLOvx only once.
    # Because we have variable size of input, w/h of image are both None (but
    # note that they will eventually have a shape)
    _x = tf.placeholder(tf.float32, [None, None, None, 3])
    _y, vars_to_restore = YOLOvx(
                            _x,
                            backbone_arch=FLAGS.backbone_arch,
                            num_anchor_boxes=FLAGS.num_anchor_boxes,
                            num_classes=FLAGS.num_classes,
                            freeze_backbone=FLAGS.freeze_backbone,
                            reuse=tf.AUTO_REUSE
                          )
    if not FLAGS.num_anchor_boxes in anchors_def:
        print("anchors not defined for anchor number {}".format(FLAGS.num_anchor_boxes))
        exit()
    anchors = anchors_def[FLAGS.num_anchor_boxes]
    _y = fit_anchor_boxes(_y, FLAGS.num_anchor_boxes, anchors)

    _y_gt = tf.placeholder(
                tf.float32,
                [None, None, None, 5*FLAGS.num_gt_bnx]
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
                  num_anchor_boxes=FLAGS.num_anchor_boxes,
                  num_classes=FLAGS.num_classes,
                  num_gt_bnx=FLAGS.num_gt_bnx,
                  global_step=global_step
                )
    loss = losscal.calculate_loss(output = _y, ground_truth = _y_gt)

    if FLAGS.exponential_decay:
        learning_rate = tf.train.exponential_decay(
                learning_rate=FLAGS.starter_learning_rate,
                global_step=global_step,
                decay_steps=FLAGS.decay_steps,
                decay_rate=FLAGS.decay_rate,
                staircase=True
            )
    else:
        learning_rate = FLAGS.starter_learning_rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = slim.learning.create_train_op(
                    loss,
                    optimizer,
                    global_step=global_step
                 )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)

    # load images and labels
    reader = DatasetReader(FLAGS.train_files_list, FLAGS.class_name_file)
    class_names = reader.get_class_names()
    #TODO we should make output_scale_placeholder a plain numpy array instead
    #of a tensor.

    # scale x/y coordinates of output of the neural network to be relative of
    # the whole image.
    output_scale_placeholder = tf.placeholder(tf.float32, [None, None, 3])
    y_scaled = scale_output(
                    _y,
                    output_scale_placeholder,
                    num_anchor_boxes=FLAGS.num_anchor_boxes,
                    num_classes=FLAGS.num_classes
               )
    validation_images = validation(
                            output=y_scaled,
                            images=_x,
                            num_anchor_boxes=FLAGS.num_anchor_boxes,
                            num_classes=FLAGS.num_classes,
                            class_names=class_names,
                            infer_threshold=FLAGS.infer_threshold
                        )
    tf.summary.image("validation_images", validation_images, max_outputs=3)

    gt_scaled = scale_ground_truth(
                    _y_gt,
                    output_scale_placeholder,
                    num_gt_bnx=FLAGS.num_gt_bnx
                )
    images_with_gt_boxes = build_images_with_ground_truth(
                                   ground_truth=gt_scaled,
                                   images=_x,
                                   num_gt_bnx=FLAGS.num_gt_bnx,
                                   class_names=class_names
                               )
    tf.summary.image("images_with_gt_boxes", images_with_gt_boxes, max_outputs=3)

    tf.logging.info("All network loss/train_step built! Yah!")

    initializer = tf.global_variables_initializer()

    run_option = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False))

    merged_summary = tf.summary.merge_all()
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    elif not os.path.isdir(FLAGS.train_log_dir):
        print("{} already exists and is not a dir.".format(FLAGS.train_log_dir))
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
    saver = tf.train.Saver(all_vars, max_to_keep=5000)

    idx = sess.run(global_step)
    while idx != FLAGS.num_steps:
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
                       num_gt_bnx=FLAGS.num_gt_bnx,
                       infinite=True
                     )

        sys.stdout.write("Running train_step[{}]...".format(idx))
        sys.stdout.flush()
        start_time = datetime.datetime.now()
        loss_val,_1, = \
            sess.run(
                [loss, train_step],
                feed_dict={
                    _x: batch_xs,
                    _y_gt: batch_ys,
                    output_scale_placeholder: outscale
                },
                options=run_option,
            )
        # validate per `summary_steps' iterations
        if idx % FLAGS.summary_steps == 0:
            train_summary = \
                sess.run(
                    merged_summary,
                    feed_dict={
                        _x: batch_xs,
                        _y_gt: batch_ys,
                        output_scale_placeholder: outscale
                    },
                    options=run_option,
                )
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

    _x = tf.placeholder(tf.float32, [None, None, None, 3], name="input_images")
    _y, vars_to_restore = YOLOvx(
                            _x,
                            backbone_arch=FLAGS.backbone_arch,
                            num_anchor_boxes=FLAGS.num_anchor_boxes,
                            num_classes=FLAGS.num_classes
                          )
    if not FLAGS.num_anchor_boxes in anchors_def:
        print("anchors not defined for anchor number {}".format(FLAGS.num_anchor_boxes))
        exit()
    anchors = anchors_def[FLAGS.num_anchor_boxes]
    _y = fit_anchor_boxes(_y, FLAGS.num_anchor_boxes, anchors)

    output_scale_placeholder = tf.placeholder(tf.float32, [None, None, 3])
    y_scaled = scale_output(
                    _y,
                    output_scale_placeholder,
                    num_anchor_boxes=FLAGS.num_anchor_boxes,
                    num_classes=FLAGS.num_classes
               )
    images_with_bboxes = build_images_with_bboxes(
                            output=y_scaled,
                            images=_x,
                            num_anchor_boxes=FLAGS.num_anchor_boxes,
                            num_classes=FLAGS.num_classes,
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

    output_num_array = get_num_bnx_from_output(
                           output=y_scaled,
                           num_anchor_boxes=FLAGS.num_anchor_boxes,
                           num_classes=FLAGS.num_classes,
                           infer_threshold=FLAGS.infer_threshold,
                           name="output_num_array"
                       )
    if FLAGS.only_export_tflite:
        print("Outputing tflite ...")

        # meta graph containing network structure (proto)
        print("Writing graph_def proto to /tmp/mymodels/model.pbtxt ...")
        tf.train.write_graph(sess.graph_def, "/tmp/mymodels", "model.pbtxt")

        # frozen graph containing network structure and weights
        print("Writing frozen graph to /tmp/mymodels/model_frozen.pb ...")
        input_graph_proto_path = "/tmp/mymodels/model.pbtxt"
        input_checkpoint_path = FLAGS.checkpoint
        output_graph_path = "/tmp/mymodels/model_frozen.pb"
        output_node_name = "output_num_array"
        freeze_graph(
                input_graph=input_graph_proto_path,
                output_graph=output_graph_path,
                input_checkpoint=input_checkpoint_path,
                input_saver="",
                checkpoint_version=2,
                input_binary=False,
                output_node_names=output_node_name,
                restore_op_name="",
                filename_tensor_name="save/Const:0", #deprecated
                clear_devices=True,
                initializer_nodes="",
                variable_names_whitelist="",
                variable_names_blacklist="",
                input_meta_graph="",
                input_saved_model_dir="",
                saved_model_tags="serve"
            )

        # TODO the python API are only available in Tensorflow1.9. Use the cmd
        # `toco` instead:
        #
        #       IMAGE_SIZE=320
        #       toco \
        #         --input_file=/tmp/mymodels/model_frozen.pb \
        #         --output_file=/tmp/mymodels/converted_model.lite \
        #         --input_format=TENSORFLOW_GRAPHDEF \
        #         --output_format=TFLITE \
        #         --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
        #         --input_array=input_images \
        #         --output_array=output_num_array \
        #         --inference_type=FLOAT \
        #         --input_data_type=FLOAT
        #
        # TODO as of this writing, lots of operation used in this project are
        # not supported. So we cannot converted this model to tflite. See this
        # issue:
        #   https://github.com/tensorflow/tensorflow/issues/20110
        #
        #input_arrays = ["input_images"]
        #output_arrays = ["output_num_array"]
        #converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
        #                                            output_graph_path,
        #                                            input_arrays,
        #                                            output_arrays
        #                                          )
        #tflite_model = converter.convert()
        #tflite_output_path = "/tmp/mymodels/converted_model.tflite"
        #print("Writing converted tflite model to {}".format(tflite_output_path))
        #open(tflite_output_path, "wb").write(tflite_model)
        return

    restorer = tf.train.Saver(all_vars)
    restorer = restorer.restore(sess, FLAGS.checkpoint)
    tf.logging.info("checkpoint restored!")

    idx = 1
    # Profile the network structure
    #
    #with tf.contrib.tfprof.ProfileContext('/tmp/profile_dir',
    #                                      trace_steps=[0],
    #                                      dump_steps=[0]) as pctx:
    #    #uncomment these to see flops benchmarks
    #    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    #    pctx.add_auto_profiling('op', opts, [0])
    #
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
                            num_anchor_boxes=FLAGS.num_anchor_boxes,
                            num_classes=FLAGS.num_classes
                          )
    if not FLAGS.num_anchor_boxes in anchors_def:
        print("anchors not defined for anchor number {}".format(FLAGS.num_anchor_boxes))
        exit()
    anchors = anchors_def[FLAGS.num_anchor_boxes]
    _y = fit_anchor_boxes(_y, FLAGS.num_anchor_boxes, anchors)
    _y_gt = tf.placeholder(
                tf.float32,
                [None, None, None, 5*FLAGS.num_gt_bnx]
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
                       num_gt_bnx=FLAGS.num_gt_bnx,
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
        train()
    elif FLAGS.test:
        tf.logging.info("Started in testing mode...")
        test()
    elif FLAGS.evaluate:
        tf.logging.info("Started in eval mode...")
    else:
        tf.logging.info("What are you going to do (train/test/evaluate)?")

if __name__ == '__main__':
    tf.app.run()
