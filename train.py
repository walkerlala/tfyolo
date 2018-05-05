#!/usr/bin/python
#coding:utf-8

import datetime
import os
import pdb
import sys
import tensorflow as tf
from utils.dataset import DatasetReader
from networks.yolovx import YOLOvx
from networks.yolovx import YOLOLoss

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
        # TODO now we just draw all the box, regardless of its classes.
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

def train():
    """ Train the YOLOvx network. """

    training_file_list = FLAGS.training_file_list
    class_name_file = FLAGS.class_name_file
    batch_size = FLAGS.batch_size
    num_of_image_scales = FLAGS.num_of_image_scales
    image_size_min = FLAGS.image_size_min
    num_of_classes = FLAGS.num_of_classes
    num_of_anchor_boxes = FLAGS.num_of_anchor_boxes
    num_of_gt_bnx_per_cell = FLAGS.num_of_gt_bnx_per_cell
    num_of_steps = FLAGS.num_of_steps
    checkpoint_file = FLAGS.checkpoint_file
    restore_all_variables = FLAGS.restore_all_variables
    train_ckpt_dir = FLAGS.train_ckpt_dir
    train_log_dir = FLAGS.train_log_dir
    freeze_backbone = FLAGS.freeze_backbone
    infer_threshold = FLAGS.infer_threshold

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

    # tf.summary.image("images_input", _x, max_outputs=3)

    # images_with_grouth_boxes = build_images_with_ground_truth(
    #                                 _x,
    #                                 _y_gt,
    #                                 num_of_gt_bnx_per_cell=num_of_gt_bnx_per_cell
    #                             )
    # tf.summary.image("images_with_grouth_boxes", images_with_grouth_boxes, max_outputs=3)

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
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss, global_step=global_step)

    validation_images = validation(output=_y, images=_x,
                                   num_of_anchor_boxes=num_of_anchor_boxes,
                                   num_of_classes=num_of_classes,
                                   infer_threshold=infer_threshold)
    tf.summary.image("images_validation", validation_images, max_outputs=3)

    tf.logging.info("All network loss/train_step built! Yah!")

    # load images and labels
    reader = DatasetReader(training_file_list, class_name_file)

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
                                num_of_gt_bnx_per_cell=num_of_gt_bnx_per_cell
                            )

        sys.stdout.write("Running train_step[{}]...".format(idx))
        sys.stdout.flush()
        start_time = datetime.datetime.now()
        train_summary, loss_val,_1,_2 = \
            sess.run(
                [merged_summary, loss, train_step, validation_images],
                feed_dict={_x: batch_xs, _y_gt: batch_ys},
                options=run_option,
                # run_metadata=run_metadata,
            )
        train_writer.add_summary(train_summary, idx)
        elapsed_time = datetime.datetime.now() - start_time
        sys.stdout.write(
          "Elapsed time: {}, LossVal: {:10.10f} | ".format(elapsed_time, loss_val)
        )
        print("Validating this batch....")

        # NOTE by now, global_step is always == idx+1, because we have do
        # `train_step`...
        if (idx+1) % 500  == 0:
            ckpt_name = os.path.join(train_ckpt_dir, "model.ckpt")
            if not os.path.exists(train_ckpt_dir):
                os.makedirs(train_ckpt_dir)
            elif not os.path.isdir(train_ckpt_dir):
                print("{} is not a directory.".format(train_ckpt_dir))
                return -1
            saver.save(sess, ckpt_name, global_step=global_step)

        idx += 1

def main(_):

    tf.logging.info("yolo.py started in training mode. Starting to train...")
    train()

if __name__ == '__main__':
    tf.app.run()
