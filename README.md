This README will guide you through:

    0. What we are doing and what we have achieved.

    1. The organization of this project, including source files structure and
       source code structure.

    2. How to train/test, how to make improvement based on this project.

    3. How to convert a tensorflow model that can run successfully on a PC to a
       tflite model that can run on android.

### what we are doing and what we have achieved

Basically we are doing Object Detection here. We build a  deep neural network
similar to that of YOLO and try to simplified the network structure so that it
can run faster and more accurate. But we haven't out-performed yet (sigh!). We
have successfully trained several models which can perform well on training
dataset but fail to generalize on testing dataset (i.e., overfitting).

### Source code/file organization

####Source files organzation

 1. backbone

    This directory contains backbone network structures. As of this writing we
    have 'vgg_16', 'inception_v1', 'inception_v2', 'resnet_v2' and a DIY 'vx'
    backbone structure.

    The reason why we have these network backbone is that our model is based on
    the following schema:

```
              +---------+    +--------+
    input =>  |         | => |        | => output bounding boxes predictions
              +---------+    +--------+
               backbone     conv' layers
```
   the backbone layer are well-trained on classification task (such as ImageNet)
   so that it can extract various features from pictures. we then stack a few
   convolutional layers on top of that backbone for prediction bounding boxes.

   So, to use those backbones, you have to obtain their parameter weights, which
   can be downloaded from
   [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
   (note that the vx model is a DIY one).

   Note that we have modified some code (mostly removing the last few layers) of
   those model to fit our object detection task.

 2. examples

    This directory contains example testing pictures.

 3. networks

    This directory contains source code for the core network structure (which is
    called YOLOvx).

 4. predictions

    This directory contains the prediction output of the testing pictures.

 5. pretrained

    This directory contains the pretrained weight downloaded from
    [tensorflow-model](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
    previously

 6. utils

    This directory contains utilization code for reading/writing data and drawing images.

 7. anchors.py

    This file contains anchors definition

 8. main.py

    This file is used to run the whole model (that is why it is called main.py)

####Source code organzation
If you want to read the source code, start from **main.py**. For training the
model, see function `train()`; for testing the model, see function `test()`.
(and ignore the `eval()` function for this moment).

 1. How `train()` trains the model

    To understand this problem, you have know how Tensorflow works. Basically
    in Tensorflow you define a *static* graph and then use this graph to train
    the model for many times. 

    So we define the neural network as follow:
    ```
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
    ```

    where `_x` is the input of the neural network and `_y` the output.

    And then we calculate the loss by
    ```
    losscal = YOLOLoss(
                  batch_size=FLAGS.batch_size,
                  num_anchor_boxes=FLAGS.num_anchor_boxes,
                  num_classes=FLAGS.num_classes,
                  num_gt_bnx=FLAGS.num_gt_bnx,
                  global_step=global_step
                )
    loss = losscal.calculate_loss(output = _y, ground_truth = _y_gt)
    ```

    And then we define a *train\_step* by:
    ```
    train_step = slim.learning.create_train_op(
                    loss,
                    optimizer,
                    global_step=global_step
                 )
    ```

    So far we have contructed the whole model for training. But
    **note that we have just contructed a static graph!!!**. When the Python
    intepreter reach this point, it will contruct a static graph, but there is
    no training done yet.

    One more thing you have to know: `train\_step` depends on `loss`, `loss`
    depends on `_y`, and `_y` depends on `_x`. So when you run `train\_step`, it
    will drives the whole graph!

    To train the model, you have to feed data into `_x`, run the network,
    get the loss and perform gradient descent; and then feed data into `_x`
    again, run the network, and perform gradient descent again ... This is done
    through the while loop following.

 2. How `test()` test/predict a picture

    Basically it follow the same schema in `train()`.

 3. How YOLOvx is contructed. See function `YOLOvx()` in *networks/yolovx.py*.

 4. How the loss is defined. See function `calculate_loss()` in
    *networks/yolovx.py*. Note that this function contains some complicated
    Tensorflow operations. You should know how YOLO do the loss calculation
    (see the YOLO v1 paper) before trying to read that.

 5. How we handle data reading/writing

    Functions for reading/writing data is defined in *utils/dataset.py*. We
    assume the training data follow the YOLO/darknet dataset schema. For
    example, if you have a training image **/path/to/somepicture.jpg**, you
    should put a file **/path/to/somepicture.txt**. containing the
    corresponding annotations for that image. Note that these two files should
    be put *in the same directory(!!)*. Inside **/path/to/somepicture.txt**,
    there are multiple lines of annotation, each line indicating a bounding box.
    For example, a line:

    > 0  0.08671875  0.329861111111  0.0546875  0.115277777778

    the first column is the classness; the second column is the x coordinate of
    the box center (relative to the whole image); the third is the y coordinate
    of the box center (relative to the whole image); the third is the width of
    the box (relative to the whole image); the fourth is the height of the box
    (relative to whole image).

    Inside **utils/dataset.py**, we transformed the x/y coordinates of the box
    from *relative to whole image* to be *relative to a single cell* (see the
    YOLO papers for what a cell means).

    Because different pictures contain different number of bounding boxes, we
    pack bounding boxes to a fixed number with fake one, which is
    [0, 0, 0, 0, 0]. See FLAGS.num_gt_bnx.

    As of this writing, most of the labeled data is stored at `/disk1/labeled/`.

    If you want to annotate more data for training, you can use this tool:

        https://github.com/tzutalin/labelImg
    or
        https://github.com/AlexeyAB/Yolo_mark

    for image labeling. Nevertheless it has to conform to the data format
    described above.

### How to train/test and how to make improvement based on this project

    We use `main.py` to drive training and testing. See `./main.py --help` for a
    complete description.
    
#### Training

```
./main.py --train                               \  
    --nofreeze_backbone                         \ # whether or not to freeze the backbone
    --backbone inception_v1                     \ # use the `inception_v1` backbone
    --norestore_all_variables                   \ # retore only the weights for the `inception_v1` backbone
    --checkpoint ./pretrained/inception_v1.ckpt \ # the checkpoint
    --batch_size 16 --freeze_backbone           \ # the batch size
    --infer_threshold 0.6                       \ # how confident a output bounding box should be before it is treated as a true one
    --num_image_scales 1                        \ # multi-scale training, how many scales
    --num_steps -1                              \ # number of training steps. -1 mean infinite training
    --num_anchor_boxes 5                        \ # num of anchor box per cell
    --starter_learning_rate 1e-1                \ # learning rate
    --summary_steps 50                          \ # how many steps before making a (full) checkpoint
    --train_ckpt_dir /disk1/yolockpts/run0      \ # where checkpoints be saved
    --train_log_dir /disk1/yolotraining/run0    \ # where the log be save.
    --train_files_list /disk1/labeled/roomonly_train.txt \ # file containing locations of all training images
    > /disk1/yolotraining/run0.txt 2>&1 &
```

   You then can use `tensorboard --logdir /disk1/yolotraining/run0/` to view the
   training process at runtime. If you want to limit the training process to
   only one GPU, do `export CUDA_VISIBLE_DEVICES=0` or 
   `export CUDA_VISIBLE_DEVICES=1` before training. If you want to use CPU for
   training, do `export CUDA_VISIBLE_DEVICES=0`.

   Right now we have several training set: COCO2014, PASCAL-VOC2007+2012 and a
   DIY in-classroom dataset, which is described by

        /disk1/labeled/trainall_coco.txt

        /disk1/labeled/trainall_voc.txt

        /disk1/labeled/roomonly_all.txt

   respectively.

#### Testing

```
./main.py --test                                        \ # 
    --backbone inception_v1                             \ # backbone
    --num_anchor_boxes 5                                \ # num of anchor boxes
    --checkpoint /disk1/yolockpts/run0/model.ckpt-25000 \ # path to checkpoint
    --infile examples/image.jpg                         \ # image to test
    --outfile predictions/test-out-vx-image-145000.jpg    # path of the the output
```

You can also test multiple images at a time:


```
./main.py --test                                        \ # 
    --backbone inception_v1                             \ # backbone
    --num_anchor_boxes 5                                \ # num of anchor boxes
    --checkpoint /disk1/yolockpts/run0/model.ckpt-25000 \ # path to checkpoint
    --multiple_images                                   \ # multiple images
    --infile file_list.txt                              \ # file containing paths to images
    --outdir predictions                                  # output directory
```

