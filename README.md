This README will guide you through:

 1. What we are doing and what we have achieved.

 2. The organization of this project, including source files structure and
    source code structure.

 3. How to train/test, how to make improvement based on this project.

 4. How to convert a tensorflow model that can run successfully on a PC to a
    tflite model that can run on android.

### I what we are doing and what we have achieved

Basically we are doing Object Detection here. We build a  deep neural network
similar to that of YOLO and try to simplified the network structure so that it
can run faster and more accurate. But we haven't out-performed YOLO (the darknet
implementation) yet (sigh!). We have successfully trained several models which
can perform well on training dataset but fail to generalize on testing dataset
(i.e., overfitting).

### II Source code/file organization

#### Source files organzation

 1. **backbone**

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
   convolutional layers on top of that backbone for predicting bounding boxes.

   So, to use those backbones, you have to obtain their parameter weights, which
   can be downloaded from
   [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
   (note that the vx model is a DIY one and thus cannot be found there). We
   previously have downloaded a few weights into the *./pretrained* directory.

   Note that we have modified some code (removing the last few layers and change
   the activation function) of some models to fit our object detection task.

 2. **examples**

    This directory contains example testing pictures.

 3. **networks**

    This directory contains source code for the core network structure (which is
    called YOLOvx).

 4. **predictions**

    This directory (should) contain(s) the prediction output of the testing pictures.

 5. **pretrained**

    This directory contains the pretrained weights previously downloaded from
    [tensorflow-model](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

 6. **utils**

    This directory contains utilization code for reading/writing data and drawing images.

 7. **anchors.py**

    This file contains anchors definitions.

 8. **main.py**

    This file is used to run the whole model (that is why it is called main.py)

#### Source code organzation

If you want to read the source code, start from *main.py*. For training the
model, see function `train()`; for testing the model, see function `test()`.
(and ignore the `eval()` function for this moment).

 1. How `train()` trains the model

    To understand this, you have know how Tensorflow works. Basically in
    Tensorflow you define a *static* graph and then use this graph to train
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

    Note that `train_step` depends on `loss`, `loss` depends on `_y`, and `_y`
    depends on `_x`. So when you run `train_step`, it will drives the whole
    graph!

    With these we have contructed the whole model for training. But
    **note that we have just contructed a static graph!!!**. When the Python
    intepreter reaches this point, it will contruct a static graph, but there is
    no training done yet.

    To train the model, you have to feed data into `_x`, run the network,
    get the loss and perform gradient descent; and then feed data into `_x`
    again, run the network, and perform gradient descent again ... This is done
    through the while loop following.

 2. How `test()` test/predict a picture

    Basically it follow the same schema in `train()`.

 3. How YOLOvx is contructed.
 
    See function `YOLOvx()` in *networks/yolovx.py*.

 4. How the loss is defined.
 
    See function `calculate_loss()` in *networks/yolovx.py*. Note that this
    function contains some complicated Tensorflow operations. You should
    first understand how YOLO do the loss calculation (see the YOLO v1 paper)
    before trying to read that.

 5. How we handle data reading/writing

    Functions for reading/writing data is defined in *utils/dataset.py*. We
    assume the training data follow the YOLO/darknet dataset schema. For
    example, if you have a training image */path/to/somepicture.jpg*, you
    should put a file */path/to/somepicture.txt* containing the
    corresponding annotations for that image. Note that these two files should
    be put **in the same directory(!!)**. Inside **/path/to/somepicture.txt**,
    there are multiple lines of annotation, each line indicating a bounding box.
    For example, a line:

    ```
    0   0.08671875   0.329861111111   0.0546875   0.115277777778
    ```

    the first column is the classness; the second column is the x coordinate of
    the box center (relative to the whole image); the third is the y coordinate
    of the box center (relative to the whole image); the third is the width of
    the box (relative to the whole image); the fourth is the height of the box
    (relative to whole image).

    Inside *utils/dataset.py*, we transformed the x/y coordinates of the box
    from *relative to whole image* to be *relative to a single cell* (see the
    YOLO papers for what a cell means).

    Because different pictures contain different number of bounding boxes, we
    pack bounding boxes to a fixed number with fake one, which is
    [0, 0, 0, 0, 0]. See FLAGS.num_gt_bnx.

    As of this writing, most of the labeled data is stored at `/disk1/labeled/`.

    If you want to annotate more data for training, you can use this tool

    https://github.com/tzutalin/labelImg

    or

    https://github.com/AlexeyAB/Yolo_mark

    for image labeling. Nevertheless it has to conform to the data format
    described above.

### III How to train/test and how to make improvement based on this project

We use `main.py` to drive both training and testing. See `./main.py --help` for
a full description.
    
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

### IIII How to make improvement based on this project

I suppose you are a young researcher like me? Congratulation! Welcome to the new
world of mysterious neural network training!

#### Tips

Some lessons learned by me which may help you:

 1. Most ideas do not work, so if you want to lay a good foundation, be patient.

 2. Test your models thoroughly before truely believing that it works.

 3. Try to visualize your model at the time of training. But also note that
    training a workable neural network is not easy and usually takes hours, so
    keep yourself easy.

 4. Try to get more data and try again.

 5. Code style matters. Write your code clearly and write comments if necessary.
    Try to learn from how
    [Googler write code](https://github.com/tensorflow/models).

 5. Commit often, and write clear commit logs! You are the person who will read
    the code most of the time following, so be nice to yourself.

#### What to get from this project

I think you should look at the code yourself. But I will try to provide you some
useful pointers:

 1. `tf.slim` is great for constructing neural networks. And we are using it
    heavily in our code. See its
    [tutorial](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

 2. The official [tensorflow-model](https://github.com/tensorflow/models) repo
    contains lots of pre-built model using tensorflow. For the object detection
    task, look at its
    [object detection directory](https://github.com/tensorflow/models/tree/master/research/object_detection)
    for various models built with Tensorflow. Note that these models are often
    in active development and are not necessary the ones used in Google
    internally, so things may break from time to time.

 3. The official YOLO implementation is implemented using C. Here is its
    [official website](https://pjreddie.com/darknet/yolo/). We have clone the
    repo into */home/yubin/project/darknet* and write some notes in the
    *README2.md* file.

 4. There are a darknet fork [here](https://github.com/AlexeyAB/darknet). Its
    README file are very comprehansive.

 5. If you have any question about Tensorflow, go to
    [Stackoverflow](https://stackoverflow.com/questions/tagged/tensorflow?sort=newest&pageSize=15)
    or post it in 
    [specific tensorflow mailing lists](https://www.tensorflow.org/community/lists).


#### Some caveat about this project.

 1. We have changed the internal activation function of most backbone network
    from `tf.nn.relu()` to `tf.nn.leaky_relu()`, which can help the model to
    fit the data better. *But*, one drawback about leaky_relu is that is will
    sometimes explode the network. So sometimes you will have errors like

    > InvalidArgumentError (see above for traceback): LossTensor is inf or nan: Tensor had NaN values

    To solve this, you can 1) decrease the `starter_learning_rate` or 2)
    decrease the `batch_size`.

#### Where to get help

 2. Here is my [email](ablacktshirt@gmail.com). Draw me some lines if needed. I
    reply emails.


### V How to converted this model to tflite

Basically we are following
[this](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) and
[this](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#0)
tutorial to convert a model to tflite. But due to tflite's limitation
(see below) we **have not successfully completed that yet**.

We are using the `toco` tool, whose docs can be found
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/).

First generate a frozen tensorflow graph by

```
./main.py --test         \
    --only_export_tflite \
    --backbone vgg_16    \
    --checkpoint /disk1/yolockpts/run-tflite-vgg/model.ckpt-500
```

And then use this command to generate a tflite model
(or the API in *main.py*, if you are using tensorflow-1.9)

```
 toco --input_file=/tmp/mymodels/model_frozen.pb        \
    --output_file=/tmp/mymodels/converted_model.tflite  \
    --input_format=TENSORFLOW_GRAPHDEF                  \
    --output_format=TFLITE                              \
    --input_shape=1,320,320,3                           \
    --input_array=input_images                          \
    --output_array=output_num_array                     \
    --inference_type=FLOAT                              \
    --input_data_type=FLOAT
```

But note that:

 1. Currently `toco` does not support batch normalization correctly
 (see [this issue](https://github.com/tensorflow/tensorflow/issues/15336)
  and [this issue](https://github.com/tensorflow/tensorflow/issues/17684))
  So to convert a model to tflite, it cannot have batch normalization.
  Therefore, we can only use VGG and VX for backbone and remove batch
  normalization code from other convolutional layers.

 2. Currently (for tensorflow 1.6 that we are using) there are lots of
 operators not supported by `toco`, including

 > CAST, ExpandDims, FLOOR, Fill, Pow, RandomUniform, SPLIT, Stack, TensorFlowGreater, TensorFlowMaximum, TensorFlowMinimum, TensorFlowShape, TensorFlowSum, TensorFlowTile

 which are heavily used in our code. We have filed a
 [issue](https://github.com/tensorflow/tensorflow/issues/20110) for that.
 Please track that issue for future progress.
