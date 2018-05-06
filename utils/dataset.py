#coding:utf-8

""" Contains some utils for reading dataset.

Note that these utils are still very primitive, and contain many network
architecture-dependent things, such as the layout of ground truth labels.
"""

import Queue
import cv2
import math
import numpy as np
import random

class DatasetReader():
    """ Reader to read images/labels """

    def __init__(self, training_file_list, class_name_file, shuffle=True):
        """
          Args:
            training_file_list: File contains paths of all training images.
            class_name_file: See FLAGS.class_name_file.
            shuffle: To or not to shuffle the dataset.

        Note that training images/labels should be in the same directory.
        """

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
                    raise ValueError(
                            "Class name file incorrect format: %s" % line
                        )
                if parts[0] in self.label_classname:
                    raise ValueError(
                            "Duplicate definition of classname: %s" % str(parts)
                        )
                self.label_classname[parts[0]] = parts[1]

    def _get_cell_ij(self, x, y, image_size):
        """Get corresponding cell x/y coordinates of @gt_box (which cell it
        belongs in the final feature map) """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        num_of_box = image_size / 32
        percent_per_box = 1.0/num_of_box
        cell_i = math.floor(x/percent_per_box)
        cell_j = math.floor(y/percent_per_box)
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

    def _get_outscale(self, image_size):
        """
        [image_size/32, image_size/32, 3]
        """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        row_num = image_size / 32
        outscale = []
        for i in range(row_num):
            outscale.append([])
            for j in range(row_num):
                outscale[i].append([])
                outscale[i][j].append(i/float(row_num))
                outscale[i][j].append(j/float(row_num))
                outscale[i][j].append(row_num)
        return outscale

    def next_batch(self, batch_size=50, image_size=320, num_of_anchor_boxes=5,
                   num_of_gt_bnx_per_cell=20, normalize_image=True,
                   shrink_coordinate_related_to_cell=True):
        """ Return next batch of images.

          Args:
            batch_size: Number of images to return.
            image_size: Size of image. If the loaded image is not in this size,
                then it will be resized to [image_size, image_size, 3].
            num_of_anchor_boxes: See FLAGS.num_of_anchor_boxes.
            num_of_gt_bnx_per_cell: See FLAGS.num_of_gt_bnx_per_cell.
            normalize_image: To or to not normalize the image (to [0, 1]).
            shrink_coordinate_related_to_cell: To or not to scale the x/y
                coordinates to be relative to their corresponding cell.
                NOTE!! There are lots of code that assumes this to True, so
                don't change that unless you know what you are doing here.
                TODO

          Return:
            batch_xs: A batch of image, i.e., a numpy array in shape
                      [batch_size, image_size, image_size, 3]
            batch_ys: A batch of ground truth bounding box value, in shape
                      [batch_size, image_size/32, image_size/32, 5*num_of_gt_bnx_per_cell]
            outscale: Scale information (of x/y coordinates) for this batch.
        """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        cell_size = 32
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

                cell_i, cell_j = self._get_cell_ij(x, y, image_size)
                if shrink_coordinate_related_to_cell:
                    x *= image_size
                    x %= cell_size
                    x /= cell_size
                    y *= image_size
                    y %= cell_size
                    y /= cell_size
                box = [label, x, y, w, h]
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

        outscale = self._get_outscale(image_size)

        return batch_xs, batch_ys, outscale

