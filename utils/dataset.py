#coding:utf-8

""" Contains some utils for reading/writing dataset.

Note that these utils are still very primitive, and contain many network
architecture-dependent things, such as the layout of ground truth labels.
"""

import cv2
import math
import os
import pdb
import random
import numpy as np

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
        self._images_list_iter = iter(self._images_list)

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

    def get_class_names(self):
        return self.label_classname.values()

    def _get_cell_ij(self, x, y, image_size):
        """Get corresponding cell x/y coordinates of @gt_box (which cell it
        belongs in the final feature map) """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        num_box = image_size / 32
        percent_per_box = 1.0/num_box
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
                                for j in range(feature_map_len)]

        gt_bnxs[cell_i][cell_j].append(box)
        return gt_bnxs

    def _pack_and_flatten_gt_box(self, gt_bnxs, image_size, num_gt_bnx):
        feature_map_len = image_size/32
        for i in range(feature_map_len):
            for j in range(feature_map_len):
                num_box = len(gt_bnxs[i][j])
                assert num_box <= num_gt_bnx, \
                        "Number of box[%d] exceed in cell[%d][%d] \
                         (with num_gt_bnx[%d]). \
                         Consider incresing FLAGS.num_gt_bnx." \
                         % (num_box, i, j, num_gt_bnx)
                for k in range(num_gt_bnx-num_box):
                    gt_bnxs[i][j].append([0,0,0,0,0])
        gt_bnxs = np.array(gt_bnxs).reshape(
                    (feature_map_len, feature_map_len, 5*num_gt_bnx)
                  )
        return gt_bnxs

    @staticmethod
    def get_outscale(image_size):
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

    def next_batch(self, batch_size=50, image_size=320, num_gt_bnx=20,
                   normalize_image=True, infinite=True):
        """ Return next batch of images.

          Args:
            batch_size: Number of images to return.
            image_size: Size of image. If the loaded image is not in this size,
                then it will be resized to [image_size, image_size, 3].
            num_gt_bnx: See FLAGS.num_gt_bnx.
            normalize_image: To or to not normalize the image (to [0, 1]).
            infinite: Whether or not to loop all images infinitely.

          Return:
            batch_xs: A batch of image, i.e., a numpy array in shape
                      [batch_size, image_size, image_size, 3]
            batch_ys: A batch of ground truth bounding box value.
                It is in shape
                    [batch_size, image_size/32, image_size/32, 5*num_gt_bnx]
            outscale: Scale information (of x/y coordinates) for this batch.
        """
        assert image_size % 32 == 0, "image_size should be multiple of 32"
        cell_size = 32

        batch_xs_filename = []
        for _ in range(batch_size):
            filename = next(self._images_list_iter, None)
            if not filename:
                if not infinite:
                    if not len(batch_xs_filename):
                        return ([], [], None)
                    elif len(batch_xs_filename):
                        batch_size = len(batch_xs_filename)
                        break
                self._images_list_iter = iter(self._images_list)
                filename = next(self._images_list_iter)
            batch_xs_filename.append(filename)
        random.shuffle(batch_xs_filename)

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
                # TODO I don't know what exactly is the 2nd parameter for
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
                if not (x<=1 and y<=1 and w<=1 and h<=1):
                    print("WRONG DATA. [x,y,w,h]: %s, [_x,_y,_w,_h]: %s, y_path:%s" % (
                          str([x,y,w,h]), str([_x,_y,_w,_h]), y_path))
                    continue
                if not (w>0 and h > 0):
                    print("WRONG DATA. w&h must > 0. w&h: %s, y_path:%s" % (
                          str([w,h]), y_path))
                    continue
                if x==0 or y==0:
                    print("WARNING: x|y == 0, x&y: %s, y_path:%s" % (str[x, y], y_path))

                cell_i, cell_j = self._get_cell_ij(x, y, image_size)
                # adjust coordinates to be relative to their corresponding cell
                x *= image_size
                x %= cell_size
                x /= cell_size
                y *= image_size
                y %= cell_size
                y /= cell_size

                box = [label, x, y, w, h]
                gt_bnxs = self._append_gt_box(
                            gt_bnxs,
                            box,
                            image_size,
                            cell_i,
                            cell_j
                        )
                assert cell_i<image_size/32 and cell_j<image_size/32, \
                       "cell_i/j too large"

            if not len(gt_bnxs):
                print("WARNING: image[{}] has no label!".format(x_path))
                gt_bnxs = [[[] for i in range(int(image_size/32))]
                                for j in range(int(image_size/32))]

            gt_bnxs = self._pack_and_flatten_gt_box(gt_bnxs, image_size,
                                                    num_gt_bnx)
            batch_ys.append(gt_bnxs)
            yfile.close()

        batch_xs = np.array(batch_xs)
        batch_ys = np.array(batch_ys)

        batch_xs_shape = (batch_size, image_size, image_size, 3)
        assert batch_xs.shape == batch_xs_shape, \
                "batch_xs shape mismatch. shape: %s, expected: %s" \
                  % (str(batch_xs.shape), str(batch_xs_shape))

        batch_ys_shape = (batch_size, image_size/32,
                          image_size/32, 5*num_gt_bnx)
        assert batch_ys.shape == batch_ys_shape, \
                "batch_ys shape mismatch. shape: %s, expected: %s" \
                  % (str(batch_ys.shape), str(batch_ys_shape))

        outscale = DatasetReader.get_outscale(image_size)

        return batch_xs, batch_ys, outscale

class ImageHandler():
    """Image handler for testing."""

    def __init__(self, multiple_images, infile):
        self._images = []
        self._names = []

        if not multiple_images:
            self._images.append(infile)
            self._names.append(os.path.basename(infile))
        else:
            with open(infile, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._images.append(line)
                        self._names.append(os.path.basename(line))
        self._iter = iter(zip(self._images, self._names))

    def next_batch(self, batch_size, scale=320):
        """Read a batch of images after scaling and normalization.

          Args:
              batch_size: Number of images to read. If there are not enough
                images, whatever number of images left will be returned (that
                is, the number of images returned may be < @batch_size.
              scale: All image will be scaled to shape of [scale, scale].
          Return:
              images: A batch of images after scaling and normalization.
              original_sizes: Size/shape of the images before scaling and
                normalization.
              outscale: Scale info for some functions to scale the prediction
                output to be relative to the whole image (rather than relative
                to the current cell).
        """
        images = []
        names = []
        original_sizes = []
        outscale = DatasetReader.get_outscale(scale)
        for _ in range(batch_size):
            image_name = next(self._iter, None)
            if not image_name:
                return (images, original_sizes, names, outscale)

            image_path, name = image_name
            if not os.path.exists(image_path):
                raise IOError("No such file: {}".format(image_path))
            orignal_image = cv2.imread(image_path)
            height, width, _ = orignal_image.shape
            original_sizes.append( (height, width) )
            im = cv2.resize(
                    orignal_image,
                    dsize=(scale, scale),
                    interpolation=cv2.INTER_LINEAR
                 )
            # normalize image
            # TODO I don't know what is the 2nd parameter for
            image = cv2.normalize(np.asanyarray(im, np.float32), np.array([]),
                                  alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            images.append(image)
            names.append(name)

        return (images, original_sizes, names, outscale)

    def _scale_final_images(self, images, batch_xs_scale_info):
        """Un-scale and un-normalize images."""
        result = []
        for image, height_width in zip(images, batch_xs_scale_info):
            image = image * 255
            height, width = height_width
            # NOTE: whereas `im.shape' return shape as a
            # "(height, width, channels)" tuple, the "dsize" parameter comes
            # with width first! Yet another example of inconsistent program
            # interface!!!!
            image = cv2.resize(
                       image,
                       dsize=(width, height),
                        interpolation=cv2.INTER_LINEAR
                    )
            result.append(image)
        return result

    def write_batch(self, final_images, batch_xs_scale_info, batch_xs_names,
                    multiple_images, outdir, outfile):
        """Write a batch of image. If there is only one image in this batch,
        use @outfile as the output path. If there are multiple images in this
        batch, use @outdir as the directory name and every prediction output
        has a name "prediction_{$original_file_basename}.jpg.

          Args:
              final_images: Output of the neural network.
              batch_xs_scale_info: Scaling info to scale @final_images to their
                original size.
              batch_xs_names: Original (base) name of @final_images.
              multiple_images: Whether or not write multiple images (i.e., write
                to a directory or a single file).
              outdir: Base name of the directory use to store multiple image
                prediction if there are multiple images in @final_images.
              outfile: Path of the output image if there is only one image in
                @final_images.
          Return:
              None.
        """
        if not len(final_images):
            print("Warning: no final images.")
            return
        final_images = self._scale_final_images(final_images, batch_xs_scale_info)
        if not multiple_images:
            if not outfile.endswith(".jpg") and not outfile.endswith(".png"):
                outfile = "{}.jpg".format(outfile)
            cv2.imwrite(outfile, final_images[0])
        else:
            for image, name in zip(final_images, batch_xs_names):
                name = os.path.basename(name)
                name = "pred_{}".format(name)
                if not name.endswith(".jpg") and not name.endswith(".png"):
                    name = "{}.jpg".format(name)
                path = os.path.join(outdir, name)
                cv2.imwrite(path, image)
