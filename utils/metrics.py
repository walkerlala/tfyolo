#!/usr/bin/python
#coding:utf-8

"""Utils for calculating object-detection metrics (e.g., mAP)"""

from collections import namedtuple

def map_batch(op_batch, gt_batch, infer_threshold):
    """ TODO only for one class. """
    Bbox = namedtuple("Bbox", ["confidence", "GT"])
