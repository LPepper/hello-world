#! /usr/bin/env python3
""" Service IV: Nodule Classification """

import sys
import os
import numpy as np
import time
import torch
import pickle
import traceback

import huaxi_morph_classifier as new_classifier
import huaxi_nodule
import classifier
from nodule import to_objects

from config import MODEL_PATH
import math
from redis_control import (
    get_service_4,
    notify_service_6,
    mark_error,
    get_resampling_status,
    del_resampling_status,
    get_signal,
)


class StuckingError(Exception):
    pass


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_nodule_matrix(nodule, image, factor = None, z_size = 64, roi_size = 64):
    """
    Get the matrix of values for every nodule
    """
    z, y, x = nodule.to_std_coordinate(factor)
    shape = np.shape(image)
    axial_scale = int(roi_size / 2)
    z_scale = int(z_size / 2)
    xmin = x - axial_scale
    ymin = y - axial_scale
    zmin = z - z_scale
    z_pad, y_pad, x_pad = [0, 0], [0, 0], [0, 0]
    if xmin < 0:
        x_pad[0] = abs(xmin)
        xmin = 0
        # print('out of bound!')
    elif x + axial_scale > shape[2]:
        x_pad[1] = abs(x + axial_scale - shape[2])
        xmin = shape[2] - 2 * axial_scale
        # print('out of bound!')
    if ymin < 0:
        y_pad[0] = abs(ymin)
        ymin = 0
        # print('out of bound!')
    elif y + axial_scale > shape[1]:
        y_pad[1] = abs(y + axial_scale - shape[1])
        ymin = shape[1] - 2 * axial_scale
        # print('out of bound!')
    if zmin < 0:
        z_pad[0] = abs(zmin)
        zmin = 0
        # print('out of bound!')
    elif z + z_scale > shape[0]:
        z_pad[1] = abs(z + z_scale - shape[0])
        zmin = shape[0] - 2 * z_scale
        # print('out of bound!')
    if z_pad != [0, 0] or y_pad != [0, 0] or x_pad != [0, 0]:
        padding = [z_pad, y_pad, x_pad]
        image = np.pad(image, pad_width=padding, mode="constant", constant_values=0)
        # print('after padding image size:',image.shape, padding)
    nodule_roi = image[
        zmin : zmin + 2 * z_scale,
        ymin : ymin + 2 * axial_scale,
        xmin : xmin + 2 * axial_scale,
    ]

    return nodule_roi


def get_rois(case_id, nodules, spacing, roi_size = 64 ,is_del = True):
    """
    Get Rois for nodules
    """
    # while not os.path.exists(os.path.join('/tmp/data', case_id, 'resampled.npy')):
    #     time.sleep(1) # wait util the resampled file is generated in service 1
    start_point = time.time()
    while get_resampling_status(case_id) is None or not os.path.exists(
        os.path.join("/tmp/data", case_id, "resampled.npy")
    ):
        current_time = time.time()
        if current_time - start_point > 100:
            raise Exception(
                "Stucking for more than 100 seconds, \
                            move to the next one"
            )
    ct_scan = np.load(os.path.join("/tmp/data", case_id, "resampled.npy"))
    benmal_ct_scan = np.load(os.path.join("/tmp/data", case_id, "original_image.npy"))
    resize_factor = np.load(os.path.join("/tmp/data", case_id, "resize_factor.npy"))
    datasets = []
    for nodule in nodules:
        matrix = get_nodule_matrix(nodule, ct_scan, resize_factor, z_size = roi_size, roi_size = roi_size)
        matrix2 = get_nodule_matrix(nodule, benmal_ct_scan, z_size = 16, roi_size = 64)
        datasets.append({"matrix": matrix, "benmal" : matrix2, "object": nodule})
    if is_del:
        del_resampling_status(case_id)
    return datasets


def classify_nodules_new(case_id, raw_nodules, spacing, ln_classifier):
   
    
    #nodules_for_mal = to_objects(raw_nodules.copy())
    #rois_nodules_for_mal = get_rois(case_id, nodules_for_mal, spacing, False)
    #rois_nodules_for_mal = mal_classifier(rois_nodules_for_mal)
    #nodules_mal = [item["object"] for item in rois_nodules_for_mal]

    nodules = huaxi_nodule.to_nodule_objects(raw_nodules)
    rois_nodules = get_rois(case_id, nodules, spacing)
    nodules = ln_classifier(rois_nodules)
    
    all_nodules_obj = []

     
    for idx, nodule in enumerate(nodules):
        morph = nodule.get_morph()
        spi = morph['spi'][0]
        lob = morph['lob'][0]
        pin = morph['pin'][0]
        
        cav = morph['cav'][0]
        vss = morph['vss'][0]
        bea = morph['bea'][0]
        cal = morph['cal'][0]
        bro = morph['bro'][0]
        den = morph['den'][0]
        mal = 0
        malProb = float(morph['benmal'][1])
        if malProb < 0.3:
            mal = 0
        elif malProb > 0.8:
            mal = 2
        else:
            mal = 1

        nodule_obj = {
            "probability": sigmoid(raw_nodules[idx][0]),
            "malignancy": mal + 1,
            "calcification": cal + 1,
            "spiculation": spi + 1,
            "texture": int(den) + 1,
            "lobulation": lob + 1,
            "pin" : pin + 1,
            "cav" : cav + 1,
            "vss" : vss + 1,
	    "bea" : bea + 1,
            "bro" : bro + 1,
            "malProb": float(morph['benmal'][1]),
            "calProb": float(morph['cal'][1]),
            "spiProb": float(morph['spi'][1]),
            "texProb": float(morph['den'][1]),
            "lobProb": float(morph['lob'][1]),
            "pinProb": float(morph['pin'][1]),
            "cavProb": float(morph['cav'][1]),
            "vssProb": float(morph['vss'][1]),
            "beaProb": float(morph['bea'][1]),
            "broProb": float(morph['bro'][1]),
        }

        all_nodules_obj.append(nodule_obj)
    with open(os.path.join("/tmp/data", case_id, "classify.pkl"), "wb") as pkl:
        pickle.dump(all_nodules_obj, pkl)


        
if __name__ == "__main__":
    gpu = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    #classifier_model_root = "/home/preprocess/db_dev/models"
    #args1 = classifier.Args(classifier_model_root, 'malignancy', gpu=gpu)
    #mal_classifier = classifier.LNClassifier(args1)


    morph_model = MODEL_PATH + 'morph.ckpt'
    den_model = MODEL_PATH + 'den.ckpt'
    sample_sizes = {'small':24,'middle':32,'large':48}
    sample_durations =  {'small':12,'middle':16,'large':32} 
 
    args2 = new_classifier.Args(morph_model, den_model, sample_sizes,sample_durations, gpu=gpu)

    ln_classifier = new_classifier.LNClassifier(args2)

    flag = 0
    while flag == 0:
        try:
            signal = get_signal()
            case_id = get_service_4()
            if case_id is not None:
                print("Service IV: ", case_id)
                nodules = np.load(os.path.join("/tmp/data", case_id, "nodules.npy"))
                spacing = np.load(os.path.join("/tmp/data", case_id, "spacing.npy"))
                classify_nodules_new(case_id, nodules, spacing, ln_classifier)
                if os.path.exists(os.path.join("/tmp/data", case_id, "segment.pkl")):
                    notify_service_6(case_id)
            elif signal != "0":
                time.sleep(0.5)
            else:
                flag = 1
        except Exception:
            exec_str = traceback.format_exc()
            mark_error(case_id, exec_str, "4")
