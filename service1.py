#! /usr/bin/env python3
""" Service I: Transform DCM Images to LumTrans Numpy Array and Three Order Image Resampling """

import os,sys
import time
import pickle
import numpy as np
import SimpleITK as sitk
from redis_control import (
    get_service_1,
    notify_service_2,
    notify_resampling_done,
    mark_error,
    get_signal,
)
from scipy.ndimage.interpolation import zoom
from PIL import Image
import traceback
from preprocess_lfz import preprocess_lfz,preprocess_csh
#from Lobeseg.lobesegment import LobeSegmentor
from config import DCM_PATH
def lumTrans(img):
    """
    Normalize image intensity to [0, 255]
    """
    lungwin = np.array([-1200.0, 600.0])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype("uint8")
    return newimg


def generate_image_256(img):
    """
    Generate the corresponding resized image of the input image
    """
    new_img = np.zeros((img.shape[0], 256, 256))
    for idx, slicee in enumerate(img):
        im = Image.fromarray(slicee).resize((256, 256))
        new_img[idx] = np.array(im, dtype=np.uint8)
    return new_img


def resample_yl(image, spacing, new_spacing=[1, 1, 1]):
    """
    Resampling / Spline Interpolation with order 3
    """
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = zoom(image, real_resize_factor, mode="nearest")
    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode="nearest")
    return image, new_spacing, real_resize_factor


def generate_image_array(case_id):
    """
    Generate the image array and notify the append the value in redis
    """
    path = DCM_PATH + case_id
    tmp_path = os.path.join("/tmp/data", case_id)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    tmp_spacing = image.GetSpacing()
    spacing = np.stack([tmp_spacing[-i] for i in range(1, 4)])
    origin = image.GetOrigin()
    image_array = sitk.GetArrayFromImage(image)
    image_512 = lumTrans(image_array)
    image_256 = generate_image_256(image_512)
    
    
    np.save(os.path.join(tmp_path, "original_image.npy"), image_array)
    np.save(
        os.path.join(tmp_path, "lumTrans_512.npy"),
        np.asarray(image_512, dtype=np.uint8),
    )
    np.save(
        os.path.join(tmp_path, "lumTrans_256.npy"),
        np.asarray(image_256, dtype=np.uint8),
    )
    np.save(os.path.join(tmp_path, "spacing.npy"), np.asarray(spacing))
    np.save(os.path.join(tmp_path, "origin.npy"), np.asarray(origin))
    #notify_service_2(case_id)
    # print("lumTrans finished")
 
    fter = sitk.IntensityWindowingImageFilter()
    fter.SetWindowMaximum(500)
    fter.SetWindowMinimum(-1300)
    itk_image = fter.Execute(image)
    ct_scan = sitk.GetArrayFromImage(itk_image)

    ct_scan, _, resize_factor  = resample_yl(image_array, spacing)
    # resampeld = {
    #     'ct_scan': ct_scan,
    #     'resize_factor': resize_factor
    # }
    np.save(os.path.join(tmp_path, "resampled.npy"), ct_scan)
    np.save(os.path.join(tmp_path, "resize_factor.npy"), resize_factor)
    #benmal data
    #fter.SetWindowMaximum(400)
    #fter.SetWindowMinimum(-1000)
    #itk_image = fter.Execute(image)
    #benmal_ct_scan = sitk.GetArrayFromImage(itk_image)    
    #np.save(os.path.join(tmp_path,"benmal.npy"),benmal_ct_scan)
    notify_resampling_done(case_id)
    # pickle.dump(resampled, )
    # import pdb; pdb.set_trace()

def preprocess(case_id):
    """
    Preprocess function
    """
    #image_512 = np.load(os.path.join("/tmp/data", case_id, "lumTrans_512.npy"))
    #image_256 = np.load(os.path.join("/tmp/data", case_id, "lumTrans_256.npy"))
    #spacing = np.load(os.path.join("/tmp/data", case_id, "spacing.npy"))
    try:
        sliceim, extendbox, spacing, shape, Mask = preprocess_lfz(DCM_PATH + case_id, need_shape = True)
    except:
        sliceim, extendbox, spacing, shape, Mask = preprocess_csh(DCM_PATH + case_id, need_shape = True)
    # import pdb; pdb.set_trace()
    obj = {}
    obj["sliceim"] = sliceim
    obj["extendbox"] = extendbox
    obj["spacing"] = spacing
    obj["shape"] = shape

    with open(os.path.join("/tmp/data", case_id, "preprocessed.pkl"), "wb") as handle:
        pickle.dump(obj, handle)
    #notify_service_3(case_id)
    return Mask

if __name__ == "__main__":
    flag = 0
    
    gpu = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    while flag == 0:
        try:
            signal = get_signal()
            if signal == "1":
                case_id = get_service_1()
                if case_id is not None:
                    print("Service I: ", case_id)
                    generate_image_array(case_id)
                    lung = preprocess(case_id)
                    
                    notify_service_2(case_id)
                else:
                    time.sleep(0.5)
            elif signal == "0":
                flag = 1
            else:
                time.sleep(1)

        except Exception:
            exec_str = traceback.format_exc()
            mark_error(case_id, exec_str, "1")
