#! /usr/bin/env python3
""" Service II: Lung Segmentation Preprocess """

import sys
import os
import pickle
import numpy as np
import traceback
import torch
import time
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from skimage.morphology import convex_hull_image

from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_fill_holes
from multiprocessing.pool import ThreadPool
import SimpleITK as sitk

from config import MODEL_PATH,DCM_PATH
from lung_seg_net import unet_model
from Lobeseg.lobesegment import LobeSegmentor
from Lobeseg.lobesegment_HC import LobeSegmentor_HC
from deploy.AirwaySeg import AirwaySegmentor
from redis_control import get_service_2, notify_service_3, mark_error, get_signal
from preprocess_lfz import preprocess_lfz,preprocess_csh
NUM_OF_THREADS = 16


def resample(imgs, spacing, new_spacing, order=2):
    # import pdb; pdb.set_trace()
    spacing = np.array(spacing)
    if len(imgs.shape) == 3:
        # import pdb; pdb.set_trace()
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        # import pdb; pdb.set_trace()
        imgs = zoom(imgs, resize_factor, mode="nearest", order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError("wrong shape")
    pass


def chunkIt(seq, num):
    """
    Cut the list of slices into chunks to put into multi-threading functions
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg
    return out


def process_mask_single_thread(idx, mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(convex_mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = binary_fill_holes(mask1)
        else:
            mask2 = binary_fill_holes(mask1)
        convex_mask[i_layer] = mask2
    return {"idx": idx, "data": convex_mask}


def process_mask(mask):
    # import pdb; pdb.set_trace()
    convex_mask = []
    diff_threads_data = chunkIt(mask, NUM_OF_THREADS)
    pool = ThreadPool(processes=NUM_OF_THREADS)
    results = []
    for i in range(len(diff_threads_data)):
        results.append(
            pool.apply_async(process_mask_single_thread, args=(i, diff_threads_data[i]))
        )
    pool.close()
    pool.join()
    for r in results:
        res = r.get()
        for item in res["data"]:
            convex_mask.append(item)
    convex_mask = np.asarray(convex_mask, dtype=np.uint8)
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


class LungValDataset(data.Dataset):
    def __init__(self, jpgs_cube):
        # jpgs_cube = np.load(npy_file)
        self.files = []
        for idx, jpg in enumerate(jpgs_cube):
            self.files.append({"img": jpg, "idx": idx})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        dataFile = self.files[index]
        img = dataFile["img"]
        idx = dataFile["idx"]
        return np.asarray([img]), idx


def mask2original_single_thread(idx, mask):
    ret_mask = np.zeros((mask.shape[0], 512, 512))
    for slice_idx, slicee in enumerate(mask):
        im = Image.fromarray(slicee).resize((512, 512))
        ret_mask[slice_idx] = np.array(im)
    return {"idx": idx, "data": ret_mask.astype(np.bool)}


def mask2original(resized_mask):
    original_mask = []
    diff_threads_data = chunkIt(resized_mask, NUM_OF_THREADS)
    pool = ThreadPool(processes=NUM_OF_THREADS)
    results = []
    for i in range(len(diff_threads_data)):
        results.append(
            pool.apply_async(
                mask2original_single_thread, args=(i, diff_threads_data[i])
            )
        )
    pool.close()
    pool.join()
    for r in results:
        res = r.get()
        for item in res["data"]:
            original_mask.append(item)
    return np.asarray(original_mask, dtype=np.bool)


def get_left_and_right_lung(mask_cube):
    left_lung = np.zeros(mask_cube.shape)
    right_lung = np.zeros(mask_cube.shape)
    for idx, resized_mask in enumerate(mask_cube):
        left_resized_mask = np.zeros_like(resized_mask)
        right_resized_mask = np.zeros_like(resized_mask)
        label_image = label(resized_mask)
        for r in regionprops(label_image):
            the_label = r.label
            com = ndi.measurements.center_of_mass(label_image == the_label)[1]
            if com > 128:
                right_resized_mask[label_image == the_label] = 1
            else:
                left_resized_mask[label_image == the_label] = 1
        left_lung[idx] = left_resized_mask
        right_lung[idx] = right_resized_mask
    return left_lung, right_lung


def two_lung_only(Mask):
    thresh = 0.25
    z_length = Mask.shape[0]
    ret_mask = np.zeros(Mask.shape)
    label_image = label(Mask)
    area_obj = {}
    areas = []
    for r in regionprops(label_image):
        area_obj[r.area] = r.label
        areas.append(r.area)
    areas.sort(reverse=True)
    rate = 0
    left_lung = np.zeros(Mask.shape)
    right_lung = np.zeros(Mask.shape)
    if len(areas) > 1:
        areas = areas[0:2]
        rate = areas[1] / areas[0]
        com0 = ndi.measurements.center_of_mass(label_image == area_obj[areas[0]])
        com1 = ndi.measurements.center_of_mass(label_image == area_obj[areas[1]])
        if (
            rate > thresh
            and abs(com0[0] - com1[0]) < z_length * 0.3
            and abs(com0[1] - com1[1]) < 50
        ):
            # ret_mask[label_image == area_obj[areas[0]]] = 1
            # ret_mask[label_image == area_obj[areas[1]]] = 1
            left_lung[label_image == area_obj[min(areas)]] = 1
            right_lung[label_image == area_obj[max(areas)]] = 1
        else:
            ret_mask[label_image == area_obj[areas[0]]] = 1
            left_lung, right_lung = get_left_and_right_lung(ret_mask)
    else:
        ret_mask[label_image == area_obj[areas[0]]] = 1
        left_lung, right_lung = get_left_and_right_lung(ret_mask)
    return left_lung, right_lung


def generate_lung_mask(sliceim, spacing, origin, path):
    sliceim_binary = sliceim
    output_img = sitk.GetImageFromArray(sliceim_binary.astype(np.ubyte), isVector=False)
    spacing = spacing
    output_img.SetSpacing(spacing)
    output_img.SetOrigin(origin)
    sitk.WriteImage(output_img, path,True)


def preprocess_csh_model(model, case_id, img, spacing, lum_trans_512):

    # im = np.load('256_jpgs/256_' + case_id + '.npy')
    im = img
    lung_val_loader = data.DataLoader(LungValDataset(im), batch_size=8, shuffle=True)

    # model = torch.load('023.pth').cuda()

    # mask1 = np.load('m1.npy')
    # mask2 = np.load('m2.npy')
    # mask = np.zeros_like(mask1)
    # mask[mask1 == 1] = 1
    # mask[mask2 == 1] = 1

    shape = im.shape

    # left_result = np.zeros((shape[0], 512, 512))
    # right_result = np.zeros((shape[0], 512, 512))
    resized_left_result = np.zeros(shape)
    resized_right_result = np.zeros(shape)
    result = np.zeros((shape[0], 512, 512))

    Mask = np.zeros(shape)

    # start = time.time()

    for imgs, idxes in tqdm(lung_val_loader):
        # imgs = imgs.cuda().float()
        imgs = imgs.float().cuda()
        preds = model(imgs)
        preds = preds > 0.5
        preds = preds.cpu().numpy()
        # first_preds = preds[:,0,:,:]
        # second_preds = preds[:,1,:,:]
        # import pdb; pdb.set_trace()
        for batch_idx, idx in enumerate(idxes):
            resized_mask = preds[batch_idx][0]
            Mask[idx] = resized_mask

    # Mask = keep_largest_region(Mask)

    # print("step1_python: ", time.time() - start)

    # resized_left_result = getLargestCC(resized_left_result)
    # resized_right_result = getLargestCC(resized_right_result)

    # import pdb; pdb.set_trace()

    # to_origin_start = time.time()
    # resized_left_result, resized_right_result = get_left_and_right_lung(Mask)
    resized_left_result, resized_right_result = two_lung_only(Mask)
    m1 = mask2original(resized_left_result)
    m2 = mask2original(resized_right_result)

    Mask = m1 + m2
    # save_mask = Mask > 0

    # np.save('BC_Mask_csh/' + case_id, Mask > 0)
    generate_lung_mask(Mask > 0, spacing[::-1], origin, DCM_PATH  + case_id + '/segmentation_lung.mha')

    resolution = np.array([1, 1, 1])

    newshape = np.round(np.array(Mask.shape) * spacing / resolution)
    # print('Mask.shape', Mask.shape)
    xx, yy, zz = np.where(Mask)
    box = np.array(
        [[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]]
    )
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype("int")
    margin = 5
    extendbox = np.vstack(
        [
            np.max([[0, 0, 0], box[:, 0] - margin], 0),
            np.min([newshape, box[:, 1] + 2 * margin], axis=0).T,
        ]
    ).T
    extendbox = extendbox.astype("int")

    # process_mask_start = time.time()

    dm1 = process_mask(m1)
    dm2 = process_mask(m2)

    # print("Dilation Mask: ", time.time() - process_mask_start)

    dilatedMask = dm1 + dm2

    extramask = dilatedMask ^ (Mask > 0)

    bone_thresh = 210
    pad_value = 170

    # sliceim = np.load(path_512 + case_id + '.npy', allow_pickle=True).tolist()['img']
    sliceim = lum_trans_512
    sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype("uint8")
    bones = sliceim * extramask > bone_thresh
    sliceim[bones] = pad_value

    sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
    sliceim2 = sliceim1[
        extendbox[0, 0] : extendbox[0, 1],
        extendbox[1, 0] : extendbox[1, 1],
        extendbox[2, 0] : extendbox[2, 1],
    ]
    sliceim = sliceim2[np.newaxis, ...]
    sliceim = np.asarray(sliceim, dtype=np.uint8)
    # print("extend_box: ", extendbox)
    return sliceim, extendbox, spacing, result.shape, Mask


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
    gpu = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    lobeSegmentor = LobeSegmentor(MODEL_PATH + 'Lobe_Pls_Net_3D_cp_CE_g12_0.0589_CP51.pth')
    lobeSegmentor_HC = LobeSegmentor_HC(MODEL_PATH + 'lobe_HC.pth')
    #airwaySegmentor = AirwaySegmentor(model_path = MODEL_PATH + 'airway_tree_best_model.pth')
    flag = 0
    
    
    while flag == 0:
        try:
            signal = get_signal()
            case_id = get_service_2()
            if case_id is not None:
                print("Service II: ", case_id)
                #test
                spacing = np.load(os.path.join("/tmp/data", case_id, "spacing.npy"))
                origin = np.load(os.path.join("/tmp/data", case_id, "origin.npy"))
                original_image = np.load(os.path.join("/tmp/data", case_id, "original_image.npy"))
                path = os.path.join(DCM_PATH, case_id)
                
                
                #get lung mask
                itkimage = sitk.ReadImage(os.path.join(DCM_PATH, case_id, "segmentation_lung.mha"))
                lung = sitk.GetArrayFromImage(itkimage)
                
                
                
                
                print('begin lobe seg')
                if 'BC' in case_id:
                    lobe_mask = lobeSegmentor(original_image,lung,spacing[::-1],origin,path)
                else:
                    lobe_mask = lobeSegmentor_HC(original_image,lung,spacing[::-1],origin,path)
                print('complete lobe seg')
                notify_service_3(case_id)
                #airway seg
                #try:
                #    airwaySegmentor(original_image, lung, spacing[::-1], origin, path)
                #except:
                #    pass
                
               
                
            elif signal != "0":
                time.sleep(0.5)
            else:
                flag = 1
        except Exception:
            exec_str = traceback.format_exc()
            mark_error(case_id, exec_str, "2")
