#! /usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import time
import SimpleITK as sitk
import pickle
import traceback

from nodule_seg_net import unet_model
from redis_control import get_service_5, notify_service_6, mark_error, get_signal
from LungSegmentsSeg.segments import lobar_segment_partition
from config import MODEL_PATH,DCM_PATH,USERNAME

def get_hist_info(array_nodule):
    
    bins = list(set(array_nodule.tolist()))
    bins = sorted(bins)
    bins.append(bins[len(bins) - 1] + 1)
    n = []
    for i in range(len(bins) - 1):
        n.append(int(np.sum(array_nodule == bins[i])))
    
    return n,bins

def get_vol_hu(ct_img, pred_mask, spacing):
    # print("get vol hu start...")
    # import pdb; pdb.set_trace()
    vol = np.sum(pred_mask) * spacing[0] * spacing[1] * spacing[2] / 1000
    mul_cube = np.multiply(ct_img, pred_mask)
    hu_max = np.max(mul_cube)
    hu_mean = np.sum(mul_cube) / np.sum(pred_mask)
    hu_min = np.min(mul_cube)
    # print("get vol hu end...")
    return vol, hu_max, hu_mean, hu_min

def find_nearest_nonzero(img, TARGET):
    nonzero = np.argwhere(img > 0)
    distances = np.sqrt((nonzero[:,0] - TARGET[0]) ** 2 + (nonzero[:,1] - TARGET[1]) ** 2  + (nonzero[:,2] - TARGET[2]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

def segment_nodules(ct_image, case_id, nodules, spacing, origin, image, net):
    """
    Segmenting nodules main function
    """
    
    lobe_path = DCM_PATH + case_id + "/" + "segmentation_lobe.mha"
    itkimage = sitk.ReadImage(lobe_path)
    lobe_mask = sitk.GetArrayFromImage(itkimage)
    
    
    all_nodules = []
    for nodule_idx, nodule in enumerate(nodules):
        current_seg_cube = np.zeros_like(image)
        z = int(round(nodule[1]))
        y = int(round(nodule[2]))
        x = int(round(nodule[3]))
        slice_idx = current_seg_cube.shape[0] - z - 1
        diameter = nodule[4]
        radius = int(round(diameter / 2))
        additional_slices = int(round(radius * spacing[2] / spacing[1]))
        # print(z, y, x, radius, additional_slices)
        shape = (96, 96)
        # im = Image.fromarray(image[z])
        total_results = []
        all_coords = []
        index_start = max(0, z - additional_slices)
        index_end = min(z + additional_slices, image.shape[0] - 1)
        for idx in range(index_start, index_end):
            im = Image.fromarray(image[idx])
            patch = np.array(
                im.crop(
                    (x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4)
                )
            )
            result = np.zeros(shape)
            result[: patch.shape[0], : patch.shape[1]] = patch
            total_results.append([result])
            all_coords.append(
                [idx, x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4]
            )

        total_results = np.asarray(total_results, dtype=float)
        inputs = torch.tensor(total_results).cuda().float()

        with torch.no_grad():
            outputs = torch.sigmoid(net.forward(inputs))
            outputs = outputs > 0.5
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()

        for idx, inputt in enumerate(inputs):
            coord = all_coords[idx]
            current_seg_cube[coord[0]][
                coord[2] : coord[4], coord[1] : coord[3]
            ] = outputs[idx][0][: coord[3] - coord[1], : coord[4] - coord[2]]
        nodule_mask = current_seg_cube > 0

        generateMHA(
            nodule_mask,
            "segmentation_nodule_"+ USERNAME + '_' + str(nodule_idx + 1) + ".mha",
            DCM_PATH + case_id,
            spacing[::-1],
            origin,
        )
        vol, hu_max, hu_mean, hu_min = get_vol_hu(ct_image, nodule_mask, spacing)
        x1 = x - radius
        x2 = x + radius
        y1 = y - radius
        y2 = y + radius
        
        #get the lobe position of nodule
        lobe_pos = lobe_mask[lobe_mask.shape[0] - slice_idx,int(y),int(x)]
        if int(lobe_pos) == 0 and 'BC' in case_id:
            nearest_cood = find_nearest_nonzero(lobe_mask,(lobe_mask.shape[0] - slice_idx,int(y),int(x)))
            lobe_pos = lobe_mask[nearest_cood[0],nearest_cood[1],nearest_cood[2]]
        #get the lung segement position of nodule
        
        segment_pos = ''
        if int(lobe_pos) != 0:
            segment_pos = lobar_segment_partition(lobe_mask,[lobe_mask.shape[0] - slice_idx,int(y),int(x)],int(lobe_pos))
        
       
        #nodule hist info      
        #itkimage = sitk.ReadImage(DCM_PATH + case_id + '/segmentation_nodule_' +USERNAME + '_' + str(nodule_idx + 1) + '.mha')
        #segment_nodule_i = sitk.GetArrayFromImage(itkimage)
        
        #generate_hist_info
        array_nodule = ct_image[np.where(nodule_mask==1)]
        
        #n, bins = np.histogram(array_nodule, bins = 30)
        n, bins = get_hist_info(array_nodule)

        nodule_hist={}
        nodule_hist['n'] = n
        nodule_hist['bins'] = bins
        nodule_hist['mean'] = float(np.mean(array_nodule))
        nodule_hist['var'] = float(np.var(array_nodule))        
 
        nodule_obj = {
            "rect_no": "a" + str(nodule_idx).zfill(3),
            "patho": "",
            'place': int(lobe_pos),
            'segment':str(segment_pos),
            "slice_idx": slice_idx,
            "nodule_no": str(nodule_idx),
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,
            'diameter': max(x2 - x1, y2 - y1) * float(spacing[1]),
            "volume": float(vol),
            "huMax": int(hu_max),
            "huMean": int(hu_mean),
            "huMin": int(hu_min),
            "nodule_hist" : nodule_hist
        }
        all_nodules.append(nodule_obj)

    with open(os.path.join("/tmp/data", case_id, "segment.pkl"), "wb") as pkl:
        pickle.dump(all_nodules, pkl)


def generateMHA(pred_mask, name, seg_dir, pixelSpacing, origin):
    """
    """

    output_img = sitk.GetImageFromArray(pred_mask.astype(np.ubyte), isVector=False)
    # import pdb; pdb.set_trace()
    output_img.SetSpacing(pixelSpacing)
    output_img.SetOrigin(origin)
    sitk.WriteImage(output_img, os.path.join(seg_dir, name), True)


if __name__ == "__main__":
    gpu = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    model = unet_model.UNet(n_channels=1, n_classes=1).cuda()
    model.load_state_dict(torch.load(MODEL_PATH + "nodule_segmentation.pth"))
    flag = 0
    while flag == 0:
        try:
            signal = get_signal()
            case_id = get_service_5()
            if case_id is not None:
                print("Service V: ", case_id)
                nodules = np.load(os.path.join("/tmp/data", case_id, "nodules.npy"))
                spacing = np.load(os.path.join("/tmp/data", case_id, "spacing.npy"))
                origin = np.load(os.path.join("/tmp/data", case_id, "origin.npy"))                
                image_512 = np.load(
                    os.path.join("/tmp/data", case_id, "lumTrans_512.npy")
                )
                ct_image = np.load(
                    os.path.join("/tmp/data", case_id, "original_image.npy")
                )
                segment_nodules(
                    ct_image, case_id, nodules, spacing, origin, image_512, model
                )
                if os.path.exists(os.path.join("/tmp/data", case_id, "classify.pkl")):
                    notify_service_6(case_id)
            elif signal != "0":
                time.sleep(0.5)
            else:
                flag = 1
        except Exception:
            exec_str = traceback.format_exc()
            mark_error(case_id, exec_str, "5")
