#! /usr/bin/env python3
""" Service III: Nodule Detection """

import os
import time
import sys
import glob
import traceback

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from config import DCM_PATH,MODEL_PATH
import pickle
import numpy as np
import SimpleITK as sitk

from res18 import *
from split_combine import SplitComb
from postprocess import *
from nodule import to_objects
import classifier
import scipy.ndimage
from PIL import Image
from redis_control import (
    get_service_3,
    notify_service_4,
    notify_service_5,
    mark_error,
    get_signal,
)



class Args(object):
    """
    Generate an Args object
    """

    def __init__(
        self, resume_checkpoint, gpu=None, margin=32, sidelen=64, batch_size=1
    ):
        self.resume_checkpoint = resume_checkpoint
        self.gpu = gpu
        self.margin = margin
        self.sidelen = sidelen
        self.batch_size = batch_size
        # self.method = method

    pass


class LNDetector(object):
    """
    The DeepLN Detector
    """

    def __init__(self, args):
        self.args = args
        # self.method = args.method
        self.config, self.net, self.loss, self.get_pbb = get_model()
        checkpoint = torch.load(args.resume_checkpoint)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.split_comber = SplitComb(
            self.args.sidelen,
            config["max_stride"],
            config["stride"],
            self.args.margin,
            config["pad_value"],
        )
        if args.gpu is not None:
            self.device = torch.device("cuda")
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        pass

    def __call__(self, filename):
        # load_pkl_file = time.time()
        pkl_file = os.path.join("/tmp/data", filename, "preprocessed.pkl")
        with open(pkl_file, "rb") as handle:
            obj = pickle.load(handle)

        slicelim = obj["sliceim"]

        extend_box = obj["extendbox"]
        spacing = obj["spacing"]
        img_shape = obj["shape"]

        stride = config["stride"]
        pad_value = config["pad_value"]
        imgs = slicelim.copy()
        nz, nh, nw = imgs.shape[1:]
        pz = int(np.ceil(float(nz) / stride)) * stride
        ph = int(np.ceil(float(nh) / stride)) * stride
        pw = int(np.ceil(float(nw) / stride)) * stride
        imgs = np.pad(
            imgs,
            [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]],
            "constant",
            constant_values=pad_value,
        )

        xx, yy, zz = np.meshgrid(
            np.linspace(-0.5, 0.5, int(imgs.shape[1] / stride)),
            np.linspace(-0.5, 0.5, int(imgs.shape[2] / stride)),
            np.linspace(-0.5, 0.5, int(imgs.shape[3] / stride)),
            indexing="ij",
        )
        coord = np.concatenate(
            [xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0
        ).astype("float32")
        imgs, nzhw = self.split_comber.split(imgs)
        coord2, nzhw2 = self.split_comber.split(
            coord,
            side_len=self.split_comber.side_len / stride,
            max_stride=self.split_comber.max_stride / stride,
            margin=self.split_comber.margin / stride,
        )
        assert np.all(nzhw == nzhw2), "split imgs not equal coords"
        imgs = torch.from_numpy((imgs.astype(np.float32) - 128) / 128)
        coord2 = torch.from_numpy(coord2)
        nzhw = np.array(nzhw)

        nodes = self.detect(imgs, coord2, nzhw, extend_box, spacing, isfliped=False)

        return nodes, img_shape, spacing

    def detect(self, imgs, coord2, nzhw, extend_box, spacing, isfliped=False):
        net = self.net
        net.eval()
        outputlist = []
        # batch_size = 2
        # start = time.time()
        num_batches = int(imgs.shape[0] / self.args.batch_size)
        num_pass = (
            num_batches
            if num_batches * self.args.batch_size == imgs.shape[0]
            else num_batches + 1
        )
        # start_num_pass = time.time()
        for i in range(num_pass):
            end_idxs = min((i + 1) * self.args.batch_size, imgs.shape[0])
            if self.args.gpu is not None:
                input_x = imgs[i * self.args.batch_size : end_idxs].to(self.device)
                # input_coord = Variable(coord2[i*self.args.batch_size:end_idxs]).cuda()
                input_coord = coord2[i * self.args.batch_size : end_idxs].to(
                    self.device
                )
                output = net(input_x, input_coord)
                outputlist.append(output.data.cpu().numpy())
            else:
                input_x = Variable(imgs[i * self.args.batch_size : end_idxs])
                input_coord = Variable(coord2[i * self.args.batch_size : end_idxs])
                output = net(input_x, input_coord)
                outputlist.append(output.data.numpy())

        output = np.concatenate(outputlist, 0)
        output = self.split_comber.combine(output, nzhw=nzhw)
        thresh = -3
        pbb, mask = self.get_pbb(output, thresh, ismask=True)
        # if spacing[0] == 1.0:
        #     pbb1 = pbb[pbb[:,0]>3]
        # elif spacing[0] == 5.0:
        #     print("HC CT pbb")
        #     pbb1 = pbb[pbb[:,0]>3]
        pbb1 = pbb[pbb[:, 0] > 3]
        pbb1 = pbb1[np.argsort(-pbb1[:, 0])]
        # print(pbb1)
        nms_th = 0.05
        pbb1 = nms(pbb1, nms_th)
        if pbb1.shape[0] > 10:
            pbb1 = pbb1[0:10]
        # print("New in Jianyong")
        # print("The time for calculating ", time.time() - start)
        # start = time.time()
        # import pdb; pdb.set_trace()

        nodes = pbb2axis(pbb1, extend_box, spacing, isfliped=False)
        # print("The post processing ", time.time() - start)
        return nodes


if __name__ == "__main__":
    gpu = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    bc_model = MODEL_PATH + "072.ckpt"
    hc_model = MODEL_PATH + "hc.ckpt"
    args_bc = Args(bc_model, gpu=gpu, batch_size=1)
    args_hc = Args(hc_model, gpu=gpu, batch_size=1)
    bc_detector = LNDetector(args_bc)
    hc_detector = LNDetector(args_hc)
    flag = 0
    while flag == 0:
        try:
            signal = get_signal()
            case_id = get_service_3()
            if case_id is not None:
                print("Service III: ", case_id)
                if "BC" in case_id:
                    nodules, reference_shape, spacing = bc_detector(case_id)
                else:
                    nodules, reference_shape, spacing = hc_detector(case_id)
                # args1 = Args(model_path, gpu=gpu, batch_size=1)
                # ln_retector = LNDetector(args1)
                # nodules, reference_shape, spacing = ln_retector(case_id)
                np.save(os.path.join("/tmp/data", case_id, "nodules.npy"), nodules)
                notify_service_4(case_id)
                notify_service_5(case_id)
            elif signal != "0":
                time.sleep(0.5)
            else:
                flag = 1
        except Exception:
            exec_str = traceback.format_exc()
            mark_error(case_id, exec_str, "3")
