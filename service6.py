#! /usr/bin/env python3

import sys
import os
import shutil
import time
import pickle

"""
from couchbase.cluster import Cluster, PasswordAuthenticator
from couchbase.n1ql import N1QLQuery
"""
import numpy as np
import traceback

from config import USERNAME
from redis_control import get_service_6, save_service6_mark, mark_error, get_signal


"""
cluster = Cluster('couchbase://localhost')
authenticator = PasswordAuthenticator('deepln', 'jevoislavieenrose')
cluster.authenticate(authenticator)

cb = cluster.open_bucket('deepln')
"""


def save_draft_in_db(case_id):
    with open(os.path.join("/tmp/data", case_id, "segment.pkl"), "rb") as handle:
        segmentations = pickle.load(handle)

    with open(os.path.join("/tmp/data", case_id, "classify.pkl"), "rb") as handle:
        classification = pickle.load(handle)

    nobj = {}
    nobj["caseId"] = case_id
    nobj["username"] = USERNAME
    nobj["status"] = "1"
    nobj["type"] = "draft"
    spacingValue = float(np.load(os.path.join("/tmp/data", case_id, "spacing.npy"))[1])
    nobj["spacing"] = spacingValue
    rects = []
    for idx, segmentation_item in enumerate(segmentations):
        classification_item = classification[idx]
        new_obj = {}
        new_obj.update(segmentation_item)
        new_obj.update(classification_item)
        new_obj["status"] = 1
        new_obj["type"] = "nodule"
        rects.append(new_obj)
    nobj_rect = []
    for idx, rect in enumerate(rects):
        nodule_key = USERNAME + "#" + case_id + "#" + str(idx) + "@nodule"
        nobj_rect.append(nodule_key)
        cb_upsert(nodule_key, rect)
    nobj["rects"] = nobj_rect

    key = USERNAME + "#" + case_id + "@draft"
    time.sleep(1)
    cb_upsert(key, nobj)
    # tmp_save(rects, nobj)

def cb_upsert(key, obj):
    filename = '/home/pkls/' + key
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)

'''
def tmp_save(rects, nobj):
    saved_obj = {
            'rects': rects,
            'nobj': nobj
            }
    with open('pkls/' + case_id + '.pkl', 'wb') as handle:
        pickle.dump(saved_obj, handle)
'''

if __name__ == "__main__":
    flag = 0
    while flag == 0:
        try:
            signal = get_signal()
            case_id = get_service_6()
            if case_id is not None:
                print("Service VI: ", case_id)
                save_draft_in_db(case_id)
                save_service6_mark(case_id)
                shutil.rmtree(os.path.join("/tmp/data", case_id), ignore_errors=True)
            elif signal != "0":
                time.sleep(0.5)
            else:
                flag = 1
        except Exception:
            exec_str = traceback.format_exc()
            mark_error(case_id, exec_str, "6")
