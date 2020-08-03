# -*- coding: UTF-8 -*-
import numpy as np
import cv2 as cv
from pathlib import Path
from ..Tracking.deep_sort import preprocessing
from ..Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from ..Tracking.deep_sort.detection import Detection
from ..Tracking import generate_dets as gdet
from ..Tracking.deep_sort.tracker import Tracker
import random
import os, time
# Use Deep-sort(Simple Online and Realtime Tracking)
# To track multi-person for multi-person actions recognition


file_path = '/home/mcw/subha/DSC/dsc_django/app/dl_scripts/Tracking/graph_model/mars-small128.pb'
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort
model_filename = file_path
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# track_box
trk_clr = (0, 255, 0)


def framewise_tracker(video_id,pose):
    frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
    joints_norm_per_frame = np.array(pose[-1])

    id_arr = []
    if bboxes:
        bboxes = np.array(bboxes)
        features = encoder(frame, bboxes)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

     
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # tracker
        tracker.predict()
        tracker.update(detections)

        # track bounding boxes ID
        trk_result = []
        for trk in tracker.tracks:
            if not trk.is_confirmed() or trk.time_since_update > 1:
                continue
            bbox = trk.to_tlwh()
            trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
            # track_ID
            trk_id = 'ID-' + str(trk.track_id)
        cnt = 0
        for d in trk_result:
            xmin = int(d[0])
            ymin = int(d[1])
            xmax = int(d[2]) + xmin
            ymax = int(d[3]) + ymin
            id = int(d[4])
            if cnt == 0:
                trk_clr = (0, 255, 0)
                cnt = 1
            else:
                trk_clr = (255,0 ,0)

            id_arr.append(id)
            try:

                # track_box human xcenter
                tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                j = np.argmin(tmp)
            except:
                j = 0

            crop = frame[ymin:ymax , xmin:xmax]

            vid_id_dir = os.path.join('crops',str(video_id))
            sus_id_dir = os.path.join(vid_id_dir,str(id))
            os.makedirs('full',exist_ok=True)
            os.makedirs('crops',exist_ok=True)
            os.makedirs(vid_id_dir,exist_ok=True)
            os.makedirs(sus_id_dir,exist_ok=True)

            cv.imwrite(sus_id_dir + '/' + str(time.time()) + '.jpg', cv.resize(crop,(300,300)))
            
            # track_box
            
            cv.putText(frame, 'ID-'+str(id), (int(bbox[0]), int(bbox[1]+bbox[3] + 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, trk_clr, 1)
            cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 1)
            cv.imwrite('full/' +str(time.time()) + '.jpg',cv.resize(frame,(700,700)))

    return id_arr, frame

