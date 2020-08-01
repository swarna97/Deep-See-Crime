import os
import cv2

from .dl_scripts.Pose.tracker import *
from .dl_scripts.Pose.pose_visualizer import TfPoseVisualizer

input_width, input_height = 656, 368

def load_pretrain_model(model):

    dyn_graph_path = {
    	'vgg_origin' : "/home/mcw/subha/DSC/dsc_django/app/dl_scripts/Pose/graph_models/VGG_origin/graph_opt.pb",
        'mobilenet_thin': "/home/mcw/subha/DSC/dsc_django/app/dl_scripts/Pose/graph_models/mobilenet_thin/graph_opt.pb",
    }

    graph_path = dyn_graph_path[model]
    if not os.path.isfile(graph_path):
        raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)

    return TfPoseVisualizer(graph_path, target_size=(input_width, input_height))

estimator = load_pretrain_model('vgg_origin')
