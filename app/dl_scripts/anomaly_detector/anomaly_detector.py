import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
import subprocess

from .utils.data_loader import VideoIter
from .utils.utils import set_logger, build_transforms
from .utils.feature_extractor import FeaturesWriter
from .utils.features_loader import FeaturesLoaderVal
from .utils.annotation_methods import annotatate_file

from .network.anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective
from .network.model import static_model
from .network.c3d import C3D


model_dir = '/home/mcw/subha/DSC/dsc_django/app/dl_scripts/anomaly_detector/models/anomaly_detector_model'
pretrained_3d = '/home/mcw/subha/DSC/dsc_django/app/dl_scripts/anomaly_detector/models/c3d.pickle'
features_dir = '/home/mcw/subha/DSC/dsc_django/app/dl_scripts/anomaly_detector/features/'
cropped_v_path = '/home/mcw/subha/DSC/dsc_django/core/static/cropped_video/'

def figure2opencv(figure):
    figure.canvas.draw()
    img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def c3d_extraction(video_path,device=None):

    batch_size=1
    train_frame_interval=2
    clip_length=16


    single_load=True #should not matter
    home=os.getcwd()


    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")


    #Load clips
    print("Loading clips")
    train_loader = VideoIter(dataset_path=None,
                                  annotation_path=video_path,
                                  clip_length=clip_length,
                                  frame_stride=train_frame_interval,
                                  video_transform=build_transforms(),
                                  name='train',
                                  single_load=single_load)


    print("train loader done, train_iter now")
    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=32,  # 4, # change this part accordingly
                                             pin_memory=True)

    #Possesing with CD3
    print("Now loading the data to C3D network")
    network = C3D(pretrained=pretrained_3d)
    network.to(device)

    if not os.path.exists(features_dir):
        os.mkdir(features_dir)

    features_writer = FeaturesWriter()

    dir_list=[]

    for i_batch, (data, target, sampled_idx, dirs, vid_names) in tqdm(enumerate(train_iter)):
        with torch.no_grad():
            outputs = network(data.cuda())

            for i, (dir, vid_name, start_frame) in enumerate(zip(dirs, vid_names, sampled_idx.cpu().numpy())):
                dir_list.append([dir,vid_name])
                dir = os.path.join(features_dir, dir)
                features_writer.write(feature=outputs[i], video_name=vid_name, start_frame=start_frame, dir=dir)

    features_writer.dump()
    print("Features Dumped")
    return dir_list

def AD_prediction(video_path_ad, model_dir,dir_list,device=None):

    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    #pediction of AD with pertrain network
    network = AnomalyDetector()
    network.to(device)
    net = static_model(net=network,
                           criterion=RegularizedLoss(network, custom_objective).cuda(),
                           model_prefix=model_dir,
                           )
    model_path = net.get_checkpoint_path(20000)
    net.load_checkpoint(pretrain_path=model_path, epoch=20000)
    net.net.to(device)

    annotation_path=annotatate_file(video_path_ad,dir_list,normal=[True],file_name="Demo_anmotation")

    #runing vedio in to network
    data_loader = FeaturesLoaderVal(features_path=features_dir,
                                    annotation_path=annotation_path)

    data_iter = torch.utils.data.DataLoader(data_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,  # 4, # change this part accordingly
                                            pin_memory=True)
    print("it is over")

    for features, start_end_couples, feature_subpaths, lengths in tqdm(data_iter):
        # features is a batch where each item is a tensor of 32 4096D features
        features = features.to(device)
        with torch.no_grad():
            input_var = torch.autograd.Variable(features)
            outputs = net.predict(input_var)[0]  # (batch_size, 32)
            outputs = outputs.reshape(outputs.shape[0], 32)
            for vid_len, couples, output in zip(lengths, start_end_couples, outputs.cpu().numpy()):
                y_true = np.zeros(vid_len)
                segments_len = vid_len // 32
                for couple in couples:
                    if couple[0] != -1:
                        y_true[couple[0]: couple[1]] = 1
                y_pred = np.zeros(vid_len)
                print()
                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = output[i]


    return y_pred


def obtain_result(video_path_new, y_pred, save=True):
    print("Obtain",video_path_new)

    DISPLAY_IMAGE_SIZE = 500

    videoReader = cv2.VideoCapture(video_path_new)

    if save == True:

        fps = videoReader.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        file_name = 'demo.mp4'
        new_out_path = cropped_v_path + file_name
        # out = cv2.VideoWriter(new_out_path, fourcc,fps,(600, 600))

    #farme_window=[0]*clip_length
    fig = plt.figure()
    frame_count = 0
    length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(videoReader.get(cv2.CAP_PROP_FPS))
    normal = 0
    anomaly = 0
    flag = 0
    # pred_array = []
    frame_count = 0
    print("Video length: ",length)
    while frame_count < length:

        ret, currentImage = videoReader.read(frame_count)

        if ret == False:
            print("Cannot Open Video")
            return 0

        frame_count = frame_count + 1

        targetSize = 600
        frame = cv2.resize(currentImage, (targetSize, targetSize))
        # print(y_pred[frame_count -  1])

        if y_pred[frame_count -1] < 0.35:            
            prediction = 0
            normal += 1

        else:
            if flag == 0:
                start_frame = frame_count
                start = round(((frame_count-1) / fps),2)
                flag = 1

            if frame_count > start_frame:
                st = str(start + 4)
                end = str(start + 9)
                cmd = ['ffmpeg', '-y', '-t', end, '-i' ,video_path_new, '-ss' ,st, new_out_path]
                subprocess.run(cmd)
                break
                # out.write(frame)

            prediction = 1
            anomaly += 1

        # if frame_count == 400:
        #     break

    # out.release()


    return file_name, round(length/fps,2), normal, anomaly, start


def obtain_crop(video_path):


    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    if os.path.exists(video_path) == False:
        print("Wrong Video Path!!")
        return 0,0,0,0


    dir_list = c3d_extraction(video_path,device=device)
    y_pred = AD_prediction(video_path, model_dir, dir_list, device=device)
    file_name, duration, normal, anomaly, start = obtain_result(video_path, y_pred)


    return file_name, duration, normal, anomaly, start


# video_path = '/home/mcw/subha/DeepSeeCrime/Datasets/UCF_Dataset_Kaggle/Data/Arson/Arson002_x264.mp4'
# print("Anomaly detector results: ", obtain_crop(video_path))