
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django import template
from django.http import HttpResponse, JsonResponse
from .models import Camera, Video, Suspect
from django.views.generic import ListView
import json
from json import dumps
from django.db.models import Q
from django.urls import reverse
import cv2
from django.contrib import messages


from .dl_scripts.anomaly_detector.anomaly_detector import *
from .dl_scripts.gvision_attributes.facedetect import *
from .dl_scripts.Crime.classify_crime import *
from .dl_scripts.person_reid.demo import *

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
from .serializers import FileSerializer
from django.http import HttpResponse
from utils import *

@api_view(["POST"])
def PredictAnomaly(request, id=None, *args, **kwargs):

    parser_class = (FileUploadParser,)

    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
        file_serializer.save()

    video = Video.objects.get(id=id)
    path = base + str(video.video)
    
    crop_path, duration, normal, anomaly, start, end = obtain_crop(path)
    if anomaly > normal:
        prediction = 'Abnormal'

    predictions = {

                'error' : '0',
                'message' : 'Successfull',
                'video_id' : video.id,
                'prediction' : prediction,
                'crime_start_time' : start,
                'crime_end_time' : end

            }

    print(predictions)

    return Response(predictions)

from kafka import KafkaConsumer

consumer = KafkaConsumer('my-topic', bootstrap_servers='192.168.14.2:9092')


def kafkastream():
    for message in consumer:
        nparr = np.fromstring(message.value, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("Consuming Demo Video")
        print(img.shape)
        

@api_view(["GET"])
def kafka_consumer(request,  *args, **kwargs ):
    
    print(kafkastream())
    return HttpResponse(kafkastream(), content_type='multipart/x-mixed-replace; boundary=frame')


@api_view(["POST"])
def get_suspect_attribute(request, id=None, *args, **kwargs):
    
    parser_class = (FileUploadParser,)

    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
        file_serializer.save()

    suspects = Suspect.objects.get(id=id)
    url = suspects.image.url
    suspect_path = base + url.split('/')[2] + '/' + url.split('/')[3]
    
    face, label, cloth_object , attr, cloth_color = get_gvision(suspect_path)

    suspects.face_df = face.to_json()
    suspects.object_df = cloth_object.to_json()
    suspects.label_df = label.to_json()
    suspects.safe_df = attr.to_json()
    suspects.cloth_color_df = cloth_color.to_json()

    suspects.save()


    predictions = {
        'suspects_id':suspects.id,
        'face_attribute': face,
        'object_attribute':cloth_object,
        'label_attribute':label,
        'safe_keys':attr,
        'cloth_color_keys':cloth_color,
            
    }

    return Response(predictions)