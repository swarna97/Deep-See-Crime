
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django import template
import json
from json import dumps
from django.db.models import Q
from django.urls import reverse
import cv2
from django.contrib import messages

# HTTP Response

from django.http import HttpResponse, JsonResponse
from django.http import HttpResponse
from django.views.generic import ListView


# Django Models

from .models import Camera, Video, Suspect


# DL Scripts

from .dl_scripts.anomaly_detector.anomaly_detector import *
from .dl_scripts.gvision_attributes.facedetect import *
from .dl_scripts.Crime.classify_crime import *
from .dl_scripts.person_reid.demo import *

# REST API

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
from .serializers import FileSerializer



@login_required(login_url="/login/")
def index(request):
    return render(request, "index.html")

@login_required(login_url="/login/")
def CameraDetailView(request, id=None, *args, **kwargs):
    camera = Camera.objects.get(id=id)
    video = Video.objects.filter(camera=id)
    context={
        'count':len(video),
        'video':video,
        'camera':camera,
    }
    return render(request, "videos.html", context)

@login_required(login_url="/login/")
def suspectDetails(request, id=None, *args, **kwargs):
    video = Video.objects.get(id=id)
    suspects = Suspect.objects.filter(video=id)
    context={
        'suspects' : suspects
    }
    return render(request, "page-user.html", context)

@login_required(login_url="/login/")
def viewsuspectDetails(request, *args, **kwargs):
    suspects = Suspect.objects.all()

    context={
        'suspects' : suspects
    }
    return render(request, "view_suspects.html", context)

class SearchVideoView(ListView):
    template_name = "search.html"

    def get_context_data(self, *args, **kwargs):
        context = super(SearchVideoView, self).get_context_data(*args, **kwargs)
        query = self.request.GET.get('q')
        context['query'] = query
        return context

    def get_queryset(self,*args,**kwargs):
        request = self.request
        method_dict=request.GET
        query=method_dict.get('q',None)
        print("Query:",query)
        if query is not None:
            print("SearchProd:",Video.objects.search(query))
            return Video.objects.search(query)
        return Video.objects.all()


# Module 1&2
@login_required(login_url="/login/")
def VideoDetailView(request, id=None, *args, **kwargs):
    
    video = Video.objects.get(id=id)
    path =  base + str(video.video)
    
    
    crop_path, duration, normal, anomaly, start = obtain_crop(path)
    norm, anom1,anom2,anom3,anom4 = get_pred(normal, anomaly, path)
    
    crop_duration = duration-start
    video.duration = crop_duration
    video.classified = True
    video.save()


    xlabels = ["Normal", "Assault", "Burglary", "Abuse", "Fighting"]
    ylabels = [norm,anom1,anom2,anom3,anom4]

    json_xlabels = dumps(xlabels)
    json_ylabels = dumps(ylabels)

    messages.success(request,"Video Analysis successfully done!!")
    context={
        'video':video,
        'crimexlabels':json_xlabels,
        'crimeylabels':json_ylabels,
    }

    return render(request, "video-details.html", context)

@login_required(login_url="/login/")
def oneSuspectData(request, id=None, *args, **kwargs):
    
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

    face_keys, face_values = list(face.columns),list(face.iloc[0])
    object_keys, object_values = list(cloth_object['name']),list(cloth_object['score'])
    label_keys, label_values = list(label['description']),list(label['score'])
    safe_keys, safe_values = list(attr.columns),list(attr.iloc[0])
    cloth_color_keys, cloth_color_values = list(cloth_color['color']),list(cloth_color['score'])

    print(cloth_color_keys, cloth_color_values)

    # Attribute Estimation 
    context = {
        'face_keys':dumps(face_keys),
        'object_keys':dumps(object_keys),
        'label_keys':dumps(label_keys),
        'safe_keys':dumps(safe_keys),
        'cloth_color_keys':dumps(cloth_color_keys),
        'face_values':dumps(face_values),
        'object_values':dumps(object_values),
        'label_values':dumps(label_values),
        'safe_values':dumps(safe_values),
        'cloth_color_values':dumps(cloth_color_values),
        'suspects':suspects     
    }
    
    
    return render(request, "ui-typography.html", context)


@login_required(login_url="/login/")
def trackPerson(request, id=None, *args, **kwargs):
    suspects = Suspect.objects.get(id=id)
    re_id_list = []
    query_id = suspects.query

    if query_id == 1:
        re_id_dict_1 = {'cam':0, 'vid':1, 'duration':0, 'path':'/static/assets/img/person_crop_2.PNG'}
        re_id_list.append(re_id_dict_1)
        re_id_dict_2 = {'cam':2, 'vid':1, 'duration':0, 'path':'/static/assets/img/person_crop_2(1).PNG'}
        re_id_list.append(re_id_dict_2)
    
    else:

        query_path, re_id_infos = re_id_suspect(query_index=query_id)
        for re_id in re_id_infos:
            re_id_dict = {'cam':re_id[0], 'vid':re_id[1], 'duration':re_id[2], 'path':re_id[3]}
            re_id_list.append(re_id_dict)

    # Returns RE_ID Images Path
    context={
        'suspects' : suspects,
        're_id_list' : re_id_list
    }
    
    return render(request, "reid_results.html", context)


@login_required(login_url="/login/")
def pages(request):
    context = {}
    
    try:

        load_template = request.path.split('/')[-1]
        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'error-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:

        html_template = loader.get_template( 'error-500.html' )
        return HttpResponse(html_template.render(context, request))

