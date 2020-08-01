# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""
from django.db import models
import random
import os
from django.db.models import Q
from django.urls import reverse
from django.contrib.auth.models import User
from jsonfield import JSONField

# Create your models here.
class Camera(models.Model):
    name = models.CharField(max_length=120, unique=True)
    place = models.CharField(max_length=120)

    def __str__(self):
        return str(self.id)

class VideoQuerySet(models.query.QuerySet):
    def search(self, query):
        lookups = (Q(crimetype__icontains=query) |
                  Q(name__icontains=query))
                #   Q(camera__icontains=query))
        return self.filter(lookups).distinct()

class VideoManager(models.Manager):
    def get_queryset(self):
        return VideoQuerySet(self.model, using=self._db)

    def search(self, query):
        return self.get_queryset().search(query)

class Video(models.Model):
    camera =  models.ForeignKey(Camera, null=True, blank=True, on_delete=models.CASCADE)
    name = models.CharField(max_length=120, unique=True)
    video  = models.FileField(upload_to="video/", null=True, blank=True)
    cropped_video = models.FileField(upload_to="cropped_video/", null=True, blank=True)
    crimetype = models.CharField(max_length=120, default="None")
    duration = models.IntegerField(default=1, null=True,)
    start_duration = models.IntegerField(default=1, null=True,)
    end_duration = models.IntegerField(default=1, null=True,)
    classified = models.CharField(max_length=120,default=False)

    objects = VideoManager()

    def __str__(self):
        return str(self.id)

class Suspect(models.Model): 
    video = models.ForeignKey(Video, null=True, blank=True, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="img/",null=True,blank=True)
    face_df = JSONField(default="None", null=True)
    label_df = JSONField(default="None", null=True)
    object_df = JSONField(default="None", null=True)
    safe_df = JSONField(default="None", null=True)
    cloth_color_df = JSONField(default="None", null=True)
    query = models.IntegerField(default=1, null=True,unique=True)
    query_img = models.ImageField(upload_to="img/",null=True,blank=True)


    def __str__(self):
        return str(self.id)
