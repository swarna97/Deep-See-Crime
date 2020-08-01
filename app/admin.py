# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib import admin

# Register your models here.
from .models import Video, Camera, Suspect

class VideoAdmin(admin.ModelAdmin):
    list_display = ['__str__', 'name']
    list_filter = ('classified','crimetype')
    class Meta:
        model = Video
    
admin.site.register(Video, VideoAdmin)

class CameraAdmin(admin.ModelAdmin):
    list_display = ['__str__', 'name']
    list_filter = ('place',)
    class Meta:
        model = Camera

admin.site.register(Camera, CameraAdmin)



class SuspectAdmin(admin.ModelAdmin):
    list_display = ['__str__']
    class Meta:
        model = Suspect

admin.site.register(Suspect, SuspectAdmin)