# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from app import views
from .views import *

urlpatterns = [
    # Matches any html file
    path('', views.index, name='home'),
    re_path('view/(?P<id>[\w-]+)/$', CameraDetailView, name='view'),
    re_path('analyse/(?P<id>[\w-]+)/$', VideoDetailView, name='analyse'),
    re_path('suspects/(?P<id>[\w-]+)/$', suspectDetails, name='suspects'),
    path('view_suspects/',viewsuspectDetails, name='view-suspects'),
    re_path('suspectDetails/(?P<id>[\w-]+)/$', oneSuspectData, name='onesuspect'),
    re_path('trackPerson/(?P<id>[\w-]+)/$',trackPerson, name='trackperson'),
    re_path(r'^.*\.html',pages, name='pages'),
    path('search/',SearchVideoView.as_view(), name='search')

    # The home page
    # path('', views.index, name='home'),
]
