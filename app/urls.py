# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from app import views
from .views import *
from .endpoints import *
from rest_framework_simplejwt import views as jwt_views

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
    re_path('search/(?P<id>[\w-]+)/$',SearchVideoView.as_view(), name='search'),
    re_path('predict_anomaly/(?P<id>[\w-]+)/$', PredictAnomaly, name='predict_anomaly'),
    re_path('get_suspect_attribute/(?P<id>[\w-]+)/$', get_suspect_attribute, name='predict_anomaly'),
    # re_path('kafka_consumer/', kafka_consumer, name='kafka_consumer'),

    
    path('api/token/', jwt_views.TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
]