import os, io
from google.cloud import vision
import pandas as pd
from numpy import random
from colormap import rgb2hex

credential_path = '/home/mcw/subha/DSC/dsc_django/app/dl_scripts/gvision_attributes/servicetokenjson.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

client = vision.ImageAnnotatorClient()


def faces(content):
    image = vision.types.Image(content=content)
    response1 = client.face_detection(image=image)
    faceAnnotations = response1.face_annotations

    #likehood = ('Unknown', 'Very Unlikely', 'Unlikely', 'Possibly', 'Likely', 'Very Likely')
    

    print('Faces:')
    face = faceAnnotations[0]
        
    df = pd.DataFrame(columns=['Angry', 'Joy', 'Sorrow','Surprised','Headwear','Exposed','Blurred','Confidence'])
    
    df = df.append(
            dict(
                Angry=face.anger_likelihood*20,
                Joy=face.joy_likelihood*20,
                Sorrow=face.sorrow_likelihood*20,
                Surprised=face.surprise_likelihood*20,
                Headwear=face.headwear_likelihood*20,
                Exposed=face.under_exposed_likelihood*20,
                Blurred=face.blurred_likelihood*20,
                Confidence=round(face.detection_confidence,2)*100
                
            ), ignore_index=True)
    return df


def labels(content):
    image = vision.types.Image(content=content)
    response3 = client.label_detection(image=image)
    labels = response3.label_annotations

    df = pd.DataFrame(columns=['description', 'score'])
    print("Labels:")
    for label in labels:
        df = df.append(
            dict(
                description=label.description,
                score=round(label.score,2)*100,
                
            ), ignore_index=True)
        
    return df


def object(content):
    image = vision.types.Image(content=content)
    response = client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations

    df = pd.DataFrame(columns=['name', 'score'])
    for obj in localized_object_annotations:
        df = df.append(
            dict(
                name=obj.name,
                score=round(obj.score,2)*100
            ),
            ignore_index=True)
        
       

    return df

def safesearch(content):
    image = vision.types.Image(content=content)
    response = client.safe_search_detection(image=image)
    safe_search = response.safe_search_annotation
  
            
    df = pd.DataFrame(columns=['Adult', 'Spoof', 'Medical','Violence','Racy'])
    
    df = df.append(
            dict(
                Adult=safe_search.adult*20,
                Spoof=safe_search.spoof*20,
                Medical=safe_search.medical*20,
                Violence=safe_search.violence*20,
                Racy=safe_search.racy*20
                
            ), ignore_index=True)
        
    return df

def properties(content):
    image = vision.types.Image(content=content)
    response2 = client.image_properties(image=image).image_properties_annotation
    dominant_colors = response2.dominant_colors


    df = pd.DataFrame(columns=['color','score'])
 
    for color in dominant_colors.colors:
        hex_value = rgb2hex(int(color.color.red),int(color.color.green),int(color.color.blue))
        df = df.append(
            dict(
                color=hex_value,
                score=color.score*100

            ), ignore_index=True)

    return df

def get_gvision(image_path):

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

        face_df = faces(content)
        label_df = labels(content)
        object_df = object(content)
        safe_df = safesearch(content)
        cloth_color_df = properties(content)

        return face_df, label_df, object_df, safe_df, cloth_color_df