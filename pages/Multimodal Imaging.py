#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation Date: July 10, 2023

@author: Aaron Wilkowitz
"""

################
### import 
################
# gcp
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from google.cloud import bigquery
import gcsfs

# others
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import ast
from datetime import datetime
import datetime, pytz
import seaborn as sns
import base64
import sys
import time
import typing
import csv
import numpy as np
from IPython.display import Image
from image_encoder.image_encoder import *

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Medical Imaging -- Data Science Labels')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-08-02')
st.write('**Purpose**: Use Multimodal model to predict cancer detection')

# Gitlink
st.write('**Go Link (Googlers)**: go/hclsgenai')
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('30 Second Video')

default_width = 80 
ratio = 1
width = max(default_width, 0.01)
side = max((100 - width) / ratio, 0.01)

_, container, _ = st.columns([side, width, side])
# container.video(data='https://youtu.be/AtVCwywl_q8')

# Architecture

st.divider()
st.header('Architecture')

# components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vT-l1SQEqf6DrvlDMT_YhULvY74U1SnVCyfC7EVgXt2bPN4c6bejjPb0GeNjt4SHnz3v0t4SHjM-S-9/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569




st.divider()
st.header('Performance Summary')


import matplotlib.pyplot as plt

# Create a table
table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Create a gradient
gradient = plt.cm.get_cmap('cool')

# Plot the table with the gradient
gradient2 = plt.imshow(table, cmap=gradient)
# plt.show()
st.write(gradient2)




# st.write('Overall Performance')
# source_url = 'https://demoexpo.cloud.looker.com/login/embed/%2Fembed%2Fdashboards%2F1457?nonce=%22gj4RFJQzhBf7jsnQ%22&time=1691171139&session_length=600&external_user_id=%22test-id-123%22&permissions=%5B%22access_data%22%2C%22see_looks%22%2C%22see_user_dashboards%22%2C%22see_lookml_dashboards%22%2C%22explore%22%5D&models=%5B%22adopt%22%5D&group_ids=%5B%5D&external_group_id=%22%22&user_attributes=%7B%7D&access_filters=%7B%7D&first_name=%22test2%22&last_name=%22test3%22&force_logout_login=true&signature=nXF%2FDWLWPfZyRx5AiXy5vDoRjGg%3D'
# components.html(f'''<iframe src={source_url} width='600' height='338' frameborder='0'></iframe>''', width=650,height=350)
# st.write('Overall Performance')
# # File Name
# measurement_kpi = st.selectbox(
#     'What KPI do you want to measure performance on?'
#     , (
#           "Patient Orientation"
#         , "Left vs Right Breast"
#         , "Mass Pathology"
#         , "Breast Density"
#         , "Breast Subtlety"
#       )
#   )

# # if file_id == "CCDA Sample 1":
# #   file_name = "synthetic-bariatric-ccda-full-9580-lines.xml"
# # elif file_id == "CCDA Sample 2":
# #   file_name = "synthetic-bariatric-ccda-full-9580-linesv2.xml"
# # elif file_id == "FHIR Sample 1":
# #   file_name = "synthetic-bariatric-fhir-full-24527-lines.json"
# # else:
# #   file_name = "unknown"



# components.html('''<iframe src='https://demoexpo.cloud.looker.com/embed/public/pYZDMrwpBNkQHsK27XPqGvJByZx7Q3Pd?apply_formatting=false&apply_vis=false&toggle=pik' width='600' height='338' frameborder='0'> </iframe>''', width=650,height=350)



















# ################
# ### define embeddings
# ################

# project_id = 'cloudadopt'

# # # Go to Colab: https://colab.sandbox.google.com/github/tankbattle/hello-world/blob/master/Build_Cloud_CoCa_Image_Embedding_Dataset_%26_Search.ipynb#scrollTo=x2BrVlM-phGN
# # # Inspired from https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple.
# # class EmbeddingResponse(typing.NamedTuple):
# #   text_embedding: typing.Sequence[float]
# #   image_embedding: typing.Sequence[float]

# # class EmbeddingPredictionClient:
# #   """Wrapper around Prediction Service Client."""
# #   def __init__(self, project : str,
# #     location : str = "us-central1",
# #     api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
# #     client_options = {"api_endpoint": api_regional_endpoint}
# #     # Initialize client that will be used to create and send requests.
# #     # This client only needs to be created once, and can be reused for multiple requests.
# #     self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
# #     self.location = location
# #     self.project = project

# #   def get_embedding(self, text : str = None, image_bytes : bytes = None):
# #     if not text and not image_bytes:
# #       raise ValueError('At least one of text or image_bytes must be specified.')

# #     instance = struct_pb2.Struct()
# #     if text:
# #       instance.fields['text'].string_value = text

# #     if image_bytes:
# #       encoded_content = base64.b64encode(image_bytes).decode("utf-8")
# #       image_struct = instance.fields['image'].struct_value
# #       image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

# #     instances = [instance]
# #     endpoint = (f"projects/{self.project}/locations/{self.location}"
# #       "/publishers/google/models/multimodalembedding@001")
# #     response = self.client.predict(endpoint=endpoint, instances=instances)

# #     text_embedding = None
# #     if text:
# #       text_emb_value = response.predictions[0]['textEmbedding']
# #       text_embedding = [v for v in text_emb_value]

# #     image_embedding = None
# #     if image_bytes:
# #       image_emb_value = response.predictions[0]['imageEmbedding']
# #       image_embedding = [v for v in image_emb_value]

# #     return EmbeddingResponse(
# #       text_embedding=text_embedding,
# #       image_embedding=image_embedding)

# # client = EmbeddingPredictionClient(project=project_id)

# # # Extract image embedding
# # def getImageEmbeddingFromImageContent(content):
# #   response = client.get_embedding(text=None, image_bytes=content)
# #   return response.image_embedding

# # def getImageEmbeddingFromGcsObject(gcsBucket, gcsObject):
# #   client = storage.Client()
# #   bucket = client.bucket(gcsBucket)
# #   blob = bucket.blob(gcsObject)

# #   with blob.open("rb") as f:
# #     return getImageEmbeddingFromImageContent(f.read())

# # def getImageEmbeddingFromFile(filePath):
# #   with open(filePath, "rb") as f:
# #     return getImageEmbeddingFromImageContent(f.read())

# # # Extract text embedding
# # def getTextEmbedding(text):
# #   response = client.get_embedding(text=text, image_bytes=None)
# #   return response.text_embedding

# ################
# ### create embeddings
# ################

# #### One time process: run all 500 embeddings on the images

# # project_id = 'cloudadopt' # @param {type: "string"}

# # bucket_name = "hcls_genai"
# # new_file_name = 'hcls/dicom/csv/image_model_input.csv'
# # df = pd.read_csv('gs://' + bucket_name + '/' + new_file_name) 
# # # df = df.head(2)
# # # st.write(df)
# # # df_embedding = pd.DataFrame(columns=['image_path', 'embedding_str'])
# # client_bq = bigquery.Client()

# # ## Add this junk line of code to ensure this does not run and re-calculate all embeddings
# # xlkjlkajdsflkjasdf

# # for index, row in df.iterrows():
# #   image_bucket_name = "hcls_genai"
# #   image_file_name = row['image_path']
# #   embedding = getImageEmbeddingFromGcsObject(image_bucket_name, image_file_name)
# #   embedding_str = str(embedding)
# #     #   embedding_dict = {
# #     #       'image_path': image_file_name
# #     #     , 'embedding_str': embedding_str
# #     #   }
# #   # Append to df using dict
# #   # df_embedding_new_row = pd.DataFrame(embedding_dict, index=[0])
# #   # df_embedding = pd.concat([df_embedding, df_embedding_new_row])
# #   # st.write(df_embedding)

# #   # Construct a BigQuery client object.
# #   table_name = 'cloudadopt.dicom_mammography.image_model_embedding'
# #   query = f"""
# #   INSERT INTO `{table_name}`
# #   SELECT current_datetime(), '{image_file_name}', '{embedding_str}' 
# #   """
# #   query_job = client_bq.query(query)  # Make an API request.
# #   # st.write(query)
# #   st.write(image_file_name + ' done')

# ################
# ### encode images 
# ################

# # encode 

# # def getImageEmbeddingFromGcsObject(gcsBucket, gcsObject):
# #   client = storage.Client()
# #   bucket = client.bucket(gcsBucket)
# #   blob = bucket.blob(gcsObject)

# #   with blob.open("rb") as f:
# #     return getImageEmbeddingFromImageContent(f.read())







# client = storage.Client() # Implicit environment set up
# # with explicit set up:
# # client = storage.Client.from_service_account_json('key-file-location')

# bucket_name = "hcls_genai"
# bucket = client.get_bucket(bucket_name)
# new_file_name = 'hcls/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.406725628213826290127343763811145520834/1-192.jpg'
# blob = bucket.blob(new_file_name)
# blob_bytes = blob.download_as_bytes()
# # image = Image(blob_bytes)
# # st.image(image)







# # Download Files
# client = storage.Client()
# project_id = "cloudadopt"
# location_id = "us-central1"

# path = 'hcls/dicom/csv/'

# # bucket_name = 'ann_index'
# # path = 'index'
# bucket_uri = "gs://" + bucket_name + "/" + new_file_name
# # var_index_path = 'input/mass_train'
# # var_bucket_uri_index = "gs://" + var_bucket_name + "/" + var_index_path

# # public_url = 'https://storage.googleapis.com/hcls_genai/hcls/dicom/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.400026462310530719000693961182033505135/1-223.jpg'




# client = storage.Client() # Implicit environment set up
# # with explicit set up:
# # client = storage.Client.from_service_account_json('key-file-location')

# bucket = client.get_bucket('bucket-name')
# blob = bucket.get_blob('images/test.png')
# Image(blob.download_as_bytes())

# blobKey = blobstore.create_gs_key('/gs' + gcs_filename)

# to_send = encode(public_url)
# st.write(to_send)
# # image = decode(to_send)

# # image = Image.open(full_file_url)
# # st.image(image)

# # encoded_image = image.encode('jpg')
# # st.write(encoded_image)


# ################
# ### load an image 
# ################

# st.divider()
# st.header('Load an Image')

# # Download Files
# client = storage.Client()
# location_id = "us-central1"
# bucket_name = "hcls_genai"

# prefix = 'https://storage.googleapis.com/'
# file_name = 'hcls/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.406725628213826290127343763811145520834/1-192.jpg'
# full_file_url = prefix + file_name 

# st.write(':blue[**Image:**] ')
# st.image(full_file_url)

# ################
# ### Find nearest neighbors using dot product
# ################

# ### Create a datatable of every embedding
# # Construct a BigQuery client object.
# table_name = 'cloudadopt.dicom_mammography.image_model_embedding'
# query = f"""
# INSERT INTO `{table_name}`
# SELECT current_datetime(), '{image_file_name}', '{embedding_str}' 
# """
# query_job = client_bq.query(query)  # Make an API request.
# # st.write(query)
# st.write(image_file_name + ' done')


# scores = np.dot(question_embeddings[question_index], question_embeddings.T)

# # See here for more documentation 
# # https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/sdk_matching_engine_create_stack_overflow_embeddings.ipynb

# var_display_name = "dicom_mammogram_images"
# var_description = "mammogram images"     
# var_dimension = 1048
# var_bucket_name = 'ann_index'
# var_path = 'index'
# var_bucket_uri = "gs://" + var_bucket_name + "/" + var_path
# var_index_path = 'input/mass_train'
# var_bucket_uri_index = "gs://" + var_bucket_name + "/" + var_index_path

# aiplatform.init(
#     project=project_id 
#   , location=location_id 
#   , staging_bucket=var_bucket_uri
# )


# ## One time operation: create the tree index 

# # ## Add this junk line of code to ensure this does not recreate the tree index
# # xlkjlkajdsflkjasdf

# # tree_ah_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
# #     display_name=var_display_name,
# #     contents_delta_uri=var_bucket_uri_index,
# #     dimensions=var_dimension,
# #     approximate_neighbors_count=150,
# #     distance_measure_type="DOT_PRODUCT_DISTANCE",
# #     leaf_node_embedding_count=500,
# #     leaf_nodes_to_search_percent=80,
# #     description=var_description,
# # )

# # var_index_resource_name = tree_ah_index.resource_name
# # st.write(var_index_resource_name)

# # Create MatchingEngineIndex backing LRO: projects/1061181208409/locations/us-central1/indexes/7830836162229960704/operations/7797147881869148160

# tree_ah_index = aiplatform.MatchingEngineIndex(index_name=var_index_resource_name)

# # INDEX_RESOURCE_NAME = tree_ah_index.resource_name
# # INDEX_RESOURCE_NAME
     
# # Using the resource name, you can retrieve an existing MatchingEngineIndex.


# # tree_ah_index = aiplatform.MatchingEngineIndex(index_name=INDEX_RESOURCE_NAME)

# ################
# ### generate a text embedding 
# ################

# st.divider()
# st.header('Generate Text Embedding from Image')

# embedding = getImageEmbeddingFromGcsObject(bucket_name, file_name)
# embedding_str = str(embedding)
# st.write(':blue[**Embedding:**] ')
# st.write(embedding_str)

# ################
# ### generate the actual results
# ################

# client_bq = bigquery.Client()

# # Construct a BigQuery client object.
# table_name = 'cloudadopt.dicom_mammography.image_model_input_final'
# query = f"""
# SELECT * 
# FROM {table_name} 
# WHERE image_path = '{file_name}'
# """
# query_job = client_bq.query(query)  # Make an API request.
# # st.write(query)
# st.write(file_name + ' done')

# ################
# ### predict values 
# ################

# st.divider()
# st.header('Predict Values about an Image')



# # {"input_text": "embedding: []", "output_text": "{"BodyPartExamined":"BREAST","PatientOrientation":"MLO","breast_density":"4","left_or_right_breast":"RIGHT\",\"test_train\":\"train\",\"mass_vs_calc\":\"calc\","calc_abnormality_id":"1","calc_type":"PUNCTATE","calc_distribution":"SEGMENTAL","calc_assessment":"4","calc_pathology":"BENIGN","calc_subtlety":"1"}"}






































# # with open('image_embedding.csv', 'w') as f:
# #   csvWriter = csv.writer(f)
# #   csvWriter.writerow(['image_path', 'embedding'])
# #   for blob in gcsBucket.list_blobs():


# # # path = 'hcls/dicom/csv/'
# # new_file_name = 'hcls/dicom/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.406725628213826290127343763811145520834/1-192.jpg'
# # blob_name = new_file_name 
# # embedding = getImageEmbeddingFromGcsObject(bucket_name, new_file_name)
# # embedding_str = str(embedding)
# # st.write(embedding_str)

# # # Download Files
# # client = storage.Client()
# # project_id = "cloudadopt"
# # location_id = "us-central1"
# # bucket_name = "hcls_genai"
# # path = 'hcls/dicom/csv/'
# # new_file_name = 'hcls/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.406725628213826290127343763811145520834/1-192.jpg'


# # # bucket = client.bucket(bucket_name)
# # # blob = str(bucket.blob(file_name).download_as_string())
# # # st.write(':green[**Complete**] File Downloaded')
# # # blob_sample = blob[:1000]
# # # st.write(':blue[**File Sample:**] ' + blob_sample)

# # file_name = 'mass_case_description_train_set.csv'
# # df = pd.read_csv('gs://' + bucket_name + '/' + path + file_name) 
# # st.write('mass case')
# # st.write(df)

# # file_name = 'meta.csv'
# # df = pd.read_csv('gs://' + bucket_name + '/' + path + file_name) 
# # st.write('meta')
# # st.write(df)

# # file_name = 'dicom_info.csv'
# # df = pd.read_csv('gs://' + bucket_name + '/' + path + file_name) 
# # st.write('dicom info')
# # st.write(df)

# # file_name = 'calc_case_description_train_set.csv'
# # df = pd.read_csv('gs://' + bucket_name + '/' + path + file_name) 
# # st.write('calc case')
# # st.write(df)


# # # bucket = client.bucket(bucket_name)
# # # st.write('bucket ' + bucket)
# # # blobs_all = list(bucket.list_blobs(prefix=path))
# # # st.write('blobsall ' + str(blobs_all))

# # # https://storage.cloud.google.com/hcls_genai/hcls/archive%202/csv/mass_case_description_train_set.csv

# # # Let user select file
# # # blob_name2 = ''
# # # blob_options_string = ''
# # # for blob in blobs_all: 
# # #     blob_name = blob.name
# # #     blob_name = blob_name.replace(path,'')
# # #     blob_name = '\'' + blob_name + '\''
# # #     blob_options_string = blob_options_string + blob_name + ','
# # # blob_options_string = blob_options_string[:-1]
# # # blob_options_string = blob_options_string[3:]
# # # st.write('blob string' + blob_options_string)

# # # string1 = 'order_id = st.selectbox('
# # # string2 = '\'What order do you want to learn about?\''
# # # string3 = f', ({blob_options_string}), index = 0)'
# # # string_full = string1 + string2 + string3
# # # exec(string_full)

# # # file_name = order_id 
# # # full_file_name = path + file_name
# # # bucket = client.bucket(bucket_name)








# # # # Download File
# # # project_id = "cloudadopt"
# # # location_id = "us-central1"
# # # bucket_name = "hcls_genai"
# # # path_name = "hcls/archive%202/csv/"
# # # file_name = path_name + "mass_case_description_train_set.csv"
# # # client = storage.Client()
# # # bucket = client.bucket(bucket_name)
# # # blob = str(bucket.blob(file_name).download_as_string())

# # # # https://storage.cloud.google.com/hcls_genai/hcls/archive%202/csv/mass_case_description_train_set.csv
# # # st.write(':green[**Complete**] File Downloaded')

# # # blob_sample = blob[:1000]
# # # st.write(blob_sample)
# # # # st.write(':blue[**File Sample:**] ' + blob_sample)

# # # if file_id == "CCDA Sample 1":
# # #   file_name = "synthetic-bariatric-ccda-full-9580-lines.xml"
# # # elif file_id == "CCDA Sample 2":
# # #   file_name = "synthetic-bariatric-ccda-full-9580-linesv2.xml"
# # # elif file_id == "FHIR Sample 1":
# # #   file_name = "synthetic-bariatric-fhir-full-24527-lines.json"
# # # else:
# # #   file_name = "unknown"

# # # st.write(':blue[**File ID:**] ' + file_id)
# # # st.write(':blue[**File Name:**] ' + file_name)



