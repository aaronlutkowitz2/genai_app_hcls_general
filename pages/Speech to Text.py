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
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.cloud import storage

# others
import os
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import pandas as pd
import json
import math

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Find Relevant Section of a Video')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-08-22')
st.write('**Purpose**: Find Relevant Section of a Video.')

# Gitlink
st.write('**Go Link (Googlers)**: go/hclsgenai')
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# # Video
st.divider()
st.header('30 Second Video')

video_url = 'https://youtu.be/KiXkBi7Udsg'
st_player(video_url)

# # Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vRH7GzAhleSvA9OMX6_ksnsW0iDREHUun8ruLP5iCadV7LwD1AwhkCxnSiEfFPoY-LGixmCstmKrKKd/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### Select video
################

# Select workflow
st.divider()
st.header('1. Select video')

# Video
param_video = st.selectbox(
    'What video do you want to learn more about?'
    , (
          'Read a chest x-ray'
        , 'Import patient data into an EHR'
        , 'Measure blood pressure'
      )
    , index = 0
  )

video = param_video
st.write(f':blue[**Video**] ' + video)

if "chest" in video:
    video_url = "https://www.youtube.com/watch?v=FbFfVnKrk78"
    default_question = "How do the left and right lungs differ?"
    file_name = "audio_read_chest_xray2.mp3"
    output_file_name = "audio_read_chest_xray2_transcript_651dcf64-0000-2868-a3fa-240588769680.json"
elif "EHR" in video:
    video_url = "https://www.youtube.com/watch?v=XQuhOkiDXfY"
    default_question = "Which environment should I use to log in?"
    file_name = "audio_ehr_log_into_meditech.mp3"
    output_file_name = "audio_ehr_log_into_meditech_transcript_658f5b87-0000-255e-a9e0-3c286d48f672.json"
elif "blood pressure" in video: 
    video_url = "https://www.youtube.com/watch?v=tnUZioIUFrA"
    default_question = "How long should I wait until I take blood pressure again?"
    file_name = "audio_blood_pressure2.mp3"
    output_file_name = "audio_recordings_transcripts_audio_blood_pressure_nhs.mp3-20230829094843.json"
else: 
    video_url = "https://www.youtube.com/watch?v=FbFfVnKrk78"
    default_question = "How do the left and right lungs differ?"
    file_name = "audio_read_chest_xray2.mp3"
    output_file_name = "audio_read_chest_xray2_transcript_651dcf64-0000-2868-a3fa-240588769680.json"

# Show Video
st_player(video_url)

################
### Create transscript of video's audio files -- ONE TIME

# (once these are created I added the URLs to transcript_output above)
################

# file_name = "audio_ehr_log_into_meditech.mp3"

# client = SpeechClient() # Instantiates a client
# gcs_uri = f"gs://hcls_genai/audio_recordings/demo/input/{file_name}" # Input
# workspace = "gs://hcls_genai/audio_recordings/demo/output" # Output
# # name = "projects/cloudadopt/locations/us-central1/recognizers/_" # Recognizer resource name
# name = "projects/cloudadopt/locations/global/recognizers/_" # Recognizer resource name
# model_name = "long"

# config = cloud_speech.RecognitionConfig(
#   auto_decoding_config={},
#   model=model_name,
#   language_codes = ['en-US'],
#   features=cloud_speech.RecognitionFeatures(
#   enable_word_time_offsets=True,
#   enable_word_confidence=True,
#   ),
# )

# output_config = cloud_speech.RecognitionOutputConfig(
#   gcs_output_config=cloud_speech.GcsOutputConfig(
#     uri=workspace),
# )

# files = [cloud_speech.BatchRecognizeFileMetadata(
#     uri=gcs_uri
# )]

# request = cloud_speech.BatchRecognizeRequest(
#     recognizer=name, config=config, files=files, recognition_output_config=output_config
# )
# operation = client.batch_recognize(request=request)
# st.write(operation.result())

################
### Create time-marked transcript 
################

# Select workflow
st.divider()
st.header('2. Create time-marked transcript')

# Pull json from Storage
client = storage.Client()
bucket_name = "hcls_genai"
path = "audio_recordings/demo/output/"
full_file_path = path + output_file_name
bucket = client.bucket(bucket_name)
blob = str(bucket.blob(full_file_path).download_as_string())

# Create json_string
blob = blob[2:][:-1].replace("\\n","").replace("\\","").replace("endTime","endOffset") ## remove first 2, last character; replace "\" escapes
# st.text(str(blob))
json_object = json.loads(blob)
# st.text(str(json_object))

# Create a list 
json_list = [{"transcript":"","timestamp":0.0}]

json_object2 = json_object["results"]
for index, value in enumerate(json_object2):
    # Grab transcript + timestamp
    transcript = value["alternatives"][0]["transcript"]
    # st.write("transcript " + transcript)
    timestamp = float(value["alternatives"][0]["words"][0]["endOffset"][:-1])
    # st.write("time_stamp " + str(timestamp))

    # Append to json dict
    dictionary = {"transcript":transcript,"timestamp":timestamp}
    json_list.append(dictionary)

# Create a df based on that list (to display)
df = pd.DataFrame(json_list).sort_values(by=['timestamp'])
st.table(df)

# convert list to string for context
json_list = str(json_list)

################
### Ask & Answer Question
################

# Select workflow
st.divider()
st.header('3. Ask & Answer Question')

custom_prompt = st.text_input('Write your question here', value = default_question)

# Model parameters
project_id = "cloudadopt"
location_id = "us-central1"
model_id = 'chat-bison@001' 
model_temperature = 0.2 
model_token_limit = 200 
model_top_k = 40 
model_top_p = 0.8

vertexai.init(
      project = project_id
    , location = location_id)
chat_model = ChatModel.from_pretrained(model_id)
parameters = {
    "temperature": model_temperature,
    "max_output_tokens": model_token_limit,
    "top_p": model_top_p,
    "top_k": model_top_k
}
chat = chat_model.start_chat(
    context=json_list,
)

# To review context
# st.text(json_list)

response = chat.send_message(custom_prompt, **parameters)
llm_response_text = response.text 
st.write(':blue[**LLM Response:**] ' + llm_response_text)

prompt2 = """
Provide the timestamp of where you learned that information. 

Only provide the number. Do not provide any other information

If there is no timestamp associated with the information, write "0"

If there are multiple timestamps associated with the information, write the earliest timestamp only"""

response = chat.send_message(prompt2, **parameters)
llm_response_text2 = response.text 
st.write(':blue[**Timestamp:**] ' + llm_response_text2)

################
### Go to video at relevant timestamp
################

# Select workflow
st.divider()
st.header('4. Go to video at relevant timestamp')

timestamp = float(llm_response_text2)
timestamp_int_str = str(int(math.floor(timestamp)) - 2) ## start 2 seconds earlier
# st.write(timestamp_int_str)
video_url_with_timestamp = video_url + '&t=' + timestamp_int_str

# Show Video at correct timestamp
st_player(video_url_with_timestamp)