# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from vertexai.preview.language_models import TextGenerationModel, ChatModel
from google.cloud import storage, speech, translate 
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

# others
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import pandas as pd
import json 
import utils_config
import base64
import math
import re
# from pydub import AudioSegment
# import io
# import soundfile as sf

# from st_audiorec import st_audiorec # audio recorder
# # from audiorecorder import audiorecorder
# from datetime import datetime, timedelta
# import os
# import pyaudio
# import pydub
# import time
# import wave
# from queue import Queue
# from termcolor import cprint
# from threading import Thread
# from typing import List, Optional

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('Transcribe Doctor-Patient Encounter & Provide Recommendations')

# Author & Date
st.write('**Author**: Aaron Wilkowitz')
st.write('**Date**: 2023-10-24')
st.write('**Purpose**: Transcribe Doctor-Patient Encounter & Provide Recommendations')

# Gitlink
# st.write('**Go Link (Googlers)**: go/hclsgenai')
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('60 Second Video')

# video_url = 'https://youtu.be/JMyGfydQGzk'
# st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vQ3xh1M7Ttzzy4Y_T4-kI7If9Bebwz4FUVuSKNdAQnydkzCaVy8A5RDHIYo2gCk7_LI-95RpGTRxeJU/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### General
################

# Define parameters
PROJECT_ID = utils_config.get_env_project_id()
LOCATION = utils_config.LOCATION

# Define model
model_id = 'text-bison' 
model_temperature = 0
model_token_limit = 500
model_top_k = 40 
model_top_p = 0.8

#########################
### One Time Recording -- Get Transcript -- START 
#########################

# file_name = "recording2.mp3"

# client = SpeechClient() # Instantiates a client
# gcs_uri = f"gs://hcls_genai/scribbles/{file_name}" # Input
# workspace = "gs://hcls_genai/scribbles/" # Output
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
### Prompt 1 -- Diarization
################

# Output the predicted speaker under the \"LLM predicted speaker\" label, output your phrasing improving of the transcript under the \"LLM corrected transcript\" label, and output your confidence for the quality of phrasing improvement under the \"LLM confidence\" label.

prompt_diary = """
Context: You are a helpful medical knowledge assistant. 
Goal: Process raw transcripts of medical conversations and transform them into corrected transcripts with speaker diarization between a Doctor and a Patient and phrasing improvements. 
Tasks: 
(1) Accurately identify & label speakers
(2) Ensure proper punctuation, capitalization, grammar, and syntax
(3) Enhance phrasing and clarity
(4) Leverage contextual understanding to resolve ambiguities and accurately convey the intended meaning

Throughout the conversation, you should: 
- Maintain consistency in vocabulary and terminology
- Eliminate repetitions and redundancies
- Make factual corrections when necessary

By applying these rules and your medical knowledge, you will provide an optimized transcript that enhances readability and effectively captures the dialogue\'s content and context. 

Examples: 

input: 
transcript: Good afternoon. How can I assist you today?
------
transcript: Hi doctor. I\'ve been experiencing a persistent cough for the past week, and it\'s not getting any better.
output: 
- LLM predicted speaker:       Doctor \n
- language_code:               en-us \n
- original transcript:         Good afternoon. How can I assist you today? \n
- LLM corrected transcript:    Good afternoon. How can I assist you today? \n
-------------------------------------------------------------------------------- \n
- LLM predicted speaker:       Patient \n
- language_code:               en-us  \n
- original transcript:         Hi doctor. I\'ve been experiencing a persistent cough for the past week, and it\'s not getting any better. \n
- LLM corrected transcript:    Hi doctor. I\'ve been experiencing a persistent cough for the past week, and it\'s not getting any better. \n

input: 
transcript:    Hmm. I see have you noticed any other symptoms accompanying the call any fever chest pain?
--------------------------------------------------------------------------------
transcript:    or shortness of breath
output: 
- LLM predicted speaker:       Doctor \n
- language_code:               en-us \n
- original transcript:         Hmm. I see have you noticed any other symptoms accompanying the call any fever chest pain? or shortness of breath \n
- LLM corrected transcript:    Hmm. I see have you noticed any other symptoms accompanying the cough? Any fever, chest pain, or shortness of breath? \n

input: 
transcript:    You know, I haven\'t had a fever or chest pain, but I do feel a bit short of breath when the coughing fits occur.
--------------------------------------------------------------------------------
transcript:    especially during physical activities
output: 
- LLM predicted speaker:       Patient \n
- language_code:               en-us \n
- original transcript:         You know, I haven\'t had a fever or chest pain, but I do feel a bit short of breath when the coughing fits occur. especially during physical activities \n
- LLM corrected transcript:    You know, I haven\'t had a fever or chest pain, but I do feel a bit short of breath when the coughing fits occur. Especially during physical activities. \n

input: 
transcript:    I understand.
--------------------------------------------------------------------------------
transcript:    Have you been in contact with anyone who has been sick recently?
output: 
- LLM predicted speaker:       Doctor \n
- language_code:               en-us \n
- original transcript:         I understand. Have you been in contact with anyone who has been sick recently? \n
- LLM corrected transcript:    I understand. Have you been in contact with anyone who has been sick recently? \n
"""

prompt_medical_advice = """
Context: As a stateful and helpful medical assistant, your role is to provide comprehensive diagnostic and treatment recommendations based on the ongoing conversation between the doctor and the patient. 

Goal: Track the conversation in real-time, monitor applicable diagnoses, treatment plans, and recommended/explicit tasks specific to the doctor and patient, and provide accurate and helpful recommendations tailored to the patient's condition and the doctor's expertise.

Tasks: 
- Throughout the conversation, you should output the names or descriptions of new diagnoses, treatment plans, recommended tasks, or explicit tasks that gain relevancy. 
- If nothing medically relevant has been said yet in context of your role when tracking the conversation, always output an empty string until something medically relevant has been said and then begin task as has been described.

Below are example inputs and outputs of a stateful running list progression through an example chat. 

Do not explicitly copy these example responses. Only use them as a guide to help you diligently learn what is means to capture a running stateful list over time.

Example: 

Input 1:
transcript: 
Good afternoon. How can I assist you today?
Hi doctor. I've been experiencing a persistent cough for the past week, and it's not getting any better.

Output 1:
Potential Diagnoses:
- Acute bronchitis: The persistent cough could be a symptom of acute bronchitis, which is an inflammation of the bronchial tubes. 
- Allergic rhinitis: The cough could be related to allergic rhinitis, as it can cause postnasal drip and throat irritation, leading to a cough. 
- Upper respiratory infection: A persistent cough can also be a result of an upper respiratory infection, such as a common cold. 

Treatment Plans:
- Symptomatic relief: Recommend over-the-counter cough suppressants and throat lozenges to alleviate the cough and soothe the throat. 
- Steam inhalation: Suggest the patient try steam inhalation to help relieve any congestion and soothe the airways. 
- Allergy management: If allergic rhinitis is suspected, advise the patient to avoid triggers, use nasal saline rinses, and consider antihistamines or nasal corticosteroids. 
- Rest and hydration: Encourage the patient to get adequate rest and stay hydrated to support the body's healing process. 

Recommended Tasks:
- Doctor: Assess the severity of the cough and ask follow-up questions to gather more information about associated symptoms, such as fever, chest pain, or shortness of breath. 
- Doctor: Perform a physical examination, including listening to the patient's lungs, to assess any abnormal sounds or signs of infection. 
- Doctor: Consider ordering diagnostic tests, such as a chest X-ray or sputum culture, if necessary, to rule out other underlying conditions. 
"""

################
### Section: Run Audio Loop
################

st.divider()

# Storage values
client = storage.Client()
BUCKET_NAME = utils_config.BUCKET_NAME
path = "scribbles/"
bucket = client.bucket(BUCKET_NAME)

# Download audio file & deal with it
input_file_name = "recording2.mp3"
input_file_path = path + input_file_name
blob = bucket.blob(input_file_path)
audio_name = "audio_data.mp3"
blob.download_to_filename(audio_name)

# Download transcript as blob & deal with it
output_file_name = "recording2_transcript_6597e0aa-0000-20c9-bf3b-d4f547f5ae6c.json"
output_file_path = path + output_file_name
blob = str(bucket.blob(output_file_path).download_as_string())
blob = blob[2:][:-1].replace("\\n","").replace("\\","").replace("endTime","endOffset") ## remove first 2, last character; replace "\" escapes
json_object = json.loads(blob)
json_object2 = json_object["results"]

# Run through Loop
transcript = ""
for index, value in enumerate(json_object2):
    # Grab transcript + timestamp
    transcript_value = value["alternatives"][0]["transcript"]
    timestamp = float(value["alternatives"][0]["words"][0]["startOffset"][:-1])

    # Adjust Timestamp
    timestamp_adjust = timestamp * (49/90)
    timestamp_floor = math.floor(timestamp_adjust)
    
    # Play Audio
    st.header("Audio #" + str(index + 1))
    st.audio(audio_name, format="audio/mp3", start_time=timestamp_floor)
    
    # Show transcript values
    st.write(":blue[Transcript (Original):] " + transcript_value)
    
    # Task 1 - Correct Transcript
    st.write(":green[LLM Task 1 -- Identify Speaker & Improve Transcript:]")
    input_text = "transcript:    " + transcript_value
    
    prompt_diary2 = f"""
    {prompt_diary}

    input:
    {input_text}

    output:

    """
    vertexai.init(
        project = PROJECT_ID
        , location = LOCATION)
    parameters = {
        "temperature": model_temperature,
        "max_output_tokens": model_token_limit,
        "top_p": model_top_p,
        "top_k": model_top_k
    }
    model = TextGenerationModel.from_pretrained(model_id)
    response = model.predict(
        f'''{prompt_diary2}''',
        **parameters
    )
    llm_response = response.text
    st.markdown(llm_response)
    
    transcript_start = llm_response.find("LLM corrected transcript:") + len("LLM corrected transcript:")
    transcript_text = llm_response[transcript_start:].strip()
    speaker_start = llm_response.find("LLM predicted speaker:") + len("LLM predicted speaker:")
    speaker_end = llm_response.find("language_code") - 5
    speaker_text = llm_response[speaker_start:speaker_end].strip()
    transcript_text_full = "(" + speaker_text + ") " + transcript_text
    st.write(":blue[Transcript (Corrected):] " + transcript_text_full)

    # Task 2 - Provide Medical Advice
    transcript += '\n\n' + transcript_text_full # use cumulative transcript 
    input_text = "transcript: \n\n " + transcript
    st.write(":green[LLM Task 2 -- Provide Medical Advice:]")
    
    
    prompt_medical_advice2 = f"""
    {prompt_medical_advice}

    input:
    {input_text}

    output:

    """
    vertexai.init(
        project = PROJECT_ID
        , location = LOCATION)
    parameters = {
        "temperature": model_temperature,
        "max_output_tokens": model_token_limit,
        "top_p": model_top_p,
        "top_k": model_top_k
    }
    model = TextGenerationModel.from_pretrained(model_id)
    response = model.predict(
        f'''{prompt_medical_advice2}''',
        **parameters
    )
    llm_response = response.text
    st.markdown(llm_response)