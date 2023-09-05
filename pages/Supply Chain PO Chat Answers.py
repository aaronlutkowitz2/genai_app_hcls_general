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
Creation Date: July 11, 2023

@author: Aaron Wilkowitz
"""

################
### import 
################
# gcp 
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
from google.cloud import storage

import utils_config

# others 
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import pandas as pd
from datetime import datetime
import datetime, pytz
import ast
import seaborn as sns

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Supply Chain PO Order - Q&A')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-07-11')
st.write('**Purpose**: Hospital has many supply chain PO orders. They need to ask questions against the order')

# Gitlink
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('30 Second Video')

video_url = 'https://youtu.be/VovFVC3pUWM'
st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vTO0Fl_ccAxfXAAZnKSjFz8eybLDpAKgO3VSR6DpZ7XPY8G83mqdwP6kRxdriQv5pitXIxsVchYFiwD/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### model inputs
################

# Model Inputs
st.divider()
st.header('1. Model Inputs')

model_id = st.selectbox(
    'Which model do you want to use?'
    , (
          'chat-bison@001'
        , 'chat-bison@latest'
      )
    , index = 0
  )
model_temperature = st.number_input(
      'Model Temperature'
    , min_value = 0.0
    , max_value = 1.0
    , value = 0.2
  )
model_token_limit = st.number_input(
      'Model Token Limit'
    , min_value = 1
    , max_value = 1024
    , value = 100
  )
model_top_k = st.number_input(
      'Top-K'
    , min_value = 1
    , max_value = 40
    , value = 40
  )
model_top_p = st.number_input(
      'Top-P'
    , min_value = 0.0
    , max_value = 1.0
    , value = 0.8
  )

################
### data
################

# Prompt Inputs
st.divider()
st.header('2. Data')

# Download Files
client = storage.Client()
BUCKET_NAME = utils_config.BUCKET_NAME
path = 'hcls/supply_chain/'
bucket = client.bucket(BUCKET_NAME)
blobs_all = list(bucket.list_blobs(prefix=path))

# Let user select file
blob_name2 = ''
blob_options_string = ''
for blob in blobs_all: 
    blob_name = blob.name
    blob_name = blob_name.replace(path,'')
    blob_name = '\'' + blob_name + '\''
    blob_options_string = blob_options_string + blob_name + ','
blob_options_string = blob_options_string[:-1]
blob_options_string = blob_options_string[3:]

string1 = 'order_id = st.selectbox('
string2 = '\'What order do you want to learn about?\''
string3 = f', ({blob_options_string}), index = 0)'
string_full = string1 + string2 + string3
exec(string_full)

file_name = order_id 
full_file_name = path + file_name
bucket = client.bucket(BUCKET_NAME)
blob = str(bucket.blob(full_file_name).download_as_string())
st.write(':green[**Complete**] File Downloaded')
st.write(':blue[**Order Data**]')
st.write(blob)

################
### LLM prompt & output
################

# Prompt Inputs
st.divider()
st.header('3. LLM Prompt & Output')

custom_prompt = st.text_input('Write your question here', value = "What was the date of this order?")

# Run the model
# Set project parameters
PROJECT_ID = utils_config.get_env_project_id()
LOCATION = utils_config.LOCATION


import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

vertexai.init(
      project = PROJECT_ID
    , location = LOCATION)
chat_model = ChatModel.from_pretrained(model_id)
parameters = {
    "temperature": 0.2,
    "max_output_tokens": 256,
    "top_p": 0.8,
    "top_k": 40
}
chat = chat_model.start_chat(
    context=blob,
)
response = chat.send_message(custom_prompt, **parameters)
# print(f"Response from Model: {response.text}")
llm_response_text = response.text 
st.write(':blue[**LLM Response:**] ' + llm_response_text)

################
### LLM JSON Converter
################

# Prompt Inputs
st.divider()
st.header('4. LLM JSON Converter')

instructions_text = """ 
You are an expert at writing json correctly. Please write a JSON blob out of the below document. Use correct json syntax.

"""

document_text = blob

# st.write(':blue[**Instructions:**] ')
# st.text(instructions_text)
# st.write(':blue[**Documentation:**] ')
# st.text(document_text)
# st.divider()

input_prompt = instructions_text + document_text
st.write(':blue[**LLM Prompt:**] ')
st.text(input_prompt)

st.divider()

model_id = "text-bison@001"
model_token_limit = 1024

# Run the model
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
    f'''{input_prompt}''',
    **parameters
)
# print(f"Response from Model: {response.text}")

llm_response_text = response.text 
st.write(':blue[**LLM Output:**]')
st.text(llm_response_text)