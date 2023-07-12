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

# others 
import streamlit as st
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
bucket_name = "hcls_genai"
path = 'hcls/supply_chain/'
bucket = client.bucket(bucket_name)
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
bucket = client.bucket(bucket_name)
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

project_id = "cloudadopt"
location_id = "us-central1"

import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

vertexai.init(
      project = project_id
    , location = location_id)
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
