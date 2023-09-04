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
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

import utils_config

# others
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import pandas as pd
import ast
from datetime import datetime
import datetime, pytz
import seaborn as sns

# looker 
import looker_sdk #Note that the pip install required a hyphen but the import is an underscore.
import os #We import os here in order to manage environment variables for the tutorial. You don't need to do this on a local system or anywhere you can more conveniently set environment variables.
import json #This is a handy library for doing JSON work.

from looker_sdk import models as mdls
from looker_sdk.rtl import api_methods
from looker_sdk.rtl import transport

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: HDE Looker GenAI Demo')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-06-22')
st.write('**Purpose**: Show how a user can pull up a FHIR record from a Looker API call, then use GenAI to ask a question about the patient.')

# Gitlink
st.write('**Go Link (Googlers)**: go/hclsgenai')
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('30 Second Video')

video_url = 'https://youtu.be/Zhwk_2McSu4'
st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vRE3vXvZ2c3HuqnXVU276wYSnwHnm_kuXWy0Jr4WrI74u3zgwGu3sMzrIOzhSyreHaSe2m-eGyn0odY/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### model inputs
################

# Model Inputs
st.divider()
st.header('1. Model Inputs')

model_id = st.selectbox(
    'Which model do you want to use?'
    , (
          'text-bison@001'
        , 'text-bison@latest'
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
    , value = 200
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
### Looker Inputs
################

# Looker inputs
st.divider()
st.header('2. Looker Inputs')

looker_sdk_base_url = "https://demoexpo.cloud.looker.com:19999"
looker_env_str = "demoexpo"
os.environ["LOOKERSDK_BASE_URL"] = looker_sdk_base_url #If your looker URL has .cloud in it (hosted on GCP), do not include :19999 (ie: https://your.cloud.looker.com).
os.environ["LOOKERSDK_API_VERSION"] = "4.0" #3.1 is the default version. You can change this to 4.0 if you want.
os.environ["LOOKERSDK_VERIFY_SSL"] = "true" #Defaults to true if not set. SSL verification should generally be on unless you have a real good reason not to use it. Valid options: true, y, t, yes, 1.
os.environ["LOOKERSDK_TIMEOUT"] = "120" #Seconds till request timeout. Standard default is 120.

#Get the following values from your Users page in the Admin panel of your Looker instance > Users > Your user > Edit API keys. If you know your user id, you can visit https://your.looker.com/admin/users/<your_user_id>/edit.

## note: this login info is for Aaron Wilkowitz' dev Looker environment; to use your own Looker work, you'll need to update it with your API keys
os.environ["LOOKERSDK_CLIENT_ID"] =  "vt3pSjGRf8vVVKPcq9dK" #No defaults.
os.environ["LOOKERSDK_CLIENT_SECRET"] = "vkthnv8GJzHgQgnGjZVny2zF" #No defaults. This should be protected at all costs. Please do not leave it sitting here, even if you don't share this document.

sdk = looker_sdk.init40()

st.write(':blue[**Looker Instance:**] ' + looker_env_str)
st.write(':green[**Looker Environment Ready**]')

patient_name = st.selectbox(
    'What patient do you want to review?'
    , (
          "Patient 1"
        , "Patient 2"
        , "Patient 3"
        , "Patient 4"
      )
  )
st.write(':blue[**Patient Name:**] ' + patient_name)

if patient_name == "Patient 1": 
    patient_id = "50ac180c-abc5-4cef-ad6f-6585e40c69a1"
elif patient_name == "Patient 1": 
    patient_id = "2e7adb9a-f170-4b83-91d3-08579f2b802d"
elif patient_name == "Patient 1": 
    patient_id = "82283e1b-bf97-445c-b935-c5cae6dd3b19"
elif patient_name == "Patient 1": 
    patient_id = "1fb6ef17-24fc-46a0-affa-eec4fd3d1917"
else: 
   patient_id = "50ac180c-abc5-4cef-ad6f-6585e40c69a1"
st.write(':blue[**Patient ID:**] ' + patient_id)

################
### Looker Data
################

# Looker data
st.divider()
st.header('3. Looker Data')

query_body = mdls.WriteQuery(
        model="fhir_hcls",
        view="fhir_hcls",
        fields=[
            "encounter.id",
            "analytics.admission_date",
            "analytics.discharge_date",
            "analytics.encounter_code",
            "analytics.encounter_type_format",
            "analytics.encounter_reason_for_visit",
            "analytics.patient_name",
            "analytics.patient_gender",
            "analytics.patient_age",
            "analytics.patient_language",
            "analytics.patient_race",
            "analytics.organization_name",
            "condition__code.text",
            "condition__code__coding.code",
            "condition.clinical_status",
            "analytics.bmi",
            "analytics.height_ft",
            "analytics.weight_lb",
            "analytics.practitioner_name",
            "analytics.procedure_code",
            "analytics.procedure_name"
        ],
        filters={
            "encounter.id": f"{patient_id}", 
            "analytics.admission_date": "NOT NULL"
        },
        limit="1",
        query_timezone="America/Los_Angeles"
    )

## Grab the query json
query_json_pre = sdk.run_inline_query(
result_format="json",
body=query_body
)

query_json = query_json_pre.replace(",", ", \n")

query_sql = sdk.run_inline_query(
result_format="sql",
body=query_body
)

st.write(':blue[**SQL Query:**] ')
st.text(query_sql)

st.write(':blue[**Patient Information:**] ')
st.text(query_json)

################
### LLM Prompt
################

# Looker data
st.divider()
st.header('4. LLM Prompt')

# Query Goal
param_goal = st.selectbox(
    'What do you want to know about the patient?'
    , (
          'Write a 3 sentence summary of the patient'
        , 'When was the patient admitted?'
        , 'What hospital was the patient admitted to?'
        , 'custom'
      )
    , index = 0
  )

custom_goal = st.text_input('If you select custom, write a custom body part here')

if "custom" in param_goal:
  goal = custom_goal
else:
  goal = param_goal
st.write(':blue[**Goal:**] ' + goal)

input_prompt_a_1_context = f'''
Below is a patient\'s medical record. 

{goal}

''' 

input_prompt_a_2_data = query_json

st.write(':blue[**Prompt:**] ')
llm_prompt = input_prompt_a_1_context + input_prompt_a_2_data
llm_prompt_display = st.text(llm_prompt)

################
### LLM Output
################

# Looker data
st.divider()
st.header('5. LLM Output')

PROJECT_ID = utils_config.get_env_project_id()
LOCATION = utils_config.LOCATION

# Run the first model
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
    f'''{llm_prompt}''',
    **parameters
)
# print(f"Response from Model: {response.text}")

llm_response_text = response.text

st.write(':blue[**LLM Response:**] ')
st.write(llm_response_text)