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
from google.cloud import storage

import utils_config

# others
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import pandas as pd
from datetime import datetime
from datetime import date
import os 
import json 
from urllib.parse import urlparse, parse_qs

# looker 
import looker_sdk 
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
st.title('GCP HCLS GenAI Demo: Looker Explore Generations')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-09-21')
st.write('**Purpose**: Type in a question and generate an explore.')

# Gitlink
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('30 Second Video')

video_url = 'https://youtu.be/caJt0z6leIg'
st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

# /pub?start=false&loop=false&delayms=3000

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vTWVPJdNWqw-pt1K-UZiGs28G---6hsL5MlCvdH_fj6e7aXUrHwoXOfyGrF1Be0HZ08UIKb0ng0F_jL/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### model inputs
################

# Model Inputs
PROJECT_ID = utils_config.get_env_project_id()
LOCATION = utils_config.LOCATION
model_id = 'text-bison@001' 
model_temperature = 0
model_token_limit = 100 
model_top_k = 1
model_top_p = 0

# Create model function
vertexai.init(project=PROJECT_ID, location=LOCATION)
parameters = {
    "temperature": model_temperature,
    "max_output_tokens": model_token_limit,
    "top_p": model_top_p,
    "top_k": model_top_k
}
model = TextGenerationModel.from_pretrained(model_id)

################
### Looker Inputs
################

# Looker inputs
st.divider()
st.header('1. Looker Inputs')

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

################
### Select Model & Explore
################

# Looker inputs
st.divider()
st.header('2. Select Model & Explore')

model_name = st.selectbox(
    'What Looker model do you want to build on?'
    , (
          "faa", "fhir_hcls" ### Update later: create a list of models based on API call
      )
    , index = 1
  )
st.write(':blue[**Model Name:**] ' + model_name)

explore_name = ''

if model_name == "faa":
    explore_name = "flights"
elif model_name == "fhir_hcls":
    explore_name = "fhir_hcls_simple"
else: 
    explore_name = "x"

st.write(':blue[**Explore Name:**] ' + explore_name)

################
### Bring in list of dimensions and measures
################

# Select question
st.divider()
st.header('3. Bring in dimensions & measures')

client = storage.Client()
bucket_name = 'hcls_genai'
path = 'looker/model_data_dictionary/'
file_name = path + model_name + '_' + explore_name + '_model_data_dictionary.txt'
bucket = client.bucket(bucket_name)
blob = bucket.blob(file_name)

# Function to pull list of fields + save to GCS
def pull_list_of_fields(model_name, explore_name):
    sdk = looker_sdk.init40()

    # API Call to pull in metadata about fields in a particular explore
    explore = sdk.lookml_model_explore(
        lookml_model_name=model_name,
        explore_name=explore_name,
        fields="id, name, description, fields, label",
    )

    measures = []
    dimensions = []

    # Pull dimension name + 5 most common values
    count_dimension = explore_name + '.count' 
    sort_text = count_dimension + ' desc'
    if explore.fields and explore.fields.dimensions:
        for dimension in explore.fields.dimensions:
            query_body = mdls.WriteQuery(
                    model=model_name,
                    view=explore_name,
                    fields=[
                        dimension.name, 
                        count_dimension
                    ],
                    limit="5", 
                    sorts=[
                        sort_text
                    ]
                )

            query_json_pre = sdk.run_inline_query(
            result_format="json",
            body=query_body
            )

            query_json_str = query_json_pre.replace("\n", "").strip()
            query_json = json.loads(query_json_str)
            try:
                top_5_values = '|'.join([d[dimension.name] for d in query_json])
            except:
                continue 

            def_dimension = {
                "name": dimension.name, 
                "top_5_values":top_5_values
            }
            st.write(def_dimension)
            dimensions.append(def_dimension) 

    dimensions_json = json.dumps(dimensions)

    if explore.fields and explore.fields.measures:
        for measure in explore.fields.measures:
          def_measure = {
             "name": measure.name
          }
          measures.append(def_measure)

    measures_json = json.dumps(measures)

    # Save as a text file in GCS bucket
    
    file_text = f'''

    **** List of Dimensions & Their Most Common 5 Values ****

    {dimensions_json}

    **** List of Measures ****

    {measures_json}

    '''
    st.write(file_text)

    blob.upload_from_string(file_text)

    st.write(f"String saved as {file_name} in {bucket_name}")


# Pull down file -- if it doesn't work, run the function to create a new file
try: 
    file_content = blob.download_as_text()
except: 
    pull_list_of_fields(model_name, explore_name)
    file_content = blob.download_as_text()

st.write(':blue[**List of Dimensions & Measures:**] ')
st.write(file_content)

################
### Select the question
################

# Select question
st.divider()
st.header('4. Select Question')

faa_questions = [
    "What airlines have the highest percent delays at BNA?"
    ,"What airports have the longest average travel time? Narrow down to large airports (> 10000 flights)"
    , "What were the most common destinations out of LAX in 2003?"
    , 'custom'
]

fhir_questions = [
      "Which 10 hospitals had the highest number of COVID cases in the last 7 days?"
    , "How many confirmed or suspected COVID cases are there in our hospitals?"
    , "Show me the demographics of my patients by gender by age"
    , 'custom'
]

faa_examples = '''
input: What airlines have the highest percent delays at BNA?
output: model="faa",view="flights",fields=["carriers.name","flights.percent_flights_delayed_clean"],filters={"flights.origin": "BNA"},vis_config={"type": "looker_grid"}

input: What airports have the longest average travel time? Narrow down to large airports (10000 flights)
output: model="faa",view="flights",fields=["flights.origin","flights.average_flight_length"],filters={"flights.flight_count": ">10000"},vis_config={"type": "looker_column"},limit=10

input: What were the most common destinations out of LAX in 2003?
output: model="faa",view="flights",fields=["destination.city_full","flights.flight_count"],filters={"flights.origin": "LAX", "flights.dep_date": "2003"},vis_config={"type": "looker_column"}
'''

fhir_examples = '''
input: Which 10 hospitals had the highest number of COVID cases in the last 7 days?
output: model="fhir_hcls",view="fhir_hcls_simple",fields=["analytics.count_total_patients","analytics.organization_name"],filters={"analytics.admission_date": "7 days","analytics.covid_status": "Confirmed"},vis_config={"type": "looker_grid"},limit=10'

input: How many confirmed or suspected COVID cases are there in our hospitals?
output: model="fhir_hcls",view="fhir_hcls_simple",fields=["analytics.count_total_patients"],filters={"analytics.covid_status": "Confirmed,Suspected"},vis_config={"type": "single_value"}'

input: Show me the demographics of my patients by gender by age
output: model="fhir_hcls",view="fhir_hcls_simple",fields=["analytics.count_total_patients","analytics.patient_gender","analytics.patient_age_tier"],pivots=["analytics.patient_gender"],sorts=["analytics.patient_age_tier"],vis_config={"type": "looker_bar"}'
'''

# input: Show me the demographics of my patients by gender by age
# output: 

if model_name == "faa":
    question_list = faa_questions
    examples = faa_examples
elif model_name == "fhir_hcls":
    question_list = fhir_questions
    examples = fhir_examples
else: 
    question_list = []

param_question = st.selectbox(
    'What question do you want to know more about?'
    , (question_list)
  )

if param_question == 'custom':
    custom_goal = st.text_input('If you select custom, write a custom question here')
    question = custom_goal
else: 
    question = param_question
st.write(':blue[**Question:**] ' + question)

################
### Create the Looker explore
################

# Create Looker query
st.divider()
st.header('5. Create Looker query')

# Prompt
prompt = f'''

*** Context: *** 
You are a developer who needs to translate questions into a structured API-based query. 
Below is a question. 
Below that is a list of the dimensions & measeures you can reference. The dimensions include 5 sample values.
Below that is a list of example questions & the API-based structure that is required for it.

*** Rules: *** 
1. Follow the format of the examples below.
2. Only use fields included in the list of dimensions & measures 
3. Only provide the output in your response
4. Do not provide the input in your response
5. Do not use the word "output" in your response - just provide the output

*** Question *** 
{question}

{file_content}

*** Examples *** 
{examples}

input: {question}
output: 

'''

# st.write(prompt)

# Run the LLM
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
    f'''{prompt}''',
    **parameters
)
llm_response = response.text

# API Call
api_call = llm_response.replace("'","")
st.write(':blue[**API Call (GenAI Geneated):**] ' + api_call)

# Query body
query_body_string = f'query_body = mdls.WriteQuery({api_call})'
exec(query_body_string)

# Grab Query ID
response = sdk.create_query(
    body=query_body
)

query_id = response.id

st.write(':blue[**Looker Query ID:**] ' + str(query_id))


## Grab the Query sql
response = sdk.run_inline_query(
    result_format="sql",
    body=query_body
)

sql_query = response

st.write(':blue[**Looker Generated SQL Query:**] ')

st.text(sql_query)

################
### Create the public Look
################

# Create public look
st.divider()
st.header('6. Create the public look')

### Delete all existing looks
st.write(':orange[**Delete Existing Looks:**] ')
response = sdk.all_looks(fields="id, title")

for r in response: 
    if "(ASW GenAI Demo)" in r.title: 
        sdk.delete_look(look_id=r.id)
        st.write(r.title + " is deleted")
    else:
        continue 

### Create a new look
st.write(':orange[**Create a New Look:**] ')

today_date = date.today()
current_time = datetime.now().strftime("%I:%M:%S %p")

param_title = f"{model_name} -- {explore_name} -- {param_question} -- {today_date} -- {current_time} (ASW GenAI Demo)"
param_user_id = "vt3pSjGRf8vVVKPcq9dK"
param_public = True
param_query_id = query_id
param_folder_id = "528"

response = sdk.create_look(
    body=mdls.WriteLookWithQuery(
        title=param_title,
        user_id=param_user_id,
        public=param_public,
        query_id=param_query_id,
        folder_id=param_folder_id
    )
    ,fields="embed_url"
    )

look_url = response.embed_url 

st.write(':green[**New Looker Look Generated**] ')
st.write(':blue[**Title**] ' + param_title)
st.write(':blue[**URL**] ' + look_url)

def show_looker_look(url):
    components.html(f'''
        <iframe src={url} 
        width='600' 
        height='450' 
        frameborder='0'>
        </iframe>
    '''
    , width=620
    , height=475
    )
show_looker_look(look_url)