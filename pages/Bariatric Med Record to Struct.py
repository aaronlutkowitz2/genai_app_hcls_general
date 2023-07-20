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
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import vertexai
from vertexai.language_models import TextGenerationModel
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
st.title('GCP HCLS GenAI Demo: CCDA Bariatrics Use Case')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-06-21')
st.write('**Purpose**: Hospital needs to know if a patient has ever had a history of bariatric surgery at another hospital system, by reviewing a patient\'s CCDA document.')

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
### prompt inputs
################

# Prompt Inputs
st.divider()
st.header('2. Prompt Inputs')

# Goal
param_goal = st.selectbox(
    'What do you want to determine about the patient?'
    , (
          'bariatric surgery'
        , 'appendectomy'
        , 'pregnancy'
        , 'custom'
      )
    , index = 0
  )

custom_goal = st.text_input('If you select custom, write a custom question here')

if "custom" in param_goal:
  goal = custom_goal
else:
  goal = 'Has the patient had a ' + param_goal + '?'

st.write(':blue[**Goal:**] ' + goal)

st.divider()

# File Name
file_id = st.selectbox(
    'What file do you want to use?'
    , (
          "CCDA Sample 1"
        , "CCDA Sample 2"
        , "FHIR Sample 1"
      )
  )

if file_id == "CCDA Sample 1":
  file_name = "synthetic-bariatric-ccda-full-9580-lines.xml"
elif file_id == "CCDA Sample 2":
  file_name = "synthetic-bariatric-ccda-full-9580-linesv2.xml"
elif file_id == "FHIR Sample 1":
  file_name = "synthetic-bariatric-fhir-full-24527-lines.json"
else:
  file_name = "unknown"

st.write(':blue[**File ID:**] ' + file_id)
st.write(':blue[**File Name:**] ' + file_name)

# Download File
project_id = "cloudadopt"
location_id = "us-central1"
bucket_name = "hcls_genai"
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = str(bucket.blob(file_name).download_as_string())
st.write(':green[**Complete**] File Downloaded')

blob_sample = blob[:1000]
st.write(':blue[**File Sample:**] ' + blob_sample)

# Generate AI Prompt
input_prompt_1_context = '''Below I have a json schema with a request for information in [brackets]. Below that I have a medical history of a patient. Could you please fill out the json schema with the medical history provided? Thank you.

'''

input_prompt_2_prompt = f'''Question: {goal}

'''

input_prompt_3_schema = '''{
  "binary": [provide a "yes" or "no" only],
  "evidence": [provide the line in the document that gives you the evidence of your answer. include relevant medical codes],
  "code": [if the answer is "yes", provide the relevant medical code; if the answer is "no", leave this blank ""],
  "code_system": [if the answer is "yes", provide the relevant medical coding system -- e.g. "SNOMED", "CPT"; if the answer is "no", leave this blank ""]
}
'''

input_prompt_4_text_body = '''Medical History:

'''

st.divider()

st.write(':blue[**Prompt:**] ')
llm_prompt_display = st.text(input_prompt_1_context + input_prompt_2_prompt + input_prompt_3_schema + '[CCDA text Doc]')

# Prep for loop

# Division Style
if "CCDA" in file_id:
  # Look for "title" in CCDA document
  section_divider_start = '<title>'
  section_divider_end = '</title>'
elif "FHIR" in file_id:
  section_divider_start = 'resourceType'
  section_divider_end = '\",'
else:
  section_divider_start = 'XYZ XYZ -- do not have a section divider'
  section_divider_end = 'XYZ XYZ -- do not have a section divider'

num_sections = blob.count(section_divider_start)+1

################
### LLM Output
################

# Loop -- outputs
st.header('3. LLM Output')

# Set df

# note: must do this step b/c data_tables function doesn't work on streamlit when on cloud run, not sure why
fake_data = {
      'section' : '.'
    , 'llm_response' : '.'
    , 'binary' : '.'
    , 'evidence' : '.'
    , 'code' : '.'
    , 'code_system' : '.'
}
df_fake_schema = pd.DataFrame(fake_data, index=[0])
data_table = st.table(df_fake_schema)

# Start loop
i = 1

# while i <= 3: 
while i <= num_sections :

  # Generate text
  text = blob.split(section_divider_start)[i-1]
  # print(text)

  # Generate section title (CCDA only)
  num_char_section_title = text.find(section_divider_end)
  if num_char_section_title > 0:
    title = text[:num_char_section_title].replace("\"","").replace(":","")
  else:
    title = 'unknown'
  # print(title)

  # Generate length
  length = len(text)
  # print(length)

  # Separate the text every 1000 characters
  n = 14000
  text_shortened = [text[i:i+n] for i in range(0, len(text), n)]

  # Start a for loop to run through every element in the text splitter
  for index, value in enumerate(text_shortened):
    if __name__ == '__main__':

      # Skip the 2 biggest sections
      if title == 'Diagnostic Results' or title == 'DiagnosticResults' or title == 'Vital Signs' or title == 'VitalSigns' or title == 'Medications' or title == 'Problems' or title == 'unknown' or title == 'C-CDA R2.1 Patient Record Joe656 Lynch190' :
      # if title != 'Surgeries':
        pass
      else:
        # Track the index #
        index_num = title + '|' + str(index)
        # print(index_num)

        # Generate text for model
        input_prompt_4_text_body_new = input_prompt_4_text_body + text_shortened[index]
        input_prompt = input_prompt_1_context + input_prompt_2_prompt + input_prompt_3_schema + input_prompt_4_text_body_new

        length_text = len(input_prompt_4_text_body_new)

        # Run the model

        vertexai.init(
              project = project_id
            , location = location_id)
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

        # if response does not start with bracket, remove text before it
        if "{" in llm_response_text:
          char_start = llm_response_text.find('{')
          llm_response_text = llm_response_text[char_start:]
        else:
          pass

        # if response is not a dict, assume it's a no
        try:
            llm_response_dict = ast.literal_eval(llm_response_text)
        except:
            llm_response_dict = {'binary': 'no', 'evidence': '', 'code': '', 'code_system': ''}

        llm_response_binary = llm_response_dict['binary']
        llm_response_evidence = llm_response_dict['evidence']
        llm_response_code = str(llm_response_dict['code'])
        llm_response_code_system =  llm_response_dict['code_system']

        if llm_response_binary == 'yes' and 'snomed' in llm_response_code_system.lower() and llm_response_code == '18692006':
          valid_check = 'yes'
        elif llm_response_binary == 'no':
          valid_check = ''
        else:
          valid_check = 'no'

        if llm_response_binary == 'yes' and 'snomed' in llm_response_code_system.lower() and llm_response_code == '18692006':
          display_name = 'bypass gastroenterostomy'
        elif llm_response_binary == 'no':
          display_name = ''
        else:
          display_name = 'unknown'

        # Create dict
        df_dict = {
              'section': title
            , 'llm_response': llm_response_text[:500]
            , 'binary': llm_response_binary
            , 'evidence': llm_response_evidence
            , 'code': llm_response_code
            , 'code_system': llm_response_code_system
            , 'valid_check': valid_check
            , 'display_name': display_name
        }

        # Write out full response

        if llm_response_binary == 'yes':
          response_format = '**' + title + '** :orange[Yes, evidence!]' + ' - There is evidence of' + llm_response_code + ' from ' + llm_response_code_system
        else:
          # response_format = '**' + title + '** No evidence'
          response_format = ''

        if valid_check == 'yes':
          validation_check = ' **Hallucination check**: This code is :green[Valid] and refers to *bypass gastroenterostomy*'
        elif valid_check == '':
          validation_check = ''
        else:
          validation_check = ' **Hallucination check**: This code is :red[NOT Valid] -- check for possible hallucination'

        st.write(response_format)
        st.write(validation_check)
      
        # Append to df using dict
        df_row = pd.DataFrame(df_dict, index=[0])
        data_table.add_rows(df_row)

  # Loop through again
  i += 1