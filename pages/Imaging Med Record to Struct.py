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

# others
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import ast
from datetime import datetime
import datetime, pytz
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
st.title('GCP HCLS GenAI Demo: Medical Imaging -- Data Science Labels')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-06-22')
st.write('**Purpose**: Data science models needs to take in radiologist imaging summary & convert that to many yes/no labels for data science training.')

# Model Inputs
st.divider()
st.header('Intro')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vT-l1SQEqf6DrvlDMT_YhULvY74U1SnVCyfC7EVgXt2bPN4c6bejjPb0GeNjt4SHnz3v0t4SHjM-S-9/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

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
### LLM A
################

# LLM Model 1 -- Determine List of Labels
st.divider()
st.header('2. LLM A - Determine List of Labels')

# Body Part
param_body_part = st.selectbox(
    'What body part does the image refer to?'
    , (
          'chest'
        , 'head'
        , 'custom'
      )
    , index = 0
  )

custom_body_part = st.text_input('If you select custom, write a custom body part here')

if "custom" in param_body_part:
  body_part = custom_body_part
else:
  body_part = param_body_part

# Modality
param_modality = st.selectbox(
    'What modality does the image refer to?'
    , (
          'x-ray'
        , 'ct-scan'
        , 'mri'
        , 'custom'
      )
    , index = 0
  )

custom_modality = st.text_input('If you select custom, write a custom modality here')

if "custom" in param_modality:
  modality = custom_modality
else:
  modality = param_modality

body_part_modality = body_part + ' ' + modality

st.write(':blue[**Imaging Type:**] ' + body_part_modality)

input_prompt_a_1_context = '''You are an expert in medical imaging & in data science. We want to transform a radiologist\'s imaging summary into a series of binary ("yes" or "no") labels to use for training in a future data science predictive model. Can you please provide a comma separated list of the labels we should check for the below?

'''

input_prompt_a_2_examples = '''
input: chest x-ray
output: atelectasis, cardiomegaly, consolidation, effusion, hilar_enlargement, infiltrate, nodule, pneumonia, pleural_thickening, pneumothorax

input: head ct-scan
output: calcification, hemorrhage, hydrocephalus, mass, midline_shift, skull_fracture, subarachnoid_hemorrhage, subdural_hematoma

'''

input_prompt_a_3_input = 'input: ' + body_part_modality

input_prompt_a_4_output = '''
output:
'''

st.write(':blue[**Prompt 1:**] ')
llm_prompt_a_display = st.text(input_prompt_a_1_context + input_prompt_a_2_examples + input_prompt_a_3_input + input_prompt_a_4_output)
llm_prompt_a = input_prompt_a_1_context + input_prompt_a_2_examples + input_prompt_a_3_input + input_prompt_a_4_output

project_id = "cloudadopt"
location_id = "us-central1"

# Run the first model
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
    f'''{llm_prompt_a}''',
    **parameters
)
# print(f"Response from Model: {response.text}")

llm_response_text_a = response.text

st.write(':blue[**List of Labels:**] ' + llm_response_text_a)

st.divider()

################
### LLM B
################

# LLM Model 2 -- Create binary labels
st.header('3. LLM B - Create binary label based on Medical Text')

# File Name
file_id = st.selectbox(
    'What file do you want to use?'
    , (
          "Imaging Result 1"
        , "Imaging Result 2"
        , "Imaging Result 3"
      )
  )

# File Text
if file_id == "Imaging Result 1":
  file_text = '''[
 {
   "key": "MeSH",
   "value": "normal"
 },
 {
   "key": "Problems",
   "value": "normal"
 },
 {
   "key": "image",
   "value": "Xray Chest PA and Lateral"
 },
 {
   "key": "indication",
   "value": "Positive TB test"
 },
 {
   "key": "comparison",
   "value": "None."
 },
 {
   "key": "findings",
   "value": "The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax."
 },
 {
   "key": "impression",
   "value": "Normal chest x-XXXX."
 }
]'''
elif file_id == "Imaging Result 2":
  file_text = '''[
 {
   "key": "MeSH",
   "value": "Opacity/lung/base/left/mild;Implanted Medical Device;Atherosclerosis/aorta;Calcinosis/lung/hilum/lymph nodes;Calcinosis/mediastinum/lymph nodes;Spine/degenerative/mild;Granulomatous Disease"
 },
 {
   "key": "Problems",
   "value": "Opacity;Implanted Medical Device;Atherosclerosis;Calcinosis;Calcinosis;Spine;Granulomatous Disease"
 },
 {
   "key": "image",
   "value": "Xray Chest PA and Lateral"
 },
 {
   "key": "indication",
   "value": "XXXX-year-old male with XXXX for 3 weeks. Possible pneumonia."
 },
 {
   "key": "comparison",
   "value": ""
 },
 {
   "key": "findings",
   "value": "There are minimal XXXX left basilar opacities, XXXX subsegmental atelectasis or scarring. There is no focal airspace consolidation to suggest pneumonia. No pleural effusion or pneumothorax. Heart size is at the upper limits of normal. Cardiac defibrillator XXXX overlies the right ventricle. The XXXX appears intact. There is aortic atherosclerotic vascular calcification. Calcified mediastinal and hilar lymph XXXX are consistent with prior granulomatous disease. Multiple calcified splenic granulomas are also noted. There are minimal degenerative changes of the spine."
 },
 {
   "key": "impression",
   "value": "Minimal left basilar subsegmental atelectasis or scarring. No acute findings."
 }
]'''
elif file_id == "Imaging Result 3":
  file_text = '''[
 {
   "key": "MeSH",
   "value": "Pneumonia/upper lobe/left;Airspace Disease/lung/upper lobe/left"
 },
 {
   "key": "Problems",
   "value": "Pneumonia;Airspace Disease"
 },
 {
   "key": "image",
   "value": "Chest PA and lateral views. XXXX, XXXX XXXX PM"
 },
 {
   "key": "indication",
   "value": "XXXX with XXXX"
 },
 {
   "key": "comparison",
   "value": "none"
 },
 {
   "key": "findings",
   "value": "XXXX XXXX and lateral chest examination was obtained. The heart silhouette is normal in size and contour. Aortic XXXX appear unremarkable. Lungs demonstrate left upper lobe airspace disease most XXXX pneumonia. There is no effusion or pneumothorax."
 },
 {
   "key": "impression",
   "value": "1. Left upper lobe pneumonia."
 }
]'''
else:
  file_text = "unknown"

st.write(':blue[**File ID:**] ' + file_id)
st.write(':blue[**File Text:**] ' + file_text)

# Convert comma separated list into list
list_of_labels_to_create = llm_response_text_a.split(',')

################
### LLM B Output
################

# Create dataframe & start loop
# note: must do this step b/c data_tables function doesn't work on streamlit when on cloud run, not sure why
fake_data = {
      'label_name' : '.'
    , 'label_outcome' : '.'
}
df_fake_schema = pd.DataFrame(fake_data, index=[0])
data_table = st.table(df_fake_schema)

for indexA, valueA in enumerate(list_of_labels_to_create):
  input_prompt_b_1_context = '''You are an expert in medical imaging. Below is a yes or no question, followed by a radiologist's imaging summary.

  Can you please answer the question **WITH ONLY** "yes" or "no"? Thank you.

  '''

  input_prompt_b_2_question = 'Question: Does the below imaging summary show evidence of ' + list_of_labels_to_create[indexA] + '?'

  input_prompt_b_3_file_text = '''
  Imaging Summary:

  ''' + file_text

  llm_prompt_b = input_prompt_b_1_context + input_prompt_b_2_question + input_prompt_b_3_file_text

  # Run the first model
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
      f'''{llm_prompt_b}''',
      **parameters
  )
  # print(f"Response from Model: {response.text}")

  llm_response_text_b = response.text

  # Create dataframe
  testdict = {
      'label_name': list_of_labels_to_create[indexA]
    , 'label_outcome': llm_response_text_b
  }
  # Append to df using dict
  df_row = pd.DataFrame(testdict, index=[0])
  data_table.add_rows(df_row)