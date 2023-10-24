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
from google.cloud import storage

# others
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json 
import utils_config

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('Highlight Relevant Text')

# Author & Date
st.write('**Author**: Aaron Wilkowitz')
st.write('**Date**: 2023-09-11')
st.write('**Purpose**: Highlight Relevant Text')

# Video
st.divider()
st.header('30 Second Video')

# video_url = 'https://youtu.be/caJt0z6leIg'
# st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')
components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSWOR_T17wZLpnzjSn17IMN4V_pTSMNSxpZddjFTRfor2P7VeuXadXIO_-Q54Daf55zYI7d9Lb0qAQ1/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### model inputs
################

# Model Inputs
PROJECT_ID = utils_config.get_env_project_id()
LOCATION = utils_config.LOCATION
model_id = 'text-bison@001' 
model_temperature = 0.2
model_token_limit = 200 
model_top_k = 40
model_top_p = 0.8

################
### Highlight Relevant Text
################

st.divider()
st.header('1. Highlight Relevant Text')

# Select a topic
topic = st.text_input(
   'Pick any topic'
   , value = "radiology"
  )
st.write(':blue[**Topic:**] ' + topic)

# First LLM - write a paragraph
prompt_text_a = f'Write a 5-10 sentence paragraph on: what is {topic}'

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
    f'''{prompt_text_a}''',
    **parameters
)
llm_response_a = response.text

st.write(':blue[**Sample Text:**] ' + llm_response_a)

# Select question
question = st.text_input(
   'Pick any question on that topic'
   , value = f"What is {topic}?"
  )
st.write(':blue[**Question:**] ' + question)

# Return back answer & evidence
prompt_text_b_pre = f'''

Context: Below is a text passage. Below that is a question. Below that is a Below I have a json schema with a request for information in [brackets]. 

Request: Could you please fill out the json schema with the information requested? 

Rule 1: Output your response with json only. Do not include any text besides the json format below
Rule 2: Make sure your json is correctly formatted
Rule 3: Make sure the column names of your json are the same as the json format below, including capitalization
Rule 4: Only include information explicitly stated in the text passage
Rule 5: If any of the columns do not have data, include the column in your json response and return Null for that field's value

Text Passage: {llm_response_a}

Question: {question}

JSON Schema:

'''

json_schema = '''
{
    "answer": [answer the question based on the text passage]
    "evidence": [provide the exact text from the text passage that answers the question]
}
'''

prompt_text_b = prompt_text_b_pre + json_schema

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
    f'''{prompt_text_b}''',
    **parameters
)
llm_response_b = response.text

json_object = json.loads(llm_response_b)

answer = json_object["answer"]
evidence = json_object["evidence"]

st.write(':blue[**Answer:**] ' + answer)
st.write(':blue[**Evidence:**] ' + evidence)

# Highlight the relevant text
text_paragraph = llm_response_a
text_highlight = evidence
text_replace = ":orange[**" + text_highlight + "**]"
# text_replace = "<span style=\"background-color: yellow;\">" + text_highlight + "<span>"

text_paragraph_with_formatting = text_paragraph.replace(text_highlight, text_replace)

st.write(':blue[**Formatted Paragraph:**] ')
st.write(text_paragraph_with_formatting)