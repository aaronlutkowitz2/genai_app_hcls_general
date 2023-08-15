#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation Date: July 10, 2023

@author: Aaron Wilkowitz
"""

################
### import 
################

# from google.cloud import bigquery
# import vertexai
# from vertexai.preview.language_models import TextGenerationModel
# from vertexai.preview.language_models import CodeGenerationModel
from google.cloud.dialogflowcx_v3beta1.services.agents import AgentsClient
from google.cloud.dialogflowcx_v3beta1.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3beta1.types import session
from google.api_core.client_options import ClientOptions

# # # others
# from langchain import SQLDatabase, SQLDatabaseChain
# from langchain.prompts.prompt import PromptTemplate
# # from langchain import LLM
# from langchain.llms import VertexAI
# from sqlalchemy import *
# from sqlalchemy.engine import create_engine
# from sqlalchemy.schema import *

import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
# from components import html

import argparse
import uuid
# import pandas as pd
# import db_dtypes 
# import ast
# from datetime import datetime
# import datetime, pytz
# import seaborn as sns
# import yaml 
# import aiohttp 

import re

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Website Info Bot')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-08-01')
st.write('**Purpose**: Pick a website and ask questions against that website')

# Gitlink
st.write('**Go Link (Googlers)**: go/hclsgenai')
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
# st.divider()
# st.header('30 Second Video')

# default_width = 80 
# ratio = 1
# width = max(default_width, 0.01)
# side = max((100 - width) / ratio, 0.01)

# _, container, _ = st.columns([side, width, side])
# container.video(data='https://youtu.be/NxlaDk6UYN4')

# Architecture

# st.divider()
# st.header('Architecture')

# components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSmJtHbzCDJsxgrrrcWHkFOtC5PkqKGBwaDmygKiinn0ljyXQ0Xaxzg4mBp2mhLzYaXuSzs_2UowVwe/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### Define function
################

# def detect_intent_texts(agent, session_id, texts, language_code, location_var):
#     """Returns the result of detect intent with texts as inputs.

#     Using the same `session_id` between requests allows continuation
#     of the conversation."""
#     session_path = f"{agent}/sessions/{session_id}"
#     # print(f"Session path: {session_path}\n")
#     st.write(":blue[Session path]: " + session_path)
#     client_options = None
#     agent_components = AgentsClient.parse_agent_path(agent)
#     location_id = location_var # agent_components["location"]
#     if location_id != "global":
#         api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
#         print(f"API Endpoint: {api_endpoint}\n")
#         client_options = {"api_endpoint": api_endpoint}
#     session_client = SessionsClient(client_options=client_options)

#     for text in texts:
#         text_input = session.TextInput(text=text)
#         query_input = session.QueryInput(text=text_input, language_code=language_code)
#         request = session.DetectIntentRequest(
#             session=session_path, query_input=query_input
#         )
#         response = session_client.detect_intent(request=request)

#         print("=" * 20)
#         print(f"Query text: {response.query_result.text}")
#         response_messages = [
#             " ".join(msg.text.text) for msg in response.query_result.response_messages
#         ]
#         print(f"Response text: {' '.join(response_messages)}\n")
#         st.write(response_messages)

def detect_intent_texts(agent, session_id, texts, language_code, location_id):
    
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"
    session_path = f"{agent}/sessions/{session_id}"
    agent_components = AgentsClient.parse_agent_path(agent)
    
    # location_id = agent_components["location"]
    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        print(f"API Endpoint: {api_endpoint}\n")
        client_options = {"api_endpoint": api_endpoint}
        # st.write("API Endpoint " + api_endpoint)
        print(f"API Endpoint: {api_endpoint}\n")

    client_options = None 
    session_client = SessionsClient(client_options=client_options)

    for text in texts:
      # Question
      st.write(":blue[**question:**] " + text)
      text_input = session.TextInput(text=text)

      # Set up query
      query_input = session.QueryInput(
          text=text_input
        , language_code=language_code
      )
      request = session.DetectIntentRequest(
          session=session_path, query_input=query_input
      )
      response = session_client.detect_intent(request=request)
      
      # Answer 
      response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
      ]
      response_text = response_messages[0]
      response_text = response_text.replace("$", "USD ")
      # st.text(response_text)
      st.write(":blue[**answer:**] " + response_text)

      # URL 
      answer_url_pre = str(response.query_result.response_messages[1]) # .payload) # 
      # st.write(":blue[**pre-URL:**] " + answer_url_pre)
      input_string = answer_url_pre
      # pattern = r'"actionLink" value { string_value: "((https?://[^"]+))" }'
      pattern = r'"((https?://[^"]+))"'
      # st.write(":blue[**pattern:**] " + pattern)
      match = re.search(pattern, input_string)
      # st.write(":blue[**match:**] " + str(match))

      if match:
          answer_url = match.group(1)
          # print(url)
          st.write(":blue[**Reference:**] " + answer_url)
      else:
          answer_url = "not found"
          st.write("URL not found")

################
### Select chatbot
################

st.divider()
st.header('Select Chatbot')

website_name = st.selectbox(
    'What website do you want a summary on?'
    , (
          "GCP"
        , "Ambetter Health"
        , "Website 3"
        , "Website 4"
      )
  )
st.write(':blue[**Chatbot:**] ' + website_name)

agent_id_gcp = "06687b2f-4a64-41e5-9ca6-b1f0cd3a6b91"
agent_id_ambetter = "1d903025-b6fb-4487-8162-f6e3fc6242bc"
agent_id_2 = "06687b2f-4a64-41e5-9ca6-b1f0cd3a6b91"

if website_name == "GCP":
   agent_id = agent_id_gcp
elif website_name == "Ambetter Health":
   agent_id = agent_id_ambetter
else: 
   agent_id = agent_id_gcp

################
### Select question
################

st.divider()
st.header('Select question')

custom_prompt = st.text_input('Write your question here', value = "What is " + website_name + "?")

project_id = "cloudadopt" 
location_id = "global"

session_id = uuid.uuid4()
texts = [custom_prompt]
language_code = "en-us"
detect_intent_texts(agent_id, session_id, texts, language_code, location_id)

