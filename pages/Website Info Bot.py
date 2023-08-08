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
st.divider()
st.header('30 Second Video')

default_width = 80 
ratio = 1
width = max(default_width, 0.01)
side = max((100 - width) / ratio, 0.01)

# _, container, _ = st.columns([side, width, side])
# container.video(data='https://youtu.be/NxlaDk6UYN4')

# Architecture

st.divider()
st.header('Architecture')

# components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSmJtHbzCDJsxgrrrcWHkFOtC5PkqKGBwaDmygKiinn0ljyXQ0Xaxzg4mBp2mhLzYaXuSzs_2UowVwe/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

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
      st.write(":blue[**question: **]" + text)
      text_input = session.TextInput(text=text)

      query_input = session.QueryInput(
          text=text_input
        , language_code=language_code
      )

      request = session.DetectIntentRequest(
          session=session_path, query_input=query_input
      )

      response = session_client.detect_intent(request=request)
      response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
      ]
      response_text = response_messages[0]
      print(response_text)
      st.write(":blue[**answer: **]" + response_text)

project_id = "cloudadopt" 
location_id = "global"
agent_id = "299df307-e528-4fbd-acbb-c6d477b33913"
session_id = uuid.uuid4()
texts = ["When was HCA founded?"]
language_code = "en-us"
detect_intent_texts(agent_id, session_id, texts, language_code, location_id)


# st.write("Session Path " + session_path)
# print(f"Session path: {session_path}\n")

# client_options = None
# json_file_path = '/Users/aaronwilkowitz/Documents/hcls_work/genai_app_hcls_general/keys/gbot-test-071-ac2f21641848.json'

# ClientOptions(
#     credentials_file=json_file_path
#   )






# print(f"Query text: {response.query_result.text}")
# response_messages = [
#     " ".join(msg.text.text) for msg in response.query_result.response_messages
# ]
# print(f"Response text: {' '.join(response_messages)}\n")
# st.write(response_messages)










#!/usr/bin/env python

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

# """DialogFlow API Detect Intent Python sample with text inputs.

# Examples:
#   python detect_intent_texts.py -h
#   python detect_intent_texts.py --agent AGENT \
#   --session-id SESSION_ID \
#   "hello" "book a meeting room" "Mountain View"
#   python detect_intent_texts.py --agent AGENT \
#   --session-id SESSION_ID \
#   "tomorrow" "10 AM" "2 hours" "10 people" "A" "yes"
# """

# # [START dialogflow_cx_detect_intent_text]
# def run_sample():
#     # TODO(developer): Replace these values when running the function
#     project_id = "gbot-test-071"
#     # For more information about regionalization see https://cloud.google.com/dialogflow/cx/docs/how/region
#     location_id = "global"
#     # For more info on agents see https://cloud.google.com/dialogflow/cx/docs/concept/agent
#     agent_id = "a4ed271a-64b3-4da5-8181-30534d468273"
#     agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"
#     # For more information on sessions see https://cloud.google.com/dialogflow/cx/docs/concept/session
#     session_id = uuid.uuid4()
#     texts = ["Hello"]
#     # For more supported languages see https://cloud.google.com/dialogflow/es/docs/reference/language
#     language_code = "en-us"

#     detect_intent_texts(agent, session_id, texts, language_code)


# def detect_intent_texts(agent, session_id, texts, language_code):
#     """Returns the result of detect intent with texts as inputs.

#     Using the same `session_id` between requests allows continuation
#     of the conversation."""
#     session_path = f"{agent}/sessions/{session_id}"
#     print(f"Session path: {session_path}\n")
#     client_options = None
#     agent_components = AgentsClient.parse_agent_path(agent)
#     location_id = agent_components["location"]
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


# # [END dialogflow_cx_detect_intent_text]

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
#     )
#     parser.add_argument(
#         "--agent", help="Agent resource name.  Required.", required=True
#     )
#     parser.add_argument(
#         "--session-id",
#         help="Identifier of the DetectIntent session. " "Defaults to a random UUID.",
#         default=str(uuid.uuid4()),
#     )
#     parser.add_argument(
#         "--language-code",
#         help='Language code of the query. Defaults to "en-US".',
#         default="en-US",
#     )
#     parser.add_argument("texts", nargs="+", type=str, help="Text inputs.")

#     args = parser.parse_args()

#     detect_intent_texts(args.agent, args.session_id, args.texts, args.language_code)

# project_id = "gbot-test-071"
# location_id = "global"
# agent_id = "a4ed271a-64b3-4da5-8181-30534d468273"
# var_agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"
# var_session_id = uuid.uuid4()
# var_texts = ["When was HCA founded?"]
# var_language_code = 'en-us'
# detect_intent_texts(var_agent, var_session_id, var_texts, var_language_code)













# ################
# ### model inputs
# ################

# # Model Inputs
# st.divider()
# st.header('1. Model Inputs')

# model_id = st.selectbox(
#     'Which model do you want to use?'
#     , (
#           'code-bison@001'
#         , 'code-bison@latest'
#       )
#     , index = 0
#   )
# model_temperature = st.number_input(
#       'Model Temperature'
#     , min_value = 0.0
#     , max_value = 1.0
#     , value = 0.2
#   )
# model_token_limit = st.number_input(
#       'Model Token Limit'
#     , min_value = 1
#     , max_value = 1024
#     , value = 200
#   )
# model_top_k = st.number_input(
#       'Top-K'
#     , min_value = 1
#     , max_value = 40
#     , value = 40
#   )
# model_top_p = st.number_input(
#       'Top-P'
#     , min_value = 0.0
#     , max_value = 1.0
#     , value = 0.8
#   )

# ################
# ### Query information 
# ################

# # Query Information
# st.divider()
# st.header('2. Query information')

# # List datasets 
# # project_id = "cloudadopt-public-data"
# project_id = "cloudadopt"
# project_id_datasets = "cloudadopt-public-data"
# location_id = "us-central1"

# client = bigquery.Client(project=project_id_datasets)
# datasets_all = list(client.list_datasets())  # Make an API request.
# dataset_string = ''
# for dataset in datasets_all: 
#     dataset_id2 = dataset.dataset_id
#     dataset_id2 = '\'' + dataset_id2 + '\''
#     dataset_string = dataset_string + dataset_id2 + ','
# dataset_string = dataset_string[:-1]

# string1 = 'dataset_id = st.selectbox('
# string2 = '\'What dataset do you want to query?\''
# string3 = f', ({dataset_string}), index = 1)'
# string_full = string1 + string2 + string3
# exec(string_full)

# # List tables
# table_names = [table.table_id for table in client.list_tables(f"{project_id_datasets}.{dataset_id}")]
# table_names_str = '\n'.join(table_names)
# st.write(':blue[**Tables:**] ')
# st.text(table_names_str)

# table_uri = f"bigquery://{project_id_datasets}/{dataset_id}"
# engine = create_engine(f"bigquery://{project_id_datasets}/{dataset_id}")

# # Vertex 
# vertexai.init(
#     project=project_id
#   , location=location_id
# )

# # LLM model
# model_name = "text-bison@001" #@param {type: "string"}
# max_output_tokens = 1024 #@param {type: "integer"}
# temperature = 0.2 #@param {type: "number"}
# top_p = 0.8 #@param {type: "number"}
# top_k = 40 #@param {type: "number"}
# verbose = True #@param {type: "boolean"}

# llm = VertexAI(
#   model_name=model_name,
#   max_output_tokens=max_output_tokens,
#   temperature=temperature,
#   top_p=top_p,
#   top_k=top_k,
#   verbose=verbose
# )

# ################
# ### Provide SQL Rules
# ################

# # Provide SQL Rules
# st.divider()
# st.header('3. Provide SQL Rules')

# include_sql_rules = st.selectbox(
#     'Do you want to include general SQL rules?'
#     , (
#           'yes'
#         , 'no'
#       )
#     , index = 0
#   )

# include_dataset_rules = st.selectbox(
#     'Do you want to include SQL rules on this particular dataset?'
#     , (
#           'yes'
#         , 'no'
#       )
#     , index = 0
#   )

# sql_quesiton = st.text_input(
#    'What is your question?'
#    , value = "How many flights were delayed in California in 2002?"
#   )

# custom_rules = st.text_input(
#    'If you need to include any additional SQL rules, provide them here'
#    , value = "Define a delay as 30 minutes or greater; use 2-letter abbreviations for states"
#   )

# if include_sql_rules == 'yes':
#    text_sql_rules = """
# You are a GoogleSQL expert. Given an input question, first create a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer to the input question.
# Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
# Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
# Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
# Use the following format:
# Question: "Question here"
# SQLQuery: "SQL Query to run"
# SQLResult: "Result of the SQLQuery"
# Answer: "Final answer here"
# Only use the following tables:
# {table_info}

# Rule 1:
# Do not filter and query for columns that do not exist in the table being queried.

# Rule 2:
# If someone asks for the column names in all the tables, use the following format:
# SELECT column_name
# FROM `{project_id}.{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
# WHERE table_name in {table_info}

# Rule 3:
# If someone asks for the column names in a particular table (let's call that provided_table_name for sake of example), use the following format:
# SELECT column_name
# FROM `{project_id}.{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
# WHERE table_name = "provided_table_name"

# # Rule 4:
# # Double check that columns used in query are correctly named and exist in the corresponding table being queried. Use actual names that exist in table, not synonyms.

# # Rule 5:
# # If someone mentions a specific table name (e.g. flights data, carriers data), assume the person is referring to the table name(s) that most closely correspond to the name.

# # Rule 6:
# # Follow each rule every time.

# """
# else:
#    text_sql_rules = ""

# if include_dataset_rules == 'yes' and 'faa' in dataset_id: 
#    text_sql_rules_specific = """
# Rule:
# When filtering on specific dates, convert the date(s) to DATE format before executing the query.

# Example:

#   User Query: "How many flights were there between January 10th and 17th in 2022?"
#   Desired Query Output:
#   SELECT count(*)
#   FROM `{project_id}.{dataset_id}.flights` 
#   AND partition_date BETWEEN DATE('2022-01-10') AND DATE('2022-01-17')

# Rule:
# Flights and airports join on flights.origin = airports.code

# Example: 

#   User Query: "How many flights were there in Florida between January 10th and 17th in 2022?"
#   Desired Query Output:
#   SELECT count(*)
#   FROM `{project_id}.{dataset_id}.flights` AS flights 
#   JOIN `{project_id}.{dataset_id}.airports` AS airports
#     ON flights.origin = airports.code
#   WHERE airports.state = 'Florida'
#   AND flights.partition_date BETWEEN DATE('2022-01-10') AND DATE('2022-01-17')

# Rule:
# Flights and carriers join on flights.carrier = carriers.code

# Example: 

#   User Query: "Which carriers had the highest percent of flights delayed by more than 15 minutes?"
#   Desired Query Output:
#   SELECT carriers.name, sum(case when dep_delay > 15 then 1 else 0 end) / count(*) as percent_delayed_flights
#   FROM `{project_id}.{dataset_id}.flights` AS flights 
#   JOIN `{project_id}.{dataset_id}.carriers` AS carriers
#     ON flights.carrier = carriers.code
#   GROUP BY 1 
#   ORDER BY 2 desc 
#   LIMIT 10

# Rule:
# If you are doing a join and referring to a table by short hand name, don't forget to name the table being joined with an "AS"

# Example:

#   SELECT carriers.name, flights.origin, flights.destination, count(*)
#   FROM `{project_id}.{dataset_id}.flights` AS flights 
#   JOIN `{project_id}.{dataset_id}.carriers` AS carriers
#     ON flights.carrier = carriers.code
#   GROUP BY 1,2,3
#   ORDER BY 4 desc 
#   LIMIT 200

# """
# else: 
#     text_sql_rules_specific = ""

# if custom_rules == "":
#    text_custom_rules = custom_rules 
# else: 
#    text_custom_rules = f"""

# Additional Context: {custom_rules}

# """

# text_final = "Question: {input}"

# sql_prompt = text_sql_rules + text_sql_rules_specific + text_custom_rules + text_final



# ################
# ### SQL Answer
# ################

# # Provide SQL Rules
# st.divider()
# st.header('4. SQL Answer')

# def bq_qna(question):
#   #create SQLDatabase instance from BQ engine
#   db = SQLDatabase(
#       engine=engine
#      ,metadata=MetaData(bind=engine)
#      ,include_tables=table_names # [x for x in table_names]
#   )

#   #create SQL DB Chain with the initialized LLM and above SQLDB instance
#   db_chain = SQLDatabaseChain.from_llm(
#       llm
#      , db
#      , verbose=True
#      , return_intermediate_steps=True)

#   #Define prompt for BigQuery SQL
#   _googlesql_prompt = sql_prompt

#   GOOGLESQL_PROMPT = PromptTemplate(
#       input_variables=[
#          "input"
#          , "table_info"
#          , "top_k"
#          , "project_id"
#          , "dataset_id"
#       ],
#       template=_googlesql_prompt,
#   )

#   #passing question to the prompt template
#   final_prompt = GOOGLESQL_PROMPT.format(
#        input=question
#      , project_id =project_id_datasets
#      , dataset_id=dataset_id
#      , table_info=table_names
#      , top_k=10000
#     )

#   # pass final prompt to SQL Chain
#   output = db_chain(final_prompt)

#   # outputs
#   st.write(':blue[**SQL:**] ')
#   sql_answer = output['intermediate_steps'][1]
#   st.code(sql_answer, language="sql", line_numbers=False)

#   st.write(':green[**Answer:**] ')
#   st.write(output['result'])


#   st.write(':blue[**Full Work:**] ')
#   st.write(output)  

# bq_qna(sql_quesiton)