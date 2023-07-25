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
from google.cloud import bigquery
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import CodeGenerationModel

# others
# from langchain import SQLDatabase, SQLDatabaseChain
# from langchain.prompts.prompt import PromptTemplate
# from langchain import LLM
from langchain.llms import VertexAI
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

import streamlit as st
import pandas as pd
import db_dtypes 
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
st.title('GCP HCLS GenAI Demo: Write a BigQuery SQL Query')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-07-24')
st.write('**Purpose**: You need to ask a question of your data. Have GenAI generate a query in BigQuery')

# ################
# ### model inputs
# ################

project_id = "cloudadopt"
location_id = "us-central1"
dataset_id = 'faa' 

client = bigquery.Client()
table_names = [table.table_id for table in client.list_tables(f"{project_id}.{dataset_id}")]
table_names_print = *table_names, sep="\n"
# st.write(table_names_print)
# st.write(*table_names, sep="\n")
st.write('check')

import sys
path = sys.path
st.write(path)

vertexai.init(project=project_id, location=location_id)

# LLM model
model_name = "text-bison@001" #@param {type: "string"}
max_output_tokens = 1024 #@param {type: "integer"}
temperature = 0.2 #@param {type: "number"}
top_p = 0.8 #@param {type: "number"}
top_k = 40 #@param {type: "number"}
verbose = True #@param {type: "boolean"}

llm = VertexAI(
  model_name=model_name,
  max_output_tokens=max_output_tokens,
  temperature=temperature,
  top_p=top_p,
  top_k=top_k,
  verbose=verbose
)

table_uri = f"bigquery://{project_id}/{dataset_id}"
engine = create_engine(f"bigquery://{project_id}/{dataset_id}")
st.write(table_uri)

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
# project_id = "cloudadopt-public-data"
# client = bigquery.Client(project=project_id)
# datasets_all = list(client.list_datasets())  # Make an API request.
# dataset_string = ''
# for dataset in datasets_all: 
#     dataset_id = dataset.dataset_id
#     dataset_id = '\'' + dataset_id + '\''
#     dataset_string = dataset_string + dataset_id + ','
# dataset_string = dataset_string[:-1]

# string1 = 'dataset = st.selectbox('
# string2 = '\'What dataset do you want to query?\''
# string3 = f', ({dataset_string}), index = 2)'
# string_full = string1 + string2 + string3
# exec(string_full)

# # List table information
# # Test query
# columns_query = f"""
# SELECT 
#     '\`' || table_catalog || '.' || table_schema || '.' || table_name || '\`' as full_table_name
#   , table_schema || '.' || table_name as short_table_name  
#   , table_name || '.' || column_name as full_column_name 
#   , column_name 
#   , data_type 
# FROM `{project_id}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
# ORDER BY table_name, ordinal_position 
# """
# df = client.query(columns_query).to_dataframe()

# columns_query_simple = f"""
# SELECT 
#     table_name || '.' || column_name as column_name 
# FROM `{project_id}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
# ORDER BY table_name, ordinal_position 
# """
# df_simple = client.query(columns_query_simple).to_dataframe()

# include_common_values = st.selectbox(
#     'Do you want to feed common values of each column into the LLM prompt?'
#     , (
#           'no - simple metadata'
#         , 'no - complex metadata'
#         , 'yes'
#       )
#     , index = 0
#   )

# st.write(':blue[**Schema:**] ')

# if include_common_values == 'no - simple metadata':
#   df_final = df_simple
#   # data_table = st.table(df_simple)
# elif include_common_values == 'no - complex metadata':
#   df_final = df
#   # data_table = st.table(df)
# else: 
#   # Create dataframe & start loop
#   # note: must do this step b/c data_tables function doesn't work on streamlit when on cloud run, not sure why
#   fake_data = {
#         'full_table_name' : '.'
#       , 'short_table_name' : '.'
#       , 'full_column_name' : '.'
#       , 'column_name' : '.'
#       , 'data_type' : '.'
#       , 'common_values' : '.'
#   }
#   df_fake_schema = pd.DataFrame(fake_data, index=[0])
#   df_final = df_fake_schema
#   data_table = st.table(df_fake_schema)
#   for ind in df.index:
#     full_table_name = df['full_table_name'][ind]
#     short_table_name = df['short_table_name'][ind]
#     column_name = df['column_name'][ind]
#     data_type = df['data_type'][ind]
#     common_values_query = f"""
#       SELECT 
#           {column_name} as column_values
#         , count(*) as count
#       FROM {full_table_name}
#       GROUP BY 1 
#       ORDER BY 2 desc
#       LIMIT 5 
#     """
#     df_common_values = client.query(common_values_query).to_dataframe()
#     common_values_list = df_common_values['column_values'].tolist()
#     common_values_string = ','.join(str(x) for x in common_values_list)

#     # Create dataframe
#     df_new_dict = {
#         'full_table_name' : full_table_name
#       , 'short_table_name' : short_table_name
#       , 'column_name' : column_name
#       , 'data_type' : data_type
#       , 'common_values' : common_values_string
#     }
#     # Append to df using dict
#     df_row = pd.DataFrame(df_new_dict, index=[0])
#     # data_table.add_rows(df_row)
#     df_final = pd.concat([df_final,df_row])
  
# data_table = st.table(df_final)

# ################
# ### Ask a Question
# ################

# # Query Information
# st.divider()
# st.header('3. Ask a Question')

# custom_prompt = st.text_input(
#    'Write your question here'
#    , value = "How many rows are in the dataset?"
#   )

# ################
# ### Produce SQL
# ################

# # Query Information
# st.divider()
# st.header('4. LLM Produce BigQuery SQL')

# llm_prompt = f"""
# Context: You are an expert at writing BigQuery SQL statements. Write a SQL statement that answers the question below. Below the question are lists of tips and database information. 

# ***Question***: How many flights were there in Florida?

# *** List of Tips START *** 
# - Only use table names & column names included in the database metadata 
# - Remember to join tables, if required
# *** List of Tips END *** 

# *** Database Metadata START *** 
# {df_final}
# *** Database Metadata END ***
# """

# st.write(':blue[**Prompt:**] ' )
# llm_prompt_display = st.text(llm_prompt)

# project_id = "cloudadopt"
# location_id = "us-central1"

# # Run the first model
# vertexai.init(
#       project = project_id
#     , location = location_id)
# parameters = {
#     "temperature": model_temperature,
#     "max_output_tokens": model_token_limit
# }
# model = CodeGenerationModel.from_pretrained(model_id)
# response = model.predict(
#     f'''{llm_prompt}''',
#     **parameters
# )
# # print(f"Response from Model: {response.text}")
# llm_response = response.text
# st.write(':blue[**LLM Response:**] ')
# st.text(llm_response)









# # # Bard Test
# # st.divider()
# # st.header('Bard Test')

# # from bardapi import Bard
# # bard = Bard(token="YwhpOVTxde5U6CfJ02ISTnC2t9HTa4hV5gqgbHIeN0K7XpTlTsuy0Pae9gKN-cqYLhtXMw.")
# # bard_answer = bard.get_answer('How are you?')
# # st.write(bard_answer)
# # url = bard.export_conversation(bard_answer, title='Example Shared conversation')
# # st.write(url)
# # st.write("check50")
# # from bardapi import Bard
# # import os
# # import requests
# # token='YwhpOVTxde5U6CfJ02ISTnC2t9HTa4hV5gqgbHIeN0K7XpTlTsuy0Pae9gKN-cqYLhtXMw.'
# # os.environ['_BARD_API_KEY'] = token
# # st.write("check60")
# # session = requests.Session()
# # session.headers = {
# #             "Host": "bard.google.com",
# #             "X-Same-Domain": "1",
# #             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
# #             "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
# #             "Origin": "https://bard.google.com",
# #             "Referer": "https://bard.google.com/",
# #         }
# # session.cookies.set("__Secure-1PSID", os.getenv("_BARD_API_KEY")) 
# # # session.cookies.set("__Secure-1PSID", token) 
# # st.write("check70")
# # bard = Bard(token=token, session=session, timeout=30)
# # bard.get_answer("What should I have for lunch?")['content']
# # st.write("check80")
# # # Continued conversation without set new session
# # bard.get_answer("What is my last prompt??")['content']
# # st.write("check90")
# # 

# # st.write("check10")

# # from bardapi import Bard
# # import os
# # token = "YwhpOVTxde5U6CfJ02ISTnC2t9HTa4hV5gqgbHIeN0K7XpTlTsuy0Pae9gKN-cqYLhtXMw."
# # # YwhpOVTxde5U6CfJ02ISTnC2t9HTa4hV5gqgbHIeN0K7XpTlTsuy0Pae9gKN-cqYLhtXMw.
# # # 'ZAhGy_kfqC1Sm-vZ8WwzDkip2s85YWoTnZKoVmuCn0oKjkEG2vSXpe9JVA4Hb1-UYsbnwg.'
# # os.environ['_BARD_API_KEY']=token
# # # bard = Bard(token=token)

# # st.write("check25")

# # custom_prompt = st.text_input(
# #    'Write your question here'
# #    , value = "How many rows are in the dataset?"
# #   )

# # st.write("check40")

# # # bard = Bard(token_from_browser=True)
# # # bard = Bard()

# # st.write("check50")

# # # answer = Bard.get_answer(custom_prompt)['content']
# # answer = Bard.get_answer("What should I have for lunch?")# ['content']


# # st.write("check60")

# # st.write("answer: " + answer)










# # if datasets:
# #     print("Datasets in project {}:".format(project))
# #     for dataset in datasets:
# #         print("\t{}".format(dataset.dataset_id))
# # else:
# #     print("{} project does not contain any datasets.".format(project))

# # # Test query
# # test_query =  """
# # SELECT
# #   CONCAT(
# #     'https://stackoverflow.com/questions/',
# #     CAST(id as STRING)) as url,
# #   view_count
# # FROM `bigquery-public-data.stackoverflow.posts_questions`
# # WHERE tags like '%google-bigquery%'
# # ORDER BY view_count DESC
# # LIMIT 10
# # """
# # client = bigquery.Client()

# # df = client.query(test_query).to_dataframe()
# # data_table = st.table(df)





# # client = bigquery.Client()
# # query_job = client.query(test_query)
# # results = query_job.result()  # Waits for job to complete.
# # # st.write('results: ' + results)
# # # st.write('url: ' + results[0].url)
# # # st.write('count: ' + results[0].count)

# # for row in results:
# #     st.write("{} : {} views".format(row.url, row.view_count))

# # # Body Part
# # param_body_part = st.selectbox(
# #     'What body part does the image refer to?'
# #     , (
# #           'chest'
# #         , 'head'
# #         , 'custom'
# #       )
# #     , index = 0
# #   )

# # custom_body_part = st.text_input('If you select custom, write a custom body part here')

# # if "custom" in param_body_part:
# #   body_part = custom_body_part
# # else:
# #   body_part = param_body_part

# # # Modality
# # param_modality = st.selectbox(
# #     'What modality does the image refer to?'
# #     , (
# #           'x-ray'
# #         , 'ct-scan'
# #         , 'mri'
# #         , 'custom'
# #       )
# #     , index = 0
# #   )

# # custom_modality = st.text_input('If you select custom, write a custom modality here')

# # if "custom" in param_modality:
# #   modality = custom_modality
# # else:
# #   modality = param_modality

# # body_part_modality = body_part + ' ' + modality

# # st.write(':blue[**Imaging Type:**] ' + body_part_modality)

# # input_prompt_a_1_context = '''You are an expert in medical imaging & in data science. We want to transform a radiologist\'s imaging summary into a series of binary ("yes" or "no") labels to use for training in a future data science predictive model. Can you please provide a comma separated list of the labels we should check for the below?

# # '''

# # input_prompt_a_2_examples = '''
# # input: chest x-ray
# # output: atelectasis, cardiomegaly, consolidation, effusion, hilar_enlargement, infiltrate, nodule, pneumonia, pleural_thickening, pneumothorax

# # input: head ct-scan
# # output: calcification, hemorrhage, hydrocephalus, mass, midline_shift, skull_fracture, subarachnoid_hemorrhage, subdural_hematoma

# # '''

# # input_prompt_a_3_input = 'input: ' + body_part_modality

# # input_prompt_a_4_output = '''
# # output:
# # '''

# # st.write(':blue[**Prompt 1:**] ')
# # llm_prompt_a_display = st.text(input_prompt_a_1_context + input_prompt_a_2_examples + input_prompt_a_3_input + input_prompt_a_4_output)
# # llm_prompt_a = input_prompt_a_1_context + input_prompt_a_2_examples + input_prompt_a_3_input + input_prompt_a_4_output

# # project_id = "cloudadopt"
# # location_id = "us-central1"

# # # Run the first model
# # vertexai.init(
# #       project = project_id
# #     , location = location_id)
# # parameters = {
# #     "temperature": model_temperature,
# #     "max_output_tokens": model_token_limit,
# #     "top_p": model_top_p,
# #     "top_k": model_top_k
# # }
# # model = TextGenerationModel.from_pretrained(model_id)
# # response = model.predict(
# #     f'''{llm_prompt_a}''',
# #     **parameters
# # )
# # # print(f"Response from Model: {response.text}")

# # llm_response_text_a = response.text

# # st.write(':blue[**List of Labels:**] ' + llm_response_text_a)

# # st.divider()

# # ################
# # ### LLM B
# # ################

# # # LLM Model 2 -- Create binary labels
# # st.header('3. LLM B - Create binary label based on Medical Text')

# # # File Name
# # file_id = st.selectbox(
# #     'What file do you want to use?'
# #     , (
# #           "Imaging Result 1"
# #         , "Imaging Result 2"
# #         , "Imaging Result 3"
# #       )
# #   )

# # # File Text
# # if file_id == "Imaging Result 1":
# #   file_text = '''[
# #  {
# #    "key": "MeSH",
# #    "value": "normal"
# #  },
# #  {
# #    "key": "Problems",
# #    "value": "normal"
# #  },
# #  {
# #    "key": "image",
# #    "value": "Xray Chest PA and Lateral"
# #  },
# #  {
# #    "key": "indication",
# #    "value": "Positive TB test"
# #  },
# #  {
# #    "key": "comparison",
# #    "value": "None."
# #  },
# #  {
# #    "key": "findings",
# #    "value": "The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax."
# #  },
# #  {
# #    "key": "impression",
# #    "value": "Normal chest x-XXXX."
# #  }
# # ]'''
# # elif file_id == "Imaging Result 2":
# #   file_text = '''[
# #  {
# #    "key": "MeSH",
# #    "value": "Opacity/lung/base/left/mild;Implanted Medical Device;Atherosclerosis/aorta;Calcinosis/lung/hilum/lymph nodes;Calcinosis/mediastinum/lymph nodes;Spine/degenerative/mild;Granulomatous Disease"
# #  },
# #  {
# #    "key": "Problems",
# #    "value": "Opacity;Implanted Medical Device;Atherosclerosis;Calcinosis;Calcinosis;Spine;Granulomatous Disease"
# #  },
# #  {
# #    "key": "image",
# #    "value": "Xray Chest PA and Lateral"
# #  },
# #  {
# #    "key": "indication",
# #    "value": "XXXX-year-old male with XXXX for 3 weeks. Possible pneumonia."
# #  },
# #  {
# #    "key": "comparison",
# #    "value": ""
# #  },
# #  {
# #    "key": "findings",
# #    "value": "There are minimal XXXX left basilar opacities, XXXX subsegmental atelectasis or scarring. There is no focal airspace consolidation to suggest pneumonia. No pleural effusion or pneumothorax. Heart size is at the upper limits of normal. Cardiac defibrillator XXXX overlies the right ventricle. The XXXX appears intact. There is aortic atherosclerotic vascular calcification. Calcified mediastinal and hilar lymph XXXX are consistent with prior granulomatous disease. Multiple calcified splenic granulomas are also noted. There are minimal degenerative changes of the spine."
# #  },
# #  {
# #    "key": "impression",
# #    "value": "Minimal left basilar subsegmental atelectasis or scarring. No acute findings."
# #  }
# # ]'''
# # elif file_id == "Imaging Result 3":
# #   file_text = '''[
# #  {
# #    "key": "MeSH",
# #    "value": "Pneumonia/upper lobe/left;Airspace Disease/lung/upper lobe/left"
# #  },
# #  {
# #    "key": "Problems",
# #    "value": "Pneumonia;Airspace Disease"
# #  },
# #  {
# #    "key": "image",
# #    "value": "Chest PA and lateral views. XXXX, XXXX XXXX PM"
# #  },
# #  {
# #    "key": "indication",
# #    "value": "XXXX with XXXX"
# #  },
# #  {
# #    "key": "comparison",
# #    "value": "none"
# #  },
# #  {
# #    "key": "findings",
# #    "value": "XXXX XXXX and lateral chest examination was obtained. The heart silhouette is normal in size and contour. Aortic XXXX appear unremarkable. Lungs demonstrate left upper lobe airspace disease most XXXX pneumonia. There is no effusion or pneumothorax."
# #  },
# #  {
# #    "key": "impression",
# #    "value": "1. Left upper lobe pneumonia."
# #  }
# # ]'''
# # else:
# #   file_text = "unknown"

# # st.write(':blue[**File ID:**] ' + file_id)
# # st.write(':blue[**File Text:**] ' + file_text)

# # # Convert comma separated list into list
# # list_of_labels_to_create = llm_response_text_a.split(',')

# # ################
# # ### LLM B Output
# # ################

# # # Create dataframe & start loop
# # # note: must do this step b/c data_tables function doesn't work on streamlit when on cloud run, not sure why
# # fake_data = {
# #       'label_name' : '.'
# #     , 'label_outcome' : '.'
# # }
# # df_fake_schema = pd.DataFrame(fake_data, index=[0])
# # data_table = st.table(df_fake_schema)

# # for indexA, valueA in enumerate(list_of_labels_to_create):
# #   input_prompt_b_1_context = '''You are an expert in medical imaging. Below is a yes or no question, followed by a radiologist's imaging summary.

# #   Can you please answer the question **WITH ONLY** "yes" or "no"? Thank you.

# #   '''

# #   input_prompt_b_2_question = 'Question: Does the below imaging summary show evidence of ' + list_of_labels_to_create[indexA] + '?'

# #   input_prompt_b_3_file_text = '''
# #   Imaging Summary:

# #   ''' + file_text

# #   llm_prompt_b = input_prompt_b_1_context + input_prompt_b_2_question + input_prompt_b_3_file_text

# #   # Run the first model
# #   vertexai.init(
# #         project = project_id
# #       , location = location_id)
# #   parameters = {
# #       "temperature": model_temperature,
# #       "max_output_tokens": model_token_limit,
# #       "top_p": model_top_p,
# #       "top_k": model_top_k
# #   }
# #   model = TextGenerationModel.from_pretrained(model_id)
# #   response = model.predict(
# #       f'''{llm_prompt_b}''',
# #       **parameters
# #   )
# #   # print(f"Response from Model: {response.text}")

# #   llm_response_text_b = response.text

# #   # Create dataframe
# #   testdict = {
# #       'label_name': list_of_labels_to_create[indexA]
# #     , 'label_outcome': llm_response_text_b
# #   }
# #   # Append to df using dict
# #   df_row = pd.DataFrame(testdict, index=[0])
# #   data_table.add_rows(df_row)


