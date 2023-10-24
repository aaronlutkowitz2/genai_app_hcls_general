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
from streamlit_player import st_player
import pandas as pd
import json 
from datetime import datetime, timedelta

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('Generate Patient Summary')

# Author & Date
st.write('**Author**: Aaron Wilkowitz')
st.write('**Date**: 2023-10-24')
st.write('**Purpose**: Generate Patient Summary')

# Gitlink
# st.write('**Go Link (Googlers)**: go/hclsgenai')
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('60 Second Video')

video_url = 'https://youtu.be/JMyGfydQGzk'
st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vTb_Qe8AqxFZ9so49raQrT0xNFiWqs66FhPrZ9uT-tFM0xBjoPePZfEY54DZGjRF0MJpKhwp0s-sIWk/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### Notes on Code
################

#### Notes on modularizing the code

# Section 1: Pull down the context files + parse them into the format required
    # See document with list of blue rules on what format is expected

# Section 2: Generate 1-line variables of each element (i.e. everything but labs, meds, vitals)
    # LLM prompt 1: generate answer & evidence for the element
    # LLM prompt 2: run hallucination check
    # LLM prompt 3: if some hallucination checks succeed, others fail, create a new answer + evidence
    # LLM prompt 4: if some hallucination checks succeed, others fail, run a relevance check

# Section 3: Generate labs, meds, vitals lists – abnormal only & full list
    # LLM prompt: generate low/normal/high on vitals

# Section 4: Generate a summary
    # LLM prompt: generate a 4-5 sentence summary

# Section 5: Generate a json sturcture with data from sections 2,3,4 above

# Section 6: Generate final report UX

# Section 7: Highlight relevant sections of code

################
### General
################

# Define parameters
project_id = "cloudadopt"
location_id = "us-central1"

# Define model
model_id = 'text-bison' 
model_temperature = 0
model_token_limit = 200
model_top_k = 40 
model_top_p = 0.8

################
### # Section 1: Process Context Files
################

@st.cache_data
def section1_pull_context_files():
    ####### Process Context File ########

    st.divider()
    st.header('1. Process Context Files')
    st.write('''Pull down the context files + parse them into the format required
    # See document with list of blue rules on what format is expected''')

    text = 'Process Files: :orange[**start**]'
    st.write(text)

    # Storage
    storage_client = storage.Client()
    bucket_name = 'hcls_genai'
    path = 'patient_summary/'
    file_name = "fake_patient_context v4.txt"
    full_file_name = path + file_name
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(full_file_name)
    file_content = blob.download_as_text()
    # st.write(file_content)

    # Break into sections
        ## ALERT !! ASW Assumption: all sections are broken out by the string "Section:"
    section_list = file_content.split('Section:')

    # Create a dict with section name & full string
        ## ALERT !! ASW Assumption: all sections titles end with a newline "\n" string 
    section_list2 = [{"section_name": s[:s.find('\\n')].strip(), "full_string": s} for s in section_list]
    # st.write(section_list2)

    ####### Process File Excluding Vitals Meds Labs ########

    # Create a dict that excludes vitals, meds, labs
        ## ALERT !! ASW Assumption: "Vitals", "Labs", "Medications" are the exact names of those sections
    section_list_no_vitals_meds_labs = [s for s in section_list2 if s.get("section_name") not in ["Vitals", "Labs", "Medications"]]
    # st.write(section_list_no_vitals_meds_labs)

    # Create a string that joins all of the values remaining
    section_list_no_vitals_meds_labs_1_string_values = [s["full_string"] for s in section_list_no_vitals_meds_labs if s.get("section_name") not in ["Vitals", "Labs", "Medications"]]
    section_list_no_vitals_meds_labs_1_string = "\n\n ".join(section_list_no_vitals_meds_labs_1_string_values)
    # st.write(section_list_no_vitals_meds_labs_1_string)

    ####### Process Instructions File ########

    file_name = "fake_patient_instructions v2.csv"
    full_file_name = "gs://" + bucket_name + '/' + path + file_name

    df = pd.read_csv(
        full_file_name
    ) 

    text = 'Process Files: :green[**done**]'
    st.write(text)

    return df, section_list_no_vitals_meds_labs_1_string_values, section_list_no_vitals_meds_labs_1_string, section_list2

# Run the function & pull out the variables I need
result = section1_pull_context_files()
df, section_list_no_vitals_meds_labs_1_string_values, section_list_no_vitals_meds_labs_1_string, section_list2 = result

################
### # Section 2: Generate elements 
################

@st.cache_data
def section2_generate_elements():

    st.divider()
    st.header('2. Generate elements')

    st.write('''
    Generate 1-line variables of each element (i.e. everything but labs, meds, vitals) \n
    LLM prompt 1: generate answer & evidence for the element \n
    LLM prompt 2: run hallucination check \n
    LLM prompt 3: if some hallucination checks succeed, others fail, create a new answer + evidence \n
    LLM prompt 4: if some hallucination checks succeed, others fail, run a relevance check \n
    ''')

    text = 'Metadata: :orange[**start**]'
    st.write(text)

    # Run a loop through every line of the instructions and ask LLM for (a) answer, (b) evidence

    # Create the initial DataFrame
    strings = ['.', '.', '.', '.', '.']
    columns=['column_name', 'column_description', 'answer', 'evidence', 'text_match']
    df_schema = pd.DataFrame([strings], columns=['column_name', 'column_description', 'answer', 'evidence', 'text_match'], index=[0])

    # data_table = st.table(df_schema)

    # while testing, use this so it only calcs 1, not all of them
    # df = df[df['column_name'].str.strip() == 'orientation']
    # df = df.head()

    for index, row in df.iterrows():

        column_name = row['column_name']
        column_description = row['column_description']
        
        # Step A-I: Generate the prompt w/ answer & evidence
        json_schema = '''
    {
        "answer": [Answer the question based on the medical history. If the medical history does not include the answer, leave this blank ""]
        "evidence": [provide the exact text from the medical history that answers the question; if there are multiple pieces of evidence, include all of them and separate them with a pipe "|" character. If the medical history does not include the answer, leave this blank ""]
    }
    '''

        prompt_text = f'''

    *** Context: *** 
    Below I have a question. Below that I have a json schema with a request for information in [brackets]. Below that I have a medical history of a patient.

    *** Request: *** 
    Please fill in the json schema that answers the question

    *** Question: *** 
    {column_description}

    *** Rules: ***
    Rule 1: Output your response with json only. Do not include any text besides the json format below
    Rule 2: Make sure your json is correctly formatted
    Rule 3: Make sure the column names of your json are the same as the json format below, including capitalization
    Rule 4: Make sure there is a comma between each element in the json response
    Rule 5: Only include information explicitly stated in the medical record
    Rule 6: If any of the columns do not have data, include the column in your json response and return with blanks "" for that field's value

    *** JSON Schema: ***
    {json_schema}

    *** Medical History: ***
    {section_list_no_vitals_meds_labs_1_string}
    '''

        model_token_limit = 200
        # Run the LLM
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
            f'''{prompt_text}''',
            **parameters
        )
        llm_response = response.text
        # st.write(llm_response)

        # st.write(prompt_text)

        json_object = json.loads(llm_response)

        answer = str(json_object["answer"]).replace("[", "").replace("]", "").replace("'", "")
        evidence = str(json_object["evidence"]).replace("[", "").replace("]", "").replace("'", "")
        
        # Step A-II: Run a second LLM to check for hallucination -- break it out by every pipe-delimited quote
        evidence_list = evidence.split("|")
        issues = 0
        no_issues = 0
        good_evidence = ''
        for element in evidence_list:
            element_strip = element.strip()
            
            prompt_text = f'''

    *** Context: *** 
    Below is a statement. Below that is a passage.

    *** Request: *** 
    Please determine whether the statement is included in the passage below. Respond with only 3 possible values: 
    - "exact_match" 
    - "distant_match"
    - "not_in_passage"

    *** Statement: *** 
    {element_strip}

    *** Passage: *** 
    {section_list_no_vitals_meds_labs_1_string}
            '''

            model_token_limit = 5
            # Run the LLM
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
                f'''{prompt_text}''',
                **parameters
            )
            llm_response = response.text
            text_match = llm_response.strip()

            if text_match == 'not_in_passage':
                text_hallucination = ':red[**' + text_match.strip() + '**] -- *action: ignore this response*'
                issues = issues + 1 
            elif text_match == 'distant_match':
                text_hallucination = ':orange[**' + text_match.strip() + '**] -- *action: ignore this response*'
                issues = issues + 1 
            else: 
                text_hallucination = ':green[**' + text_match.strip() + '**]'
                if good_evidence == "":
                    good_evidence = element_strip
                else: 
                    good_evidence = good_evidence + ' | ' + element_strip
                no_issues = no_issues + 1 
            text = ':blue[**' + column_name.strip() + '**] : ' + answer + '; :blue[**evidence**] : ' + element_strip + '; :blue[**hallucination check**] : ' + text_hallucination
            st.write(text)

        # Step B-I: If every answer was exact match, just go with the answer as intended
        if issues == 0 and no_issues > 0:
            new_row_data = [column_name, column_description, answer, evidence, text_match]
            df_new_row = pd.DataFrame([new_row_data], columns=['column_name', 'column_description', 'answer', 'evidence', 'text_match'], index=[0])
            df_schema = pd.concat([df_schema, df_new_row], ignore_index=True)
        # Step B-II: If no answer was an exact match, ignore it and move on
        elif issues > 0 and no_issues == 0:
            continue 
        # Step B-III: If some were not exact matches, some were exact matches, concatenate all the exact matches & summarize into a single new answer
        else: 

            ## Step B-III-1: summarize a new answer

            prompt_text = f'''Summarize this passage: 
            {good_evidence}
            '''
            model_token_limit = 100
            # Run the LLM
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
                f'''{prompt_text}''',
                **parameters
            )
            llm_response = response.text
            answer = llm_response
            evidence = good_evidence
            text_match = 'exact_match'
            
            ## Step B-III-2: double check for relevance
            
            prompt_text = f'''

    *** Context: *** 
    Below is a statement. Below that is a passage.

    *** Request: *** 
    Please determine whether the statement is relevant to the passage below. Respond with only 3 possible values: 
    - "relevant" 
    - "distantly_relevant"
    - "not_relevant"

    *** Statement: *** 
    {answer}

    *** Passage: *** 
    {column_description}
            '''
            model_token_limit = 5
            # Run the LLM
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
                f'''{prompt_text}''',
                **parameters
            )
            llm_response = response.text
            relevance = llm_response

            if relevance != "relevant":
                text_not_relevant = ':orange[**' + relevance.strip() + '**] -- *action: ignore this response*'
                text = '*FIXED* -- :blue[**' + column_name.strip() + '**] : ' + answer + '; :blue[**evidence**] : ' + good_evidence + '; :blue[**relevance check**] : ' + text_not_relevant
                st.write(text)
            else: 
                text_not_relevant = ':green[**' + relevance.strip() + '**]'
                text = '*FIXED* -- :blue[**' + column_name.strip() + '**] : ' + answer + '; :blue[**evidence**] : ' + good_evidence + '; :blue[**relevance check**] : ' + text_not_relevant
                st.write(text)

                new_row_data = [column_name, column_description, answer, evidence, text_match]
                df_new_row = pd.DataFrame([new_row_data], columns=['column_name', 'column_description', 'answer', 'evidence', 'text_match'], index=[0])
                df_schema = pd.concat([df_schema, df_new_row], ignore_index=True)

    # Create variables for each of these
    try:
        room = df_schema[df_schema['column_name'].str.strip() == "room"]['answer'].iloc[0]
        room_text = f"- **Room**: {room} \n"
    except:
        room_text = ''
    try:
        patient_name = df_schema[df_schema['column_name'].str.strip() == "patient_name"]['answer'].iloc[0]
        patient_name_text = f"- **Patient Name**: {patient_name} \n"
    except:
        patient_name_text = ''
    try:
        code_status  = df_schema[df_schema['column_name'].str.strip() == "code_status"]['answer'].iloc[0]
        code_status_text = f"- **Code Status**: {code_status} \n"
    except:
        code_status_text = ''
    try:
        allergies = df_schema[df_schema['column_name'].str.strip() == "allergies"]['answer'].iloc[0]
        allergies_text = f"- **Allergies**: {allergies} \n"
    except:
        allergies_text = ''
    try:
        primary_doctor = df_schema[df_schema['column_name'].str.strip() == "primary_doctor"]['answer'].iloc[0]
        primary_doctor_text = f"- **Primary Doctor**: {primary_doctor} \n"
    except:
        primary_doctor_text = ''
    try:
        isolation_precautions  = df_schema[df_schema['column_name'].str.strip() == "isolation_precautions"]['answer'].iloc[0]
        isolation_precautions_text = f"- **Iso Prec**: {isolation_precautions} \n"
    except:
        isolation_precautions_text = ''
    try:
        pain_status = df_schema[df_schema['column_name'].str.strip() == "pain_status"]['answer'].iloc[0]
        pain_status_text = f"- **Pain Status**: {pain_status} \n"
    except:
        pain_status_text = ''
    try:
        imaging_delays = df_schema[df_schema['column_name'].str.strip() == "imaging_delays"]['answer'].iloc[0]
        imaging_delays_text = f"- **Imaging Delays**: {imaging_delays} \n"
    except:
        imaging_delays_text = ''
    try:
        chief_complaint = df_schema[df_schema['column_name'].str.strip() == "chief_complaint"]['answer'].iloc[0]
        chief_complaint_text = f"- **Chief Complaint**: {chief_complaint} \n"
    except:
        chief_complaint_text = ''
    try:
        diet = df_schema[df_schema['column_name'].str.strip() == "diet"]['answer'].iloc[0]
        diet_text = f"- **Diet**: {diet} \n"
    except:
        diet_text = ''
    try:
        fall_risk = df_schema[df_schema['column_name'].str.strip() == "fall_risk"]['answer'].iloc[0]
        fall_risk_text = f"- **Fall Risk**: {fall_risk} \n"
    except:
        fall_risk_text = ''
    try:
        point_of_contact = df_schema[df_schema['column_name'].str.strip() == "point_of_contact"]['answer'].iloc[0]
        point_of_contact_text = f"- **POC**: {point_of_contact} \n"
    except:
        point_of_contact_text = ''
    try:
        mental_status = df_schema[df_schema['column_name'].str.strip() == "mental_status"]['answer'].iloc[0]
        mental_status_text = f"- **Mental Status**: {mental_status} \n"
    except:
        mental_status_text = ''
    try:
        gastrointestinal  = df_schema[df_schema['column_name'].str.strip() == "gastrointestinal"]['answer'].iloc[0]
        gastrointestinal_text = f"- **GI**: {gastrointestinal} \n"
    except:
        gastrointestinal_text = ''
    try:
        genitourinary  = df_schema[df_schema['column_name'].str.strip() == "genitourinary"]['answer'].iloc[0]
        genitourinary_text = f"- **GU**: {genitourinary} \n"
    except:
        genitourinary_text = ''
    try:
        input_output = df_schema[df_schema['column_name'].str.strip() == "input_output"]['answer'].iloc[0]
        input_output_text = f"- **I/O**: {input_output} \n"
    except:
        input_output_text = ''
    try:
        language = df_schema[df_schema['column_name'].str.strip() == "language"]['answer'].iloc[0]
        language_text = f"- **Language**: {language} \n"
    except:
        language_text = ''
    try:
        alcohol_smoking_drug_history  = df_schema[df_schema['column_name'].str.strip() == "alcohol_smoking_drug_history"]['answer'].iloc[0]
        alcohol_smoking_drug_history_text = f"- **Alc Smok Drug Hx**: {alcohol_smoking_drug_history} \n"
    except:
        alcohol_smoking_drug_history_text = ''
    try:
        orientation = df_schema[df_schema['column_name'].str.strip() == "orientation"]['answer'].iloc[0]
        orientation_text = f"- **Orientation**: {orientation} \n"
    except:
        orientation_text = ''
    try:
        disabilities = df_schema[df_schema['column_name'].str.strip() == "disabilities"]['answer'].iloc[0]
        disabilities_text = f"- **Disabilities**: {disabilities} \n"
    except:
        disabilities_text = ''
    try:
        medical_history = df_schema[df_schema['column_name'].str.strip() == "medical_history"]['answer'].iloc[0]
        medical_history_text = f"- **Med Hx**: {medical_history} \n"
    except:
        medical_history_text = ''
    try:
        surgical_history = df_schema[df_schema['column_name'].str.strip() == "surgical_history"]['answer'].iloc[0]
        surgical_history_text = f"- **Surg Hx**: {surgical_history} \n"
    except:
        surgical_history_text = ''

    text = 'Metadata: :green[**done**]'
    st.write(text)

    return room_text, patient_name_text, code_status_text, allergies_text, primary_doctor_text, isolation_precautions_text, pain_status_text, imaging_delays_text, chief_complaint_text, diet_text, fall_risk_text, point_of_contact_text, mental_status_text, gastrointestinal_text, genitourinary_text, language_text, alcohol_smoking_drug_history_text, orientation_text, disabilities_text, medical_history_text, surgical_history_text, df_schema

# Run the function & pull out the variables I need
result = section2_generate_elements()
room_text, patient_name_text, code_status_text, allergies_text, primary_doctor_text, isolation_precautions_text, pain_status_text, imaging_delays_text, chief_complaint_text, diet_text, fall_risk_text, point_of_contact_text, mental_status_text, gastrointestinal_text, genitourinary_text, language_text, alcohol_smoking_drug_history_text, orientation_text, disabilities_text, medical_history_text, surgical_history_text, df_schema= result

################
### # Section 3: Labs, Vitals, Meds
################

@st.cache_data
def section3_labs_vitals_meds():

    st.divider()
    st.header('3. Labs, Vitals, Meds')

    st.write('''
    Generate labs, meds, vitals lists – abnormal only & full list \n
    LLM prompt: generate low/normal/high on vitals
    ''')

    ####### Labs ###########

    st.divider()
    text = 'Lab Results: :orange[**start**]'
    st.write(text)

    #### Create Labs List
        ## ALERT !! ASW Assumption: "Labs" is the exact name of this sections
    labs_string = [s["full_string"] for s in section_list2 if s.get("section_name") == "Labs"]
    labs_string = labs_string[0]
        ## ALERT !! ASW Assumption: "Labs" has every line broken out by "\n"
    labs_list_pre = labs_string.split('\n')
    # st.write(labs_list_pre)

    # Create list
    labs_list = []
    for string in labs_list_pre:
        values = string.split(',')
        ## ALERT !! ASW Assumption: "Labs" has 6 comma-separated values using this schema:
        if len(values) == 7:
            dict = {
                "labs_type": values[0].strip(),
                "lab_test": values[1].strip(),
                "measurement": values[2].strip(),
                "unit": values[3].strip(),
                "reading": values[4].strip(),
                "date": values[5].strip(),
                "time": str(datetime.strptime(values[6].replace("\\n","").strip(), "%I:%M %p").time())
            }
            labs_list.append(dict)

    # Create df
    df_labs = pd.DataFrame(labs_list)

    # Add 2 columns + sort
    df_labs['full_lab_name'] = df_labs['labs_type'] + '|' + df_labs['lab_test']
    df_labs['date_time'] = df_labs['date'] + ' ' + df_labs['time']
    df_labs = df_labs.sort_values(by=['date_time','full_lab_name'], ascending=[False,True])

    # Add red bold if reading is abnormal
    def add_string(row):
        if row['reading'] in ["low", "high"]:
            return ':red[**' + row['reading'] + '**]'
        else:
            return row['reading']
    df_labs['reading_color'] = df_labs.apply(add_string, axis=1)

    # Create a final string value, all results
    df_labs['string'] = '- ' + df_labs['labs_type'] + ', ' + df_labs['lab_test'] + ', ' + df_labs['measurement'] + ' ' + df_labs['unit'] + ', ' + df_labs['reading_color'] + ', ' + df_labs['date_time']
    df_labs_publish = df_labs.loc[:, ['labs_type', 'lab_test', 'measurement', 'unit', 'reading', 'date_time']]
    lab_results_all = df_labs['string'].str.cat(sep=' \n ')

    # Create a final string value, only abnormal, only keep first one
    df_labs = df_labs[df_labs['reading'].isin(["low", "high"])]
    df_labs = df_labs.drop_duplicates(subset='full_lab_name', keep='first')
    df_labs_publish_abnormal = df_labs.loc[:, ['labs_type', 'lab_test', 'measurement', 'unit', 'reading', 'date_time']]
    lab_results_abnormal = df_labs['string'].str.cat(sep=' \n ')

    text = 'Lab Results: :green[**done**]'
    st.write(text)

    ####### Vitals ###########

    # Vitals List

    st.divider()
    text = 'Vitals: :orange[**start**]'
    st.write(text)

        ## ALERT !! ASW Assumption: "Vitals" is the exact name of this sections
    vitals_string = [s["full_string"] for s in section_list2 if s.get("section_name") == "Vitals"]
    vitals_string = vitals_string[0]
    # st.write(vitals_string)
        ## ALERT !! ASW Assumption: "Vitals" has every line broken out by "\n"
    vitals_list_pre = vitals_string.split('\n')
    # st.write(vitals_list_pre)

    vitals_list = []
    for string in vitals_list_pre:
        values = string.split(',')
        ## ALERT !! ASW Assumption: "Vitals" has 6 comma-separated values using this schema:
        if len(values) == 6:
            dict = {
                "vitals_type": values[0].strip(),
                "vitals_sub_type": values[1].strip(),
                "measurement": values[2].strip(),
                "unit": values[3].strip(),
                "reading": "not read yet",
                "date": values[4].strip(),
                "time": str(datetime.strptime(values[5].replace("\\n","").strip(), "%I:%M %p").time())
            }
            vitals_list.append(dict)
    # st.write(vitals_list)

    # Create df
    df_vitals = pd.DataFrame(vitals_list)

    # Run an LLM loop to determine if the result is normal
    for index, row in df_vitals.iterrows():
        vitals_type = row['vitals_type'] + ' ' + row['vitals_sub_type']
        vitals_reading = row['measurement'] + ' ' + row['unit']

        prompt_text = f'''
    *** Context: *** 
    Below is a vitals reading for a patient. 

    *** Request: *** 
    Please determine whether the vital is normal, low, or high. Respond with only 3 possible values: 
    - "normal" 
    - "low"
    - "high"

    *** Additional Details: *** 
    Resperatory rate (or "resp rate"): Normal is 12-18 breaths per minute (br/min)
    Fasting blood sugar: normal is 70-100 mg/dL (3.9 and 5.6 mmol/L)
    Random blood sugar: normal is less than 125 mg/dL (6.9 mmol/L)
    O2 (oxygen) saturation: normal is 95% or greater
    If the blood sugar test is not specified, assume it's a random blood sugar reading
    Assume any height is normal 
    Assume any weight is normal

    *** Patient Vitals: *** 
    Test: {vitals_type}
    Measurement: {vitals_reading}
    '''

        model_token_limit = 5
        # Run the LLM
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
            f'''{prompt_text}''',
            **parameters
        )
        llm_response = response.text

        # Update reading column
        reading = llm_response.strip()
        df_vitals.loc[index, 'reading'] = reading
        text = ':blue[**' + vitals_type.strip() + '**] : ' + vitals_reading + ' -- **' + reading + '**'
        st.write(text)

    # Add 2 columns + sort
    df_vitals['full_vital_name'] = df_vitals['vitals_type'] + ' ' + df_vitals['vitals_sub_type']
    df_vitals['date_time'] = df_vitals['date'] + ' ' + df_vitals['time']
    df_vitals = df_vitals.sort_values(by=['date_time','full_vital_name'], ascending=[False,True])

    # Add red bold if reading is abnormal
    def add_string(row):
        if row['reading'] in ["low", "high"]:
            return ':red[**' + row['reading'] + '**]'
        else:
            return row['reading']
    df_vitals['reading_color'] = df_vitals.apply(add_string, axis=1)

    ### Height, weight, BMI -- take most recent value 
        ## ALERT !! ASW Assumption: Height, weight, BMI are listed as "Height", "Weight", "BMI"
    # Height - take only most recent value
    df_hwb = df_vitals[df_vitals['vitals_type'].isin(["Height"])].reset_index(drop=True)
    df_hwb = df_hwb.drop_duplicates(subset='vitals_type', keep='first')
    try: 
        height = '- **Height**: ' + df_hwb.at[0, 'measurement'] + ' ' + df_hwb.at[0, 'unit'] + ' \n '
    except:
        height = '' 

    df_hwb = df_vitals[df_vitals['vitals_type'].isin(["Weight"])].reset_index(drop=True)
    df_hwb = df_hwb.drop_duplicates(subset='vitals_type', keep='first')
    try: 
        weight = '- **Weight**: ' + df_hwb.at[0, 'measurement'] + ' ' + df_hwb.at[0, 'unit'] + ' \n '
    except:
        weight = '' 

    df_hwb = df_vitals[df_vitals['vitals_type'].isin(["BMI"])].reset_index(drop=True)
    df_hwb = df_hwb.drop_duplicates(subset='vitals_type', keep='first')
    try: 
        bmi = '- **BMI**: ' + df_hwb.at[0, 'measurement'] + ' ' + df_hwb.at[0, 'unit'] + ' \n '
    except:
        bmi = '' 

    # Create a final string value, all results
    df_vitals['string'] = '- ' + df_vitals['full_vital_name'] + ', ' + df_vitals['measurement'] + ' ' + df_vitals['unit'] + ', ' + df_vitals['reading_color'] + ', ' + df_vitals['date_time']
    df_vitals_publish = df_vitals.loc[:, ['full_vital_name', 'measurement', 'unit', 'reading', 'date_time']]
    vital_results_all = df_vitals['string'].str.cat(sep=' \n ')

    # Create a final string value, only abnormal, only keep first one
    df_vitals = df_vitals[df_vitals['reading'].isin(["low", "high"])]
    df_vitals = df_vitals.drop_duplicates(subset='full_vital_name', keep='first')
    df_vitals_publish_abnormal = df_vitals.loc[:, ['full_vital_name', 'measurement', 'unit', 'reading', 'date_time']]
    vital_results_abnormal = df_vitals['string'].str.cat(sep=' \n ')

    text = 'Vitals: :green[**done**]'
    st.write(text)

    ####### Meds ############  

    st.divider()
    text = 'Meds: :orange[**done**]'
    st.write(text)

    ## Meds List
        ## ALERT !! ASW Assumption: "Medications" is the exact name of this sections
    meds_string = [s["full_string"] for s in section_list2 if s.get("section_name") == "Medications"]
    meds_string = meds_string[0]
        ## ALERT !! ASW Assumption: "Meds" has every line broken out by "\n"
    meds_list_pre = meds_string.split('\n')
    # st.write(meds_list_pre)

    meds_list = []
    for string in meds_list_pre:
        values = string.split(',')
        ## ALERT !! ASW Assumption: "Meds" has 6 comma-separated values using this schema:
        if len(values) == 6:
            dict = {
                "medication_name": values[0].strip(),
                "medication_dose": values[1].strip(),
                "medication_frequency": values[2].strip(),
                "medication_route": values[3].strip(),
                "date": values[4].strip(),
                "time": str(datetime.strptime(values[5].replace("\\n","").strip(), "%I:%M %p").time())
            }
            meds_list.append(dict)
    # st.write(meds_list)

    # Create df
    df_meds = pd.DataFrame(meds_list)

    # Add last + next medication time
        ## ALERT !! ASW Assumption: "medication_frequency" has values "Q12H", "Q8H", "Q6H", "Q4H", "QD", etc.: 
    df_meds['date_time'] = df_meds['date'] + ' ' + df_meds['time']
    def map_frequency(value):
        if value == "Q12H":
            return 12
        elif value == "Q8H":
            return 8
        elif value == "Q6H":
            return 6
        elif value == "Q4H":
            return 4
        elif value == "QD":
            return 24
        else:
            return None  # Handle other values if needed
    df_meds['hours_until_next'] = df_meds['medication_frequency'].apply(map_frequency)
    df_meds['next_med_time'] = df_meds['date_time'].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"))

    # Sort by latest date_time first, de-dup, then sort by next_time_up first
    df_meds = df_meds.sort_values(by=['date_time','medication_name'], ascending=[False,True])
    df_meds = df_meds.drop_duplicates(subset='medication_name', keep='first')
    df_meds = df_meds.sort_values(by=['next_med_time','date_time','medication_name'], ascending=[True,False,True])

    # Non-PRN values
        ## ALERT !! ASW Assumption: "PRN" is capitalized and included in medication frequency column: 
    df_meds_non_prn = df_meds[~df_meds['medication_frequency'].str.contains('PRN')]
    df_meds_non_prn['string'] = '- ' + df_meds_non_prn['medication_name'] + ', ' + df_meds_non_prn['medication_dose'] + ', ' + df_meds_non_prn['medication_frequency'] + ', ' + df_meds_non_prn['medication_route'] + '; :blue[**Next Medication Time:**] **' + df_meds_non_prn['next_med_time'] + '** (last took: ' + df_meds_non_prn['date_time'] + ')'
    df_meds_non_prn_publish = df_meds_non_prn.loc[:, ['medication_name', 'medication_dose', 'medication_frequency', 'medication_route', 'next_med_time', 'date_time']]
    med_results_non_prn = df_meds_non_prn['string'].str.cat(sep=' \n ')

    # PRN values
    df_meds_prn = df_meds[df_meds['medication_frequency'].str.contains('PRN')]
    df_meds_prn['string'] = '- ' + df_meds_prn['medication_name'] + ', ' + df_meds_prn['medication_dose'] + ', ' + df_meds_prn['medication_frequency'] + ', ' + df_meds_prn['medication_route'] + ' (last took: ' + df_meds_prn['date_time'] + ')'
    df_meds_prn_publish = df_meds_prn.loc[:, ['medication_name', 'medication_dose', 'medication_frequency', 'medication_route', 'date_time']]
    med_results_prn = df_meds_prn['string'].str.cat(sep=' \n ')

    text = 'Meds: :green[**done**]'
    st.write(text)

    return lab_results_abnormal, vital_results_abnormal, med_results_non_prn, med_results_prn, height, weight, bmi, lab_results_all, vital_results_all, df_labs_publish_abnormal, df_labs_publish, df_vitals_publish_abnormal, df_vitals_publish, df_meds_non_prn_publish, df_meds_prn_publish

# Run the function & pull out the variables I need
result = section3_labs_vitals_meds()
lab_results_abnormal, vital_results_abnormal, med_results_non_prn, med_results_prn, height, weight, bmi, lab_results_all, vital_results_all, df_labs_publish_abnormal, df_labs_publish, df_vitals_publish_abnormal, df_vitals_publish, df_meds_non_prn_publish, df_meds_prn_publish = result

################
### # Section 4: Generate summary
################

@st.cache_data
def section4_generate_summary():

    st.divider()
    st.header('4. Generate Summary')

    st.write('''
    Generate a summary
    # LLM prompt: generate a 4-5 sentence summary
    ''')

    # Build prompt for summary
    prompt_text = f'''
Context: Below are details about a medical patient 

Request: Please write a 4-5 sentence summary of the patient

Rules: 
1. Only use information provided in the patient details section below
2. Do not make up any information about the patient 
3. Do not make any assumptions about the patient that is not explicitly included in the patient details below

Patient Details: 

# General Details
{patient_name_text}
{primary_doctor_text}
{allergies_text}
{isolation_precautions_text}
{imaging_delays_text}
{chief_complaint_text}
{diet_text}
{fall_risk_text}
{mental_status_text}
{gastrointestinal_text}
{genitourinary_text}
{language_text}
{alcohol_smoking_drug_history_text}
{orientation_text}
{disabilities_text}
{medical_history_text}
{surgical_history_text}

# Abnormal Vitals
{vital_results_abnormal}

# Abnormal Labs 
{lab_results_abnormal}

# Medications
{med_results_non_prn}
{med_results_prn}
'''
    model_token_limit = 200
    # Run the LLM
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
        f'''{prompt_text}''',
        **parameters
    )
    llm_response = response.text
    patient_summary = llm_response
    st.write(":blue[**Summary**] : ", patient_summary)

    return patient_summary

# Run the function & pull out the variables I need
result = section4_generate_summary()
patient_summary = result

################
### # Section 5: Generate json structure
################

@st.cache_data
def section5_generate_json_structure():

    st.divider()
    st.header('5. Generate JSON Structure')

    st.write('''
    Generate a json sturcture with data from sections 2,3,4 above
    ''')

    json_data = {
        'patient_summary': patient_summary,
        'room': room_text,
        'patient_name': patient_name_text,
        'code_status': code_status_text,
        'allergies': allergies_text,
        'primary_doctor': primary_doctor_text,
        'isolation_precautions': isolation_precautions_text,
        'pain_status': pain_status_text,
        'imaging_delays': imaging_delays_text,
        'chief_complaint': chief_complaint_text,
        'diet': diet_text,
        'fall_risk': fall_risk_text,
        'point_of_contact': point_of_contact_text,
        'mental_status': mental_status_text,
        'gastrointestinal': gastrointestinal_text,
        'genitourinary': genitourinary_text,
        'language': language_text,
        'alcohol_smoking_drug_history': alcohol_smoking_drug_history_text,
        'orientation': orientation_text,
        'disabilities': disabilities_text,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'medical_history': medical_history_text,
        'surgical_history': surgical_history_text,
        'lab_results_abnormal': df_labs_publish_abnormal.to_dict(orient='records'),
        'lab_results_all': df_labs_publish.to_dict(orient='records'),
        'vital_results_abnormal': df_vitals_publish_abnormal.to_dict(orient='records'),
        'vital_results_all': df_vitals_publish.to_dict(orient='records'),
        'med_results_non_prn': df_meds_non_prn_publish.to_dict(orient='records'),
        'med_results_prn': df_meds_prn_publish.to_dict(orient='records')
    }

    # Convert dictionary to json
    st.write(json_data)
    json_string = json.dumps(json_data, indent=4)

section5_generate_json_structure()

################
### # Section 6: Generate final report
################

@st.cache_data
def section6_generate_final_report():

    st.divider()
    st.header('6. Generate Final Report')

    st.write('''
    Generate final report UX
    ''')

    final_report = f'''

# Summary 
{patient_summary}
        
# Header
{room_text}
{patient_name_text}
{code_status_text}
{allergies_text}
{primary_doctor_text}
{isolation_precautions_text}
{pain_status_text}
{imaging_delays_text}

# Always want to know
{chief_complaint_text}
{diet_text}
{fall_risk_text}
{point_of_contact_text}

# Abnormals / By Exception
## Labs
{lab_results_abnormal}

## Vitals 

{vital_results_abnormal}

## Assessment Items
{mental_status_text}
{gastrointestinal_text}
{genitourinary_text}
{language_text}
{alcohol_smoking_drug_history_text}
{orientation_text}
{disabilities_text}

# Medication / Orders

## Next Medications Due
{med_results_non_prn}

## PRN Meds
{med_results_prn}

# Medical Summary
## General Info
{height}
{weight}
{bmi}
{medical_history_text}
{surgical_history_text}

## All Labs 
{lab_results_all}

## All Vitals
{vital_results_all}
    '''

    st.markdown(final_report)

section6_generate_final_report() 

################
### # Section 7: Highligh Evidence
################

def section7_highlight_evidence():

    st.divider()
    st.header('7. Highlight Evidence')

    st.write('''
    Highlight relevant sections of code
    ''')

    # Select a value
    column_list = df_schema['column_name'].tolist()
    param_column = st.selectbox(
        "Select an option:"
        , column_list
        , index = 1 
    )
    st.write(":blue[**Attribute**] : ", param_column)

    # Narrow down df 
    df = df_schema[df_schema['column_name'].str.strip() == param_column.strip()]
    answer_text = df['answer'].iloc[0]
    evidence_text = df['evidence'].iloc[0]
    st.write(":blue[**Answer**] : ", answer_text)
    st.write(":blue[**Evidence**] : ", evidence_text)

    # Find the position of the string
    main_string = section_list_no_vitals_meds_labs_1_string
    position = main_string.find(evidence_text)
    if position == -1:
        st.write(f"The substring '{evidence_text}' was not found in the main string.")
    else:
        # Grab the X # characters before and after
        num_characters = 250
        start = max(0, position - num_characters)
        end = min(len(main_string), position + len(evidence_text) + num_characters)
        text_pararaph_subset = main_string[start:end]

        # Highlight the relevant text
        text_paragraph = text_pararaph_subset
        text_highlight = evidence_text
        text_replace = ":orange[**" + text_highlight + "**]"

        text_paragraph_with_formatting = text_paragraph.replace(text_highlight, text_replace)

        st.write(':blue[**Evidence in Context:**] ')
        st.write(text_paragraph_with_formatting)

section7_highlight_evidence() 