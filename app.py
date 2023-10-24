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

import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title

# Make page wide
st.set_page_config(
    page_title="Google Cloud Generative AI",
    layout="wide",
  )

# Title
st.title('Google Cloud HCLS Generative AI Demos')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')

# Get started
st.write('Click on a demo on the left hand side to get started')

# Gitlink
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

add_page_title() # By default this also adds indentation

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be

# list of emojis: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

show_pages(
    [
          Page("app.py", "Home", ":house:")
        
        , Section("QUESTION AND ANSWER", ":question:")
        , Page("pages/Looker HDE Ask Question.py", "QNA on HDE Looker")
        , Page("pages/Supply Chain PO Chat Answers.py", "QNA on Supply Chain Purchase Orders")
        
        , Section("SUMMARIZATION", ":books:")
        , Page("pages/Patient Summary.py", "Generate Patient Summary")
        , Page("pages/Highlight Demo.py", "Highlight Demo")

        , Section("CATEGORIZATION", ":twisted_rightwards_arrows:")
        , Page("pages/Bariatric Med Record to Struct.py", "Label Medical Record to Struct")
        , Page("pages/Imaging Med Record to Struct.py", "Label Medical Imaging for Data Sci Model")

        , Section("WORKFLOW", ":gear:")
        , Page("pages/Doctor_Appt_Concierge.py", "Doctor Appt Scheduling")
        , Page("pages/Travelling Nurse.py", "Map Route for Travelling Nurse")

        , Section("MULTIMODAL", ":camera:")
        , Page("pages/Multimodal Imaging.py", "Read Mammogram Image & Text")
        , Page("pages/Speech to Text.py", "QNA with a Video")

        , Section("CODE GENERATION", ":computer:")
        , Page("pages/Looker Explore Generator.py", "Text 2 Looker")
        , Page("pages/Write a SQL Query.py", "Text 2 SQL")

        # , Section("Under Construction - Come Back Later", ":building_construction:")
        # , Page("pages/Large Doc QA.py", "Large Doc Q&A")
        # , Page("pages/Vertex Search Large Docs.py", "Large Doc QNA")
    ]
)