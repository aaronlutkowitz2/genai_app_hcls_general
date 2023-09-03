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
        
        , Section("QUESTION ANSWERING", ":question:")
        , Page("pages/Looker HDE Ask Question.py", "Q&A on HDE / Looker")
        , Page("pages/Supply Chain PO Chat Answers.py", "Q&A on Supply Chain (Purchase Orders)")
        , Page("pages/Website Info Bot.py", "Q&A on Public Websites (Many Documents)")
        
        , Section("CATEGORIZATION", ":twisted_rightwards_arrows:")
        , Page("pages/Bariatric Med Record to Struct.py", "Label Sections of Medical Record, to Struct")
        , Page("pages/Imaging Med Record to Struct.py", "Label Medical Imaging for Data Sci Model")

        , Section("WORKFLOW", ":gear:")
        , Page("pages/Doctor_Appt_Concierge.py", "Doctor Appt Scheduling")

        , Section("MULTIMODAL", ":camera:")
        , Page("pages/Multimodal Imaging.py", "Read Mammogram (Image & Text)")
        , Page("pages/Speech to Text.py", "Q&A with a Video")

        , Section("CODE GENERATION", ":computer:")
        , Page("pages/Write a SQL Query.py", "Write a SQL Query")

        # , Section("Under Construction - Come Back Later", ":building_construction:")
        # , Page("pages/Large Doc QA.py", "Large Doc Q&A")
        
    ]
)