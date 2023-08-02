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
st.write('**Date**: 2023-08-02')
st.write('**Purpose**: Use Multimodal model to predict cancer detection')

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

_, container, _ = st.columns([side, width, side])
# container.video(data='https://youtu.be/AtVCwywl_q8')

# Architecture

st.divider()
st.header('Architecture')

# components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vT-l1SQEqf6DrvlDMT_YhULvY74U1SnVCyfC7EVgXt2bPN4c6bejjPb0GeNjt4SHnz3v0t4SHjM-S-9/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

