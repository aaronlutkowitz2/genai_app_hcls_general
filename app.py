#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation Date: July 10, 2023

@author: Aaron Wilkowitz
"""

import streamlit as st

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demos')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')

# Get started
st.header('Click on a demo on the left hand side to get started')
