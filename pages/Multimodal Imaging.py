#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation Date: July 10, 2023

@author: Aaron Wilkowitz
"""
################
### Notes on steps I took
################

# 1. Downloaded CBIS-DDSM dataset: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
# 2. Uploaded those files to public GCS bucket: hcls_genai/hcls/dicom  
# 3. Ran I ran the first section of this sql script: https://github.com/aaronlutkowitz2/genai_app_hcls_general/blob/main/non_python_code/mammogram/load_dicom_data.sql 
  #  -- this script produces a list of image_id, gcs_image_path, and metadata
# 4. Created vectors for every row in that table -- see the bottom of the script with a section called "### APPENDIX: create embeddings"
  #  -- inserted those vectors into cloudadopt.dicom_mammography.image_model_embedding
# 5. Ran section section of this sql script: https://github.com/aaronlutkowitz2/genai_app_hcls_general/blob/main/non_python_code/mammogram/load_dicom_data.sql 
  #  -- this calcs similarity scores from every image to every other image (using dot products)
  #  -- note: we also use dot product ^ 10 and dot product ^ 100 b/c the similiarity scores are so similar, we wanted to create more differentiation
  #  -- Eventually, we get to `cloudadopt.dicom_mammography.image_model_input_4` (wide) and `cloudadopt.dicom_mammography.image_model_input_5` (long), which we use in the scripts below
# In the demo, we "pre-baked" / pre-uploaded 3 images to run through, but you could easily code this to upload a brand new image, vectorize it, & run through the same process

## Note: I added in a section for text as well - same concept 

################
### import 
################
# gcp
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from google.cloud import bigquery
import gcsfs

# others
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import ast
from datetime import datetime
import datetime, pytz
import seaborn as sns
import base64
import sys
import time
import typing
import csv
import numpy as np
from IPython.display import Image
# from image_encoder.image_encoder import *

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
st.header('60 Second Video')

default_width = 80 
ratio = 1
width = max(default_width, 0.01)
side = max((100 - width) / ratio, 0.01)

_, container, _ = st.columns([side, width, side])
container.video(data='https://youtu.be/b7JXKP2-dFQ')

# Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSC37AXlqWdajZZhnK25SVWOhZIGzFX9HZ90hMdqTkwRx6I7vD1v7SNCTL3AStKA2tmDm1uzcW5v2Dx/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### select patient
################

st.divider()
st.header('1. Select a Patient')

patient_select = st.selectbox(
    'Which patient do you want to select?'
    , (
          "Mammogram of Patient 15 (Benign)"
        , "Mammogram of Patient 44 (Malignant)"
        , "Mammogram of Patient 25 (Benign wo Callback)"
        , "Text of benign patient, left breast, MLO orientation"
        , "Text of malignant patient, right breast, CC orientation"
      )
  )

if "15" in patient_select:
    patient_id = 15
    image_url = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/id15_benign.jpg'
    image_url_nearest1 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/15_nearest1_id8.jpg'
    image_url_nearest2 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/15_nearest2_id480.jpg'
    image_url_nearest3 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/15_nearest3_id287.jpg'
    image_file_name = 'hcls/dicom/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.403050124711420404835698594840910402827/1-254.jpg'
    original_string = 'XX'
elif "44" in patient_select:
    patient_id = 44
    image_url = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/id44_malig.jpg'
    image_url_nearest1 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/44_nearest1_id289.jpg'
    image_url_nearest2 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/44_nearest2_id436.jpg'
    image_url_nearest3 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/44_nearest3_id415.jpg'
    image_file_name = 'hcls/dicom/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.408053815012156684010240636493066172601/1-201.jpg'
    original_string = 'XX'
elif "25" in patient_select:
    patient_id = 25 
    image_url = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/id250_benign_wo_cb.jpg'
    image_url_nearest1 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/25_nearest1_id675.jpg'
    image_url_nearest2 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/25_nearest2_id82.jpg'
    image_url_nearest3 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/25_nearest3_id20.jpg'
    image_file_name = 'hcls/dicom/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.43851505613003999219808826221286958753/1-202.jpg' 
    original_string = 'XX'
elif "MLO orientation" in patient_select:
    patient_id = 10001
    image_url = 'XX'
    image_url_nearest1 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/15_nearest2_id480.jpg'
    image_url_nearest2 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/25_nearest2_id82.jpg'
    image_url_nearest3 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/text1_nearest3_id424.jpg'
    image_file_name = 'XX' 
    original_string = "A mammogram of a benign reading of a left breast with MLO orientation"
elif "CC orientation" in patient_select:
    patient_id = 10002 
    image_url = 'XX'
    image_url_nearest1 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/text2_nearest1_id18.jpg'
    image_url_nearest2 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/text2_nearest2_id543.jpg'
    image_url_nearest3 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/text2_nearest3_id546.jpg'
    image_file_name = 'XX'
    original_string = "A mammogram of a malignant reading of a right breast with CC orientation"
else:
    patient_id = 15
    image_url = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/id15_benign.jpg'
    image_url_nearest1 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/15_nearest1_id8.jpg'
    image_url_nearest2 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/15_nearest2_id480.jpg'
    image_url_nearest3 = 'https://raw.githubusercontent.com/aaronlutkowitz2/genai_app_hcls_general/main/data/images/mammogram/15_nearest3_id287.jpg'
    image_file_name = 'hcls/dicom/jpeg/file4/1.3.6.1.4.1.9590.100.1.2.403050124711420404835698594840910402827/1-254.jpg'
    original_string = 'XX'

st.write(':blue[**Original Mammogram:**] ')
if patient_id < 1000:
    question_type = 'image'
    st.image(image_url, width=400)
else:
    question_type = 'text'
    st.write(original_string)

st.write(':blue[**Patient Facts:**] ')
client_bq = bigquery.Client()

table_name = f'cloudadopt.dicom_mammography.{question_type}_model_input_4'
sql = f"""
SELECT 
    id
  , patientorientation 
  , left_or_right_breast
  , breast_density
  , mass_pathology
  , mass_subtlety
FROM `{table_name}`
WHERE id = {patient_id}
LIMIT 1
"""
# st.text(sql)
df = client_bq.query(sql).to_dataframe()
patient_orientation = df['patientorientation'].values[0] 
left_right_breast = df['left_or_right_breast'].values[0] 
breast_density = df['breast_density'].values[0] 
mass_pathology = df['mass_pathology'].values[0] 
mass_subtlety = df['mass_subtlety'].values[0] 
st.table(df)

################
### define embeddings
################

project_id = 'cloudadopt'

# Go to Colab: https://colab.sandbox.google.com/github/tankbattle/hello-world/blob/master/Build_Cloud_CoCa_Image_Embedding_Dataset_%26_Search.ipynb#scrollTo=x2BrVlM-phGN
# Inspired from https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple.
class EmbeddingResponse(typing.NamedTuple):
  text_embedding: typing.Sequence[float]
  image_embedding: typing.Sequence[float]

class EmbeddingPredictionClient:
  """Wrapper around Prediction Service Client."""
  def __init__(self, project : str,
    location : str = "us-central1",
    api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
    client_options = {"api_endpoint": api_regional_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    self.location = location
    self.project = project

  def get_embedding(self, text : str = None, image_bytes : bytes = None):
    if not text and not image_bytes:
      raise ValueError('At least one of text or image_bytes must be specified.')

    instance = struct_pb2.Struct()
    if text:
      instance.fields['text'].string_value = text

    if image_bytes:
      encoded_content = base64.b64encode(image_bytes).decode("utf-8")
      image_struct = instance.fields['image'].struct_value
      image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

    instances = [instance]
    endpoint = (f"projects/{self.project}/locations/{self.location}"
      "/publishers/google/models/multimodalembedding@001")
    response = self.client.predict(endpoint=endpoint, instances=instances)

    text_embedding = None
    if text:
      text_emb_value = response.predictions[0]['textEmbedding']
      text_embedding = [v for v in text_emb_value]

    image_embedding = None
    if image_bytes:
      image_emb_value = response.predictions[0]['imageEmbedding']
      image_embedding = [v for v in image_emb_value]

    return EmbeddingResponse(
      text_embedding=text_embedding,
      image_embedding=image_embedding)

client = EmbeddingPredictionClient(project=project_id)

# Extract image embedding
def getImageEmbeddingFromImageContent(content):
  response = client.get_embedding(text=None, image_bytes=content)
  return response.image_embedding

def getImageEmbeddingFromGcsObject(gcsBucket, gcsObject):
  client = storage.Client()
  bucket = client.bucket(gcsBucket)
  blob = bucket.blob(gcsObject)

  with blob.open("rb") as f:
    return getImageEmbeddingFromImageContent(f.read())

def getImageEmbeddingFromFile(filePath):
  with open(filePath, "rb") as f:
    return getImageEmbeddingFromImageContent(f.read())

# Extract text embedding
def getTextEmbedding(text):
  response = client.get_embedding(text=text, image_bytes=None)
  return response.text_embedding

################
### Generate a text embedding 
################

st.divider()
if question_type == 'image':
    st.header('2. Generate a Unique Vector for Image')
else:
    st.header('2. Generate a Unique Vector for Text')

st.write(f"We will now use Vertex AI to create a unique 1048-dimension vector describing this {question_type}")

image_bucket_name = "hcls_genai"
# Note: the line below is what you'd actually use to pull embeddings -- but no need to run it everytime -- what I'm using just for the demo is a partial hard-coded embedding
# embedding = getImageEmbeddingFromGcsObject(image_bucket_name, image_file_name)
embedding = '[0.0192286577, 0.0154227763, -0.00861405395, -0.0110541359, 0.00163213036, 0.0169473756, 0.0313014351, 0.0117419316, -0.0133915171, -0.0278525781, -0.0225973539, -0.0145366518, -0.00996714551, 0.0100956941, -0.0322998278, 0.00426189415, 0.000678597658, 0.00289598852, 0.043383915, -0.0168046039, -0.0338218361, 0.0273827296, -0.0387609936, 0.0243778713, -0.017838059, 0.0262691453, 0.0141063891, 0.0187563412, 0.00534455059, -0.0493374169, 0.0303051434, -0.0160473026, -0.0371797644, 0.00692701759, 0.0283875391, 0.0137267867, -0.031491518, -0.00344658154, -0.0162953809, 0.0282125603, 0.0195857305, 0.000363359926, -0.00886089, -0.0321243443, -0.0129843829, -0.0413851812, 0.00696028071, -0.00951314904, 0.00556926243, -0.00392219285, 0.0123352194, -0.00571729336, -0.0315009, -0.0318648852, 0.0276487563, -0.00751679204, -0.0159519389, -0.0142956199, -0.0331410542, -0.0134425694, -0.0391240343, -0.029512519, -0.0197206736, 0.00707804412, -0.00492096599, -0.00647117896, -0.0617857724, 0.018267259, 0.00858118385, -0.0132169547, -0.0118825734, -0.0259654224, 0.0140632382, 0.0322629586, -0.0155665344, 0.00599491037, -0.00619707862, -0.010116837, 0.014256373, 0.00291279797, -0.0188439488, -0.0233990755, -0.0196359698, 0.0290332343, 0.0343161412, 0.011853938, -0.0242382586, -0.0174036417, 0.0190506708, 0.0239307974, 0.0375967883, 0.00169354863, -0.00696445908, -0.212390184, -0.0393398367, -0.00412424514, 0.0287610441, -0.0191484429, -0.0194245372, 0.036573682, 0.00335917715, 0.0283351112, 0.00356064108, -0.00763658434, 0.0164249223, -0.0351715796, 0.0103293359, 0.0342771858, 0.0513855182, 0.0299044587, 0.00864771195, -0.00549998507, -0.0343904719, 0.053017538, -0.0121561456, -0.00862567406, -0.0286054928, -0.00373028382, 0.0181783382, -0.0164996106, 0.000548590499, -0.0232111346, -0.0140768243, -0.0108239083, -0.0181836095, -0.0924405232, -0.0562132485, 0.0230981708, -0.0188030675, -0.0289928503, -0.000302867556, -0.00210259855, 0.0382087156, -0.0170394182, -0.0265383273, -0.000453168206, 0.0123043815, 0.0101008462, 0.00863749906, 0.00581911532, -0.0136427572, 0.0227771848, -0.0166442432, -0.015468156, 0.0328641944, -0.0121320169, 0.010943478, 0.0164464042, 0.0587355979, -0.0292913243, 0.0260012206, -0.0259103961, -0.00273697358, -0.0440225638, -0.0198445078, 0.0131408246, -0.00456004776, -0.0125101386, 0.0281321425, -0.00958281476, -0.0104698669, 0.00703377603, -0.0381017551, -0.0235447641, -0.00269313878, -0.00408230396, -0.00427079573, 0.00809079502, 0.00154427579, -0.0168978013, -0.00532221142, -0.0203234, 0.00934536103, -0.00743831834, -0.00404534582, -0.00416333741, 0.00497120526, 0.0105548538, -0.0134845609, -0.0241767615, -0.0119690448, 0.0243872032, -0.0192877501, 0.00852868706, -0.0427979454, -0.00119108474]'
embedding_str = str(embedding)
# embedding_str500 = embedding_str[:500]

st.write(':blue[**Embedding:**] ')
st.text(embedding_str)

################
### Compare Performance vs Similar Images
################

st.divider()
st.header('3. What other mammograms have "similar" vectors?')

st.write("We have a database of several hundred other mammograms that we compare this vector against.")

images = [image_url_nearest1, image_url_nearest2, image_url_nearest3]
captions = ["Most Similar Image", "2nd Most Similar Image", "3rd Most Similar Image"]

st.image(images, width=200, caption=captions)

st.write("The similarity score is just the dot product between the two vectors")

st.write("Below is a table describing attributes of the 20 most similar images")

table_name = f'cloudadopt.dicom_mammography.{question_type}_model_input_4'
sql = f"""
SELECT 
    id_neighbor as id 
  , dot_product as sim_score
  , patientorientation_neighbor as patientorientation 
  , left_or_right_breast_neighbor as left_or_right_breast
  , breast_density_neighbor as breast_density
  , mass_pathology_neighbor as mass_pathology
  , mass_subtlety_neighbor as mass_subtlety
FROM `{table_name}`
WHERE id = {patient_id}
ORDER BY 2 desc 
LIMIT 20  
"""
df_rank = client_bq.query(sql).to_dataframe()

def highlight1(row):
    ret = ["" for _ in row.index]
    if row.patientorientation == patient_orientation:
        ret[row.index.get_loc("patientorientation")] = "color: green"
    else: 
        ret[row.index.get_loc("patientorientation")] = "color: red"

    if row.left_or_right_breast == left_right_breast:
        ret[row.index.get_loc("left_or_right_breast")] = "color: green"
    else: 
        ret[row.index.get_loc("left_or_right_breast")] = "color: red"

    if row.breast_density == breast_density:
        ret[row.index.get_loc("breast_density")] = "color: green"
    elif abs(int(breast_density) - int(row.breast_density)) <= 1 :
       ret[row.index.get_loc("breast_density")] = "color: orange"
    else: 
        ret[row.index.get_loc("breast_density")] = "color: red"

    if row.mass_pathology == mass_pathology:
        ret[row.index.get_loc("mass_pathology")] = "color: green"
    elif "BENIGN" in row.mass_pathology and "BENIGN" in mass_pathology :
       ret[row.index.get_loc("mass_pathology")] = "color: orange"
    else: 
        ret[row.index.get_loc("mass_pathology")] = "color: red"

    if row.mass_subtlety == mass_subtlety:
        ret[row.index.get_loc("mass_subtlety")] = "color: green"
    elif abs(int(mass_subtlety) - int(row.mass_subtlety)) <= 1 :
       ret[row.index.get_loc("mass_subtlety")] = "color: orange"
    else: 
        ret[row.index.get_loc("mass_subtlety")] = "color: red"
    return ret

df_rank_style = df_rank.style.apply(highlight1, axis=1)
st.table(df_rank_style)

st.divider()
st.header('4. Make "Predictions"')

st.write("Caveat: We're not actually making predictions. Instead we're taking the nearest 50 neighbors and weighting similarity scores to see which answer was most common in the most similar images")

def populate_prediction(question_type_var,attribute):
    st.write(f':blue[**{attribute}:**]')
    table_name = f'cloudadopt.dicom_mammography.{question_type_var}_model_input_5'
    sql = f"""    
    SELECT value_neighbor AS prediction_value
    FROM `{table_name}`
    WHERE measurement = '{attribute}' 
    GROUP BY 1 
    ORDER BY 1 
    """
    df = client_bq.query(sql).to_dataframe()
    prediction_values = df["prediction_value"].values.tolist()
    
    string = "'"
    prediction_values_new = [string + x + string for x in prediction_values]
    prediction_values_str = ",".join(prediction_values_new)

    sql = f"""
    with pre_data as (
    SELECT
            value AS actual
        , value_neighbor AS prediction
        , AVG(dot_product100) * 100 * 100 AS predicted_value
    FROM `{table_name}` a
    WHERE id = {patient_id}
    AND measurement = '{attribute}' 
    AND rank_neighbor < 50
    GROUP BY 1,2
    )
    , sum_predicted_values as (
    SELECT sum(predicted_value) as total_value
    FROM pre_data
    )
    , updated_values as (
    SELECT a.* except(predicted_value) 
    , a.predicted_value / b.total_value as predicted_value
    FROM pre_data a 
    , sum_predicted_values b 
    )
    , max_pred as (
    SELECT max(predicted_value) as max_predicted
    FROM updated_values
    )
    , pred_winner as (
    SELECT 
        a.actual
        , a.prediction 
    FROM updated_values a 
    INNER JOIN max_pred b
        ON a.predicted_value = b.max_predicted
    )
    , pivot_out as (
    SELECT * 
    FROM updated_values
    PIVOT(AVG(round(predicted_value,5)) FOR prediction IN ({prediction_values_str}))
    )
    SELECT 
        a.*
    , b.* except (actual)
    FROM pred_winner a 
    , pivot_out b
    """
    # st.text(sql)
    df = client_bq.query(sql).to_dataframe()
    prediction_value = df['prediction'].values[0]
    actual_value = df['actual'].values[0]

    if actual_value == prediction_value: 
        answer = 'Correct!'
        explanation = ""
    else: 
        answer = "Incorrect" 
        explanation = "The actual answer is " + actual_value + " while the model guessed " + prediction_value

    if actual_value == prediction_value: 
        st.write(f':green[**{answer}**]')
    else: 
        st.write(f':red[**{answer}**] ')
        st.write(explanation)

    def highlight2(row):
        ret = ["" for _ in row.index]
        if actual_value == prediction_value: 
            ret[row.index.get_loc("prediction")] = "background-color: green"
        else: 
            ret[row.index.get_loc("prediction")] = "background-color: red"
        return ret

    def highlight3(xyz):
        exec_string = f'xyz.background_gradient(axis=1, subset=[{prediction_values_str}], cmap=\"Blues\")'
        # st.write(exec_string)
        exec(exec_string)
        return xyz 

    def highlight4(row):
        ret = ["" for _ in row.index]
        ret[row.index.get_loc("prediction")] = "color: white"
        return ret

    df_style = df.style.pipe(highlight3).apply(highlight2, axis=1).apply(highlight4, axis=1)
    st.table(df_style)

    return attribute 

populate_prediction(question_type, "patient_orientation")
populate_prediction(question_type, "left_right_breast")

if question_type == 'image':
    populate_prediction(question_type, "breast_density")
else:
    pass 

populate_prediction(question_type, "mass_pathology")

if question_type == 'image':
    populate_prediction(question_type, "mass_subtlety")
else:
    pass 

################
### Overall Performance 
################

st.divider()
st.header('5. Overall Performance')

def show_looker_look(url):
    components.html(f'''
        <iframe src={url} 
        width='600' 
        height='338'
        frameborder='0'>
        </iframe>
    '''
    , width=620
    , height=358
    )
st.write('**Overall Performance**')
st.write('The 3 columns show the score for the same values, the score for different values, and the difference')
st.write('The higher the right most column, the better the model is at predicting the true value')
st.write('The model is quite good at identifying patient orientation')
st.write('The model is ok at identifying left/right breast, pathology, and density')
st.write('The model is not yet good at predicting mass subtlety')
source_url = 'https://demoexpo.cloud.looker.com/embed/public/TzMs3Vzb5x7Xvnb9QtnygHnS7dWjn2Cm'
show_looker_look(source_url)

st.write('**Patient Orientation**')
st.write('Blue boxes show values the models think are more similar')
st.write('Red boxes show values the models think are less similar')
st.write('Encouragingly, the model has higher similarities when actual & predicted are both CC or both MLO')
st.write('Encouragingly, the model has lower similarities when actual & predicted are different')
source_url = 'https://demoexpo.cloud.looker.com/embed/public/pYZDMrwpBNkQHsK27XPqGvJByZx7Q3Pd'
show_looker_look(source_url)

st.write('**Left or Right Breast**')
st.write('Encouragingly, the model has higher similarities when actual & predicted are both left or right')
st.write('Encouragingly, the model has lower similarities when actual & predicted are different')
source_url = 'https://demoexpo.cloud.looker.com/embed/public/srGgS6ZBnDxjsZ2cvbfhRdzP3hsvJJSx'
show_looker_look(source_url)

st.write('**Mass Pathology**')
st.write('The model is quite good at identifying benign without callback - the bright blue box in the center')
st.write('Benign & malignant is much murkier, purple - the model struggles to differentiate')
source_url = 'https://demoexpo.cloud.looker.com/embed/public/dbB3pFnvVHCkj9dzbGvM888y7wzmWHVk'
show_looker_look(source_url)

st.write('**Breast Density**')
st.write('Here the story is murkier - optimally, there would be blue boxes where actual & predicted are both 1,2,3, or 4')
st.write('Instead, while the model is pretty good at identifying 3s & 4s correctly (blue in bottom right boxes), it is less good at 1s and 2s')
source_url = 'https://demoexpo.cloud.looker.com/embed/public/3cjMpd4gt4FD8bNsmjXY6FRkZkrBJmPK'
show_looker_look(source_url)

st.write('**Mass Subtlety**')
st.write('The model struggles with subtlety - its predictions do not strongly correlate with actual values')
source_url = 'https://demoexpo.cloud.looker.com/embed/public/GW3vVFKJxKdWpTBqQX7MQTbfXsKwkQNt'
show_looker_look(source_url)






################################################################
### APPENDIX: create embeddings -- one time to generate embeddings on all images
################################################################

################
### SECTION 1: Images
################


#### One time process: run all 500 embeddings on the images

# project_id = 'cloudadopt' # @param {type: "string"}

# bucket_name = "hcls_genai"
# new_file_name = 'hcls/dicom/csv/image_model_input.csv'
# df = pd.read_csv('gs://' + bucket_name + '/' + new_file_name) 
# # df = df.head(2)
# # st.write(df)
# # df_embedding = pd.DataFrame(columns=['image_path', 'embedding_str'])
# client_bq = bigquery.Client()

# ## Add this junk line of code to ensure this does not run and re-calculate all embeddings
# xlkjlkajdsflkjasdf

# for index, row in df.iterrows():
#   image_bucket_name = "hcls_genai"
#   image_file_name = row['image_path']
#   embedding = getImageEmbeddingFromGcsObject(image_bucket_name, image_file_name)
#   embedding_str = str(embedding)
#     #   embedding_dict = {
#     #       'image_path': image_file_name
#     #     , 'embedding_str': embedding_str
#     #   }
#   # Append to df using dict
#   # df_embedding_new_row = pd.DataFrame(embedding_dict, index=[0])
#   # df_embedding = pd.concat([df_embedding, df_embedding_new_row])
#   # st.write(df_embedding)

#   # Construct a BigQuery client object.
#   table_name = 'cloudadopt.dicom_mammography.image_model_embedding'
#   query = f"""
#   INSERT INTO `{table_name}`
#   SELECT current_datetime(), '{image_file_name}', '{embedding_str}' 
#   """
#   query_job = client_bq.query(query)  # Make an API request.
#   # st.write(query)
#   st.write(image_file_name + ' done')

################
### SECTION 2: TEXT
################

# #### One time process: run embeddings on text

# sample_text1 = "A mammogram of a benign reading of a left breast with MLO orientation"
# sample_text2 = "A mammogram of a malignant reading of a right breast with CC orientation"
# sample_text3 = "A mammogram of a left breast with MLO orientation"
# sample_text4 = "A mammogram of a right breast with CC orientation"
# sample_text5 = "A mammogram with MLO orientation"
# sample_text6 = "A mammogram CC orientation"

# sample_list = [sample_text1, sample_text2, sample_text3, sample_text4, sample_text5, sample_text6]

# st.write(type(sample_list))
# project_id = 'cloudadopt' # @param {type: "string"}

# # bucket_name = "hcls_genai"
# # new_file_name = 'hcls/dicom/csv/image_model_input.csv'
# # df = pd.read_csv('gs://' + bucket_name + '/' + new_file_name) 
# # df = df.head(2)
# # st.write(df)
# # df_embedding = pd.DataFrame(columns=['image_path', 'embedding_str'])
# client_bq = bigquery.Client()

# ## Add this junk line of code to ensure this does not run and re-calculate all embeddings
# # xlkjlkajdsflkjasdf

# for index, value in enumerate(sample_list):
#   embedding = getTextEmbedding(value)
#   embedding_str = str(embedding)
#   index_value = index + 1 
#   value_value = value

#   # Construct a BigQuery client object.
#   table_name = 'cloudadopt.dicom_mammography.text_model_embedding'
#   query = f"""
#   INSERT INTO `{table_name}`
#   SELECT current_datetime(), {index_value}, '{value_value}', '{embedding_str}' 
#   """
#   query_job = client_bq.query(query)  # Make an API request.
#   # st.write(query)
#   st.write(value + ' done')
