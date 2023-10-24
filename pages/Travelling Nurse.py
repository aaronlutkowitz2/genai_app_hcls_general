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
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import struct_pb2
from google.cloud import bigquery
import gcsfs
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import utils_config

# others
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import pandas as pd
import requests
from urllib.parse import quote
import json
import urllib
import math
import googlemaps
from datetime import datetime

from ipywidgets import embed
import gmaps

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Travelling Nurse Maps')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-10-20')
st.write('**Purpose**: Travelling Nurse: Optimize Routes')

# Gitlink
# st.write('**Go Link (Googlers)**: go/hclsgenai')
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('60 Second Video')

# video_url = 'https://youtu.be/b7JXKP2-dFQ'
# st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vT-MlgO6vtKwyHIbjvqD1kF-x2no9XXEYxblaCk30IjgRc3xlXWFuAg9RGVmbOBkWd-hSJccUajlFmi/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### Optimize Routes
################

st.divider()

st.header('1. Provide List of Addresses')

num_address = st.number_input(
    label = "How many addresses does the nurse go to on their route?", 
    min_value = 2, 
    max_value = 24, 
    value = 24
)

# Pull in list of addresses
project_name = "cloudadopt"
file_name = "travelling_nurse/map_addresses.csv"
bucket_name = "hcls_genai"
client = storage.Client(project_name)
bucket = client.bucket(bucket_name)
blob = bucket.blob(file_name)
csv_data = blob.download_as_text()
st.write(":blue[step]: file " + file_name + " downloaded :green[done]")
csv_lines = csv_data.split('\n')
csv_lines = csv_lines[1:]
# st.write(csv_lines)
csv_split = [string.split(',') for string in csv_lines]
df = pd.DataFrame(csv_split, columns=['street_address','city','state','zip','address_type'])
df['row_number'] = df.index + 1
df['full_address'] = df['street_address'] + " " + df["city"] + ", " + df["state"] # + " " + df["zip"].astype(str)
df = df.sort_values(by=['street_address', 'address_type'], ascending=[True, False])
address_list_pre = df['full_address'].tolist()
address_list = [x.replace(' ','+').replace(',','') for x in address_list_pre]
address_list = address_list[:num_address]
st.write(":blue[step]: address list created :green[done]")
# st.write(address_list)

address_selected = st.multiselect(
    'What addresses does the nurse need to go to?',
    address_list,
    address_list
)

st.write('The nurse will go to the following addresses:', address_selected)

st.header('2. Create a distance matrix')
st.write('Use Google Maps to create the # minutes (or # miles) from every point to every other point')

# Documentation -- https://developers.google.com/optimization/routing/vrp#distance_matrix_api

def create_data():
  """Creates the data."""
  data = {}
  data['API_key'] = 'AIzaSyBAdBDCcxdRJGwpoLLJrnKXXuXzeHfsTjk'
  data['addresses'] = address_list
  return data

def create_distance_matrix(data):
  addresses = data["addresses"]
  API_key = data["API_key"]
  # Distance Matrix API only accepts 100 elements per request, so get rows in multiple requests.
  max_elements = 100
  num_addresses = len(addresses) # 16 in this example.
  # Maximum number of rows that can be computed per request (6 in this example).
  max_rows = max_elements // num_addresses
  # num_addresses = q * max_rows + r (q = 2 and r = 4 in this example).
  q, r = divmod(num_addresses, max_rows)
  dest_addresses = addresses
  distance_matrix = []
  # Send q requests, returning max_rows rows per request.
  for i in range(q):
    origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)

  # Get the remaining remaining r rows, if necessary.
  if r > 0:
    origin_addresses = addresses[q * max_rows: q * max_rows + r]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)
  return distance_matrix

def send_request(origin_addresses, dest_addresses, API_key):
  """ Build and send request for the given origin and destination addresses."""
  def build_address_str(addresses):
    # Build a pipe-separated string of addresses
    address_str = ''
    for i in range(len(addresses) - 1):
      address_str += addresses[i] + '|'
    address_str += addresses[-1]
    return address_str

  request = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial'
  origin_address_str = build_address_str(origin_addresses)
  dest_address_str = build_address_str(dest_addresses)
  request = request + '&origins=' + origin_address_str + '&destinations=' + \
                       dest_address_str + '&key=' + API_key
  jsonResult = urllib.request.urlopen(request).read()
  response = json.loads(jsonResult)
  return response

def build_distance_matrix(response):
  distance_matrix = []
  for row in response['rows']:
    row_list = [row['elements'][j]['duration']['value'] for j in range(len(row['elements']))]
    distance_matrix.append(row_list)
  return distance_matrix

def main():
  """Entry point of the program"""
  st.write(":blue[step]: distance matrix :orange[start]")
  # Create the data.
  data = create_data()
  # addresses = data['addresses']
  # API_key = data['API_key']
  distance_matrix = create_distance_matrix(data)
  st.write(distance_matrix)
  st.write(":blue[step]: distance matrix :green[done]")
  return distance_matrix
if __name__ == '__main__':
  distance_matrix = main()

st.header('3. Solve the route')
st.write('Use Google OR (optimization research) tools to create the route with the shortest number of total minutes of drive time')

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    total_seconds = solution.ObjectiveValue()
    total_minutes = total_seconds / 60
    total_minutes_int = int(total_minutes)
    st.markdown(f'**Total Time:** The fastest route will take the nurse **{total_minutes_int} minutes** to drive')
    # st.write('Objective: {} seconds'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Optimal Route: \n '
    route_string = ''
    route_distance = 0
    while not routing.IsEnd(index):
        route_string += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += route_string + ' {}\n'.format(manager.IndexToNode(index))
    # st.write(plan_output)
    
    # Convert route string to a route list, then join that back to addresses 
    route_string += ' {}\n'.format(manager.IndexToNode(index)) #  "0 -> 1 -> 3 -> 4 -> 2 -> 5 -> 0"
    route_list = [int(item.strip()) for item in route_string.split('->')]
    address_list_route = sorted(zip(address_list, route_list), key=lambda x: x[1])
    route_list_final = [pair[0] for pair in address_list_route]
    route_list_print = '**Optimal Route:** \n\n - ' + "\n\n - ".join(route_list_final)
    st.markdown(route_list_print)
    st.write(":blue[step]: solve the route :green[done]")

    # Show a map of the route
    st.divider()
    st.header('4. Map the Route')
    map_url1 = 'https://www.google.com/maps/embed/v1/directions?'
    apikey = 'AIzaSyBAdBDCcxdRJGwpoLLJrnKXXuXzeHfsTjk' # figure out later how not to repeat this
    map_apikey = f'key={apikey}' 
    origin = route_list_final[0]
    map_origin = f'&origin={origin}'
    waypoints = "|".join(route_list_final[1:-1])
    map_waypoints = f'&waypoints={waypoints}'
    destination = route_list_final[0]
    map_destination = f'&destination={destination}'
    map_html = map_url1 + map_apikey + map_origin + map_waypoints + map_destination
    components.iframe(map_html,height=800) # width=960,height=569

def main():
    """Entry point of the program."""
    st.write(":blue[step]: solve the route :orange[start]")
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution)

if __name__ == '__main__':
    main()