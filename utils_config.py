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

"""
Configuration variables to be used by the streamlit app.
"""

import os
import requests

# Location
LOCATION = "us-central1"

# Chatbot Location: region for CX agent
CHATBOT_LOCATION = "global"

# Signing service account
# used in Large Document Question Answering
SIGNING_SERVICE_ACCOUNT = "837081393813-compute@developer.gserviceaccount.com"  # @param {type:"string"}
### TODO -- double check if I need this

# """Ensure your user account has the ability to sign URLs using the service account."""

# # Grant Service Account Token Creator role on the signing service acocunt
# !gcloud iam service-accounts add-iam-policy-binding {SIGNING_SERVICE_ACCOUNT} \
#   --member=user:{USER_EMAIL} \
#   --role=roles/iam.serviceAccountTokenCreator \
#   --billing-project {PROJECT_ID} \
#   --project {PROJECT_ID} \
#   -q


# HCLS GCS Bucket
# this is Aaron's public bucket of assets; no need to change this unless you create your own GCS bucket
BUCKET_NAME = "hcls_genai"


# Get project ID from metadata
def get_env_project_id() -> str:
    """Returns the Project ID from GAE or Cloud Run"""
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    if not project_id:
        project_id = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id", 
            headers={"Metadata-Flavor":"Google"}
        ).text

    return project_id
