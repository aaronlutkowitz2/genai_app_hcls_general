# Generative AI Healthcare Examples

A sample application that shows off Generative AI use cases for Healthcare / Life Sciences (HCLS)

## Set up 

Select a directory to clone repo into

```bash
git clone https://github.com/aaronlutkowitz2/genai_app_hcls_general 
cd genai_app_hcls_general/
```

## Requirements

Install requirements, including Streamlit

```bash
pip install -r requirements.txt
```

## Test Locally

Optionally, test locally

```bash
export PROJECT_ID=$(gcloud info --format='value(config.project)')

streamlit run app.py 
```


## Deploy to Google Cloud

Two options, create a docker container and then push the Artifact Registry or directly build & deploy to Cloud Run.

### Create a container locally (optional)

build docker image (you need to install docker first)
```bash
docker build . -t genai_hcls 
```

login to your GCP account to carry code

```bash
export PROJECT_ID=$(gcloud info --format='value(config.project)')
gcloud config set project $PROJECT_ID
gcloud auth application-default login 
```

### Build with Cloud Build

build docker container on container registry

```bash
export PROJECT_ID=$(gcloud info --format='value(config.project)')
gcloud builds submit --tag gcr.io/${PROJECT_ID}/genai-hcls
```

###  Deploy to Cloud Run

Go to Cloud Run in your GCP project, create (or update) an app to point to the newest container in Container Registry --> genai_hcls

```bash
gcloud run deploy genai-hcls --image gcr.io/${PROJECT_ID}/genai-hcls --allow-unauthenticated
```


## Troubleshooting

If there's any auth issues, go to IAM page to add more permissions to the cloud compute engine service account

If there's an issue with one of the container commands, test that docker code works on docker (locally) 

```bash
docker run -p 8080:8080 --name test_container genai_hcls
```

Note: you may have to delete test_container first - Then the URL should work with latest code

```bash
docker rm test_container
```

# Other Guides
https://medium.com/@faizififita1/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-ba776487c5fe

https://medium.com/analytics-vidhya/deploying-streamlit-apps-to-google-app-engine-in-5-simple-steps-5e2e2bd5b172

# Aaron notes (ignore these)
```bash
docker build . -t genai_hcls
```

```bash
gcloud builds submit --tag gcr.io/cloudadopt/genai_hcls --timeout=2h
```

```bash
docker build . -t genai_hca
```

```bash
gcloud builds submit --tag gcr.io/cloudadopt/genai_hca --timeout=2h 
```

```bash 
virtualenv -p python3 venv
```

```bash 
source venv/bin/activate
```
