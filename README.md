# streamlit-test-app
Streamlit Demo App 


General HCLS demo: https://genaihcls-zzsg7awyia-uc.a.run.app/

Example code for General HCLS demo: 

# Terminal on local machine
>> cd Documents/ 
>> mkdir git_repos

# clone repo
>> git clone https://github.com/aaronlutkowitz2/genai_app_hcls_general 
>> cd genai_app_hcls_general/

# test app locally
>> streamlit run app.py 

# build docker image (you need to install docker first)
>> docker build . -t genai_hcls 

# login to your GCP account to carry code
>> gcloud auth application-default login 

# build docker container on container registry
>> gcloud builds submit --tag gcr.io/cloudadopt/genai_hcls --timeout=2h 

# Go to Cloud Run in your GCP project, create (or update) an app to point to the newest container in Container Registry --> genai_hcls
# If there's any auth issues, go to IAM page to add more permissions to the cloud engine service account
# If there's an issue with one of the container commands, test that docker code works on docker (locally) 
>> docker run -p 8080:8080 --name test_container genai_hcls

# Note: you may have to delete test_container first - Then the URL should work with latest code
>> docker rm test_container
