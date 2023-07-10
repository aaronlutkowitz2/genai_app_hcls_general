#Base Image to use
FROM python:3.8-slim

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt
COPY cloudadopt-test-genai-real.json app/cloudadopt-test-genai-real.json

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

# mount service account key to docker
# RUN gcloud auth activate-service-account --key-file=cloudadopt-ef95f1cf8614.json
RUN export GOOGLE_APPLICATION_CREDENTIALS="app/cloudadopt-test-genai-real.json"

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]


# COPY cloudadopt-ef95f1cf8614.json app/cloudadopt-ef95f1cf8614.json
# RUN apt-get -y update
# RUN apt-get -y install curl
# RUN apt-get -y install tar
# RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-437.0.1-linux-x86_64.tar.gz

# RUN tar -xf google-cloud-cli-437.0.1-linux-x86_64.tar.gz
# RUN ./google-cloud-sdk/install.sh
# RUN gcloud auth application-default login --impersonate-service-account cloudadopt-test-owner@cloudadopt.iam.gserviceaccount.com
