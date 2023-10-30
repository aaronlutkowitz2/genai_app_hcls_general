#Base Image to use
FROM python:3.8-slim

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt 
RUN pip install --no-dependencies langchain==0.0.242

### Potential scribbles additionals
# RUN apt-get update && apt-get install -y \
#     libasound2-dev \
#     portaudio19-dev \
#     libportaudio2 \
#     libportaudiocpp0 \
#     ffmpeg
# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt


#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]



#############
## Graveyard
#############

# COPY cloudadopt-test-genai-real.json app/cloudadopt-test-genai-real.json

# mount service account key to docker
# RUN gcloud auth activate-service-account --key-file=cloudadopt-ef95f1cf8614.json
# RUN export GOOGLE_APPLICATION_CREDENTIALS="app/cloudadopt-test-genai-real.json"

# COPY cloudadopt-ef95f1cf8614.json app/cloudadopt-ef95f1cf8614.json
# RUN apt-get -y update
# RUN apt-get -y install curl
# RUN apt-get -y install tar
# RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-437.0.1-linux-x86_64.tar.gz

# RUN tar -xf google-cloud-cli-437.0.1-linux-x86_64.tar.gz
# RUN ./google-cloud-sdk/install.sh
# RUN gcloud auth application-default login --impersonate-service-account cloudadopt-test-owner@cloudadopt.iam.gserviceaccount.com

# manually install scann
# RUN git clone https://github.com/google-research/google-research/
# RUN python google-research/scann/setup.py install
# RUN pip install scann==1.2.9