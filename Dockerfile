#Base Image to use
FROM python:3.8-slim

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt
# COPY cloudadopt-ef95f1cf8614.json app/cloudadopt-ef95f1cf8614.json

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

# mount service account key to docker
# RUN gcloud auth activate-service-account --key-file=cloudadopt-ef95f1cf8614.json
# RUN export GOOGLE_APPLICATION_CREDENTIALS="app/cloudadopt-ef95f1cf8614.json"


#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
