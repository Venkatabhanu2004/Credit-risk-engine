# Credit-risk-engine
This project implements a Credit Risk Engine that predicts loan approval using a Random Forest Classifier.  It includes robust logging for monitoring, a Streamlit-based user interface for real-time predictions,  and Docker support for easy deployment across environments.
## Execution Guide

Follow these steps to build and run the project using Docker:

### Step 1: Build the Docker Image
Run the following command in your VS Code terminal:
docker build -t credit-risk-engine .

### Step 2: Run the Container
Expose the Streamlit app on port 8501:
docker run -p 8501:8501 credit-risk-engine

### Step 3: Access the Streamlit UI
Once the container is running, open the following URL in your browser:

Local URL: http://localhost:8501
### Note: Make sure Docker Desktop is open and running in background,so the server stays active.
