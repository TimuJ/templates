# Install base Python image
FROM python:3.8-slim-buster

# Copy files to the container
COPY *.joblib /model/
COPY requirements.txt /model/

# Set working directory to previously added app directory
WORKDIR /model/

# Install dependencies
RUN pip install -r  requirements.txt

# Expose the port uvicorn is running on
EXPOSE 19

# Run uvicorn server
CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "19"]