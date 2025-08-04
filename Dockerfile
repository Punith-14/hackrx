# Use an official Python runtime as a parent image
# Using a 'slim' version keeps the final container size smaller
FROM python:3.11-slim

# Set the working directory inside the container
# All subsequent commands will be run from this directory
WORKDIR /code

# Copy the dependencies file to the working directory
# This is done first to leverage Docker's layer caching.
# If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces the image size by not storing the download cache
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container
# This includes your 'app' directory and any other project files.
COPY ./app /code/app

# Expose the port the app runs on
# This tells Docker that the container will listen on this port at runtime.
EXPOSE 8080

# Define the command to run your app
# This is the command that will be executed when the container starts.
# It runs the Uvicorn server, making it accessible from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
