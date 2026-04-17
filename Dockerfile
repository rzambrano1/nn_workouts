# Lightweight Python base image
FROM python:3.11-slim

# Set the directory inside the container
WORKDIR /app

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to optimize build speed
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code and config folders
COPY . .

# Setting environment variables (optional but helpful)
ENV PYTHONUNBUFFERED=1

# Setting up the command to run your script
ENTRYPOINT ["python", "train.py"]