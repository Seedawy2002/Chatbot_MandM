# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first
COPY requirements.txt /app/

# Debugging step: List the contents of the requirements file
RUN cat /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt || true

# Copy the rest of the application files
COPY . /app/

# Download the SpaCy model explicitly (if needed)
RUN python -m spacy download en-core-web-sm || true

# Expose the application's port
EXPOSE 7860

# Define the command to run the application
CMD ["python", "src/app.py"]
