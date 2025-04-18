# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# Add ffmpeg here
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8000 available to the world outside this container (for FastAPI)
EXPOSE 8000

# Define environment variable (optional, e.g., for logging level)
# ENV LOG_LEVEL=INFO

# Command to run the application (e.g., FastAPI server)
# Use CMD to specify the default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
