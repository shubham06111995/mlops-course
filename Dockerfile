# Start from a small, official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements file first (Docker caching trick -
# this layer only rebuilds if requirements.txt actually changes)
COPY requirements.txt .

# Install the Python packages inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of our project files into the container
COPY train.py .
COPY predict.py .

# When the container starts, train the model then run predictions
CMD ["sh", "-c", "python train.py && python predict.py"]
