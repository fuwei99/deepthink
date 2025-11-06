# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code into the container
# The .dockerignore file will prevent logs, outputs, and other unnecessary files from being copied.
COPY . .

# 5. Expose the port the app runs on
EXPOSE 7860

# 6. Define the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
# Omit --reload for a production-like environment
# Add a longer keep-alive timeout to prevent proxies from closing the connection
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "300"]
