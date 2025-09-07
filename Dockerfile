# Use a slim Python base image for smaller size and faster builds
FROM python:3.9-slim-buster

# Create a non-root user for security best practices
RUN groupadd --system appuser && \
    useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

# Install system dependencies required for various Python packages and document processing
# Combine update and install to ensure fresh packages and efficient layering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    curl gnupg \
    python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* # Clean up apt cache to reduce image size

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's build cache
# If requirements.txt doesn't change, this layer and subsequent pip install layer will be cached
COPY --chown=appuser:appuser requirements.txt ./requirements.txt

# Print requirements.txt content for debugging during build
RUN echo "--- Contents of requirements.txt during build ---" && cat requirements.txt

# Install Python dependencies
# Using --no-cache-dir for smaller final image size, though it might make local rebuilds slightly slower
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Removed redundant 'pip install wheel streamlit' as streamlit is in requirements.txt

# --- DIAGNOSTIC STEP: Verify Streamlit installation ---
RUN echo "--- Verifying Streamlit installation ---" && pip list | grep "streamlit"
RUN echo "--- Full pip list after all installs ---" && pip list

# Copy the rest of your application code
# This layer will only be rebuilt if application code changes, not dependencies
COPY --chown=appuser:appuser . .

# Create directories for documents, FAISS indices, user data, and temporary image storage
# Ensure correct permissions for the appuser for all directories the app will interact with
RUN mkdir -p /app/documents/professor_docs /app/documents/shared_docs \
             /app/faiss_index/professor/images_faiss_index /app/faiss_index/student/images_faiss_index \
             /tmp/uploaded_chat_images /tmp/extracted_images_for_all_kbs && \
    chown -R appuser:appuser /app/documents /app/faiss_index /tmp/uploaded_chat_images /tmp/extracted_images_for_all_kbs && \
    touch /app/users.json && \
    chown appuser:appuser /app/users.json

# Switch to the non-root user
USER appuser

# Expose the port Streamlit runs on (container internal port)
EXPOSE 8501

# Define environment variables (can be overridden by docker-compose.yml)
# These are crucial for your Python scripts to locate paths and Ollama settings.
ENV OLLAMA_HOST="http://ollama:11434"
ENV MODEL="gemma3:27b" 
ENV TEXT_EMBEDDING_MODEL="nomic-embed-text"
ENV MULTIMODAL_OLLAMA_MODEL="gemma3:27b" 
ENV UI_LANG="en"
ENV PROF_DOCS_PATH="/app/documents/professor_docs"
ENV SHARED_DOCS_PATH="/app/documents/shared_docs"
ENV PROF_VECTORSTORE_PATH="/app/faiss_index/professor"
ENV STUDENT_VECTORSTORE_PATH="/app/faiss_index/student"
# NEW: Explicitly define the environment variable for extracted images output directory
# Corrected env var name
ENV PERSISTENT_EXTRACTED_IMAGES_DIR="/app/faiss_index/extracted_images" 
ENV USERS_FILE="/app/users.json"

# Command to run the Streamlit application
# --server.port specifies the port Streamlit listens on inside the container
# --server.enableCORS false and --server.enableXsrfProtection false are recommended for Docker deployments
CMD ["streamlit", "run", "chatbot.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
