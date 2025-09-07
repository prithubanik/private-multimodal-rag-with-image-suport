Knowledge Management Chatbot with Local LLMs
This repository contains the source code for a private, multimodal knowledge management chatbot. Powered by local Large Language Models via Ollama and a Streamlit frontend, this application allows users to build and interact with a secure knowledge base using their own documents and images. It features a Retrieval-Augmented Generation (RAG) pipeline for accurate, context-aware answers, role-based access control, and is fully containerized with Docker for easy deployment and scalability.

🌟 Features
Local & Private: All processing and model inference happens locally. No data ever leaves your machine.

Retrieval-Augmented Generation (RAG): The chatbot answers questions based on a knowledge base you create by uploading your own documents.

Multimodal: Supports processing and understanding of both text (PDF, DOCX) and images (PNG, JPG), enabling visual search and analysis.

Role-Based Access: A simple login system differentiates between user roles (e.g., professor, student) to provide access to different document sets.

User-Friendly Interface: A clean and interactive web interface built with Streamlit for chatting, document upload, and knowledge base management.

Containerized: The entire application stack (Streamlit App, Ollama Server) is managed with Docker for easy deployment.

🏗️ Architecture
The application consists of three main services orchestrated by Docker Compose:

app (Streamlit Service):

Handles the user interface and user interactions.

Communicates with the Ollama service for LLM inference.

Manages document uploads, processing, and vector store creation.

Relies on app.py and document_processor_with_image_embedding.py.

ollama (Ollama Service):

Based on the official ollama/ollama image.

Serves the Large Language Models.

The Dockerfile.ollama pre-loads the required models during the image build process to ensure fast startup times.

Volumes:

documents: Persistent storage for uploaded documents.

faiss_index: Persistent storage for the generated FAISS vector stores (knowledge bases).

ollama_data: Persistent storage for Ollama models to avoid re-downloading on container restart.

🧠 How It Works
The application follows a Retrieval-Augmented Generation (RAG) workflow to provide accurate answers based on your documents.

Document Ingestion: A user uploads documents (PDFs, DOCX, images) via the Streamlit interface. The files are saved to a persistent volume.

Processing & Extraction: The backend script (document_processor_with_image_embedding.py) processes each file. It extracts text content from PDFs and DOCX files, performs OCR on standalone images, and extracts embedded images from PDFs.

Chunking: The extracted text is divided into smaller, semantically coherent chunks.

Embedding Generation:

For Text: Each text chunk is converted into a numerical vector (embedding) using the nomic-embed-text model.

For Images: The multimodal gemma3:27b model generates a detailed text description of each image. This description is then converted into an embedding using nomic-embed-text.

Vector Store Creation: All embeddings are stored and indexed in a FAISS vector database, creating a searchable knowledge base.

User Query: A user asks a question in the chat interface.

Query Embedding: The user's question is converted into an embedding using the same nomic-embed-text model.

Similarity Search: The system searches the FAISS vector store to find the text chunks and image descriptions whose embeddings are most similar to the query embedding.

Context Augmentation: The most relevant chunks and descriptions are retrieved to form the "context".

Answer Generation: The original question, chat history, and the retrieved context are passed to the gemma3:27b model. The model generates a natural language answer based only on the provided information.

Response: The final, context-aware answer is displayed to the user in the chat.

🚀 Getting Started
Prerequisites
Docker

Docker Compose

A machine with sufficient RAM (16GB+ recommended) and a GPU for better performance with Ollama.

Installation & Running
Clone the repository:

git clone [https://github.com/prithubanik/private-multimodal-rag-with-image-suport.git](https://github.com/prithubanik/private-multimodal-rag-with-image-suport.git)
cd private-multimodal-rag-with-image-suport

Create required directories:
The application uses mounted volumes to persist data. Create the necessary host directories if they don't exist.

mkdir -p documents/professor_docs documents/shared_docs
mkdir -p faiss_index/professor faiss_index/student

documents/professor_docs: For documents accessible only by the 'professor' role.

documents/shared_docs: For documents accessible by all roles.

faiss_index: This directory will be populated automatically with the vector stores.

Build and run the services using Docker Compose:

docker-compose up --build

The first build may take a significant amount of time as it will download the base Docker images and the LLMs specified in Dockerfile.ollama.

Access the application:
Once the containers are running, open your web browser and navigate to:
http://localhost:8501

🤖 Models Used
This application is configured to use the following models, which are automatically pulled by the ollama service as defined in Dockerfile.ollama:

nomic-embed-text: A high-performance text embedding model used for creating vector representations of the document content.

gemma3:27b: A powerful multimodal model used for chat completion, summarization, metadata extraction, and image understanding.