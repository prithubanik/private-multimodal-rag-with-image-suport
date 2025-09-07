import sys
import os
import time
import logging
import requests
import string
from typing import List, Optional, Generator, Any
import uuid
import re
import json
import shutil
import gc
import base64
import mimetypes

# --- Streamlit Import ---
import streamlit as st

# === Streamlit UI Setup ===
# Set page config and custom CSS
st.set_page_config(
    page_title="IMFAA Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #e9ecef;
        color: #343a40;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
        border-radius: 18px;
        background-color: #ffffff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .stTextInput>div>div>input {
        border-radius: 28px;
        padding: 14px 22px;
        border: 1px solid #ced4da;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease-in-out;
        font-size: 1rem;
    }
    .stTextInput>div>div>input:focus {
        border-color: #28a745;
        box_shadow: inset 0 2px 5px rgba(0,0,0,0.08), 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
        outline: none;
    }

    .stButton>button {
        border-radius: 28px;
        border: none;
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 14px 30px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 6px 15px rgba(40, 167, 69, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #218838 0%, #1e7e34 100%);
        box_shadow: 0 8px 20px rgba(40, 167, 69, 0.4);
        transform: translateY(-3px);
    }
    .stButton>button:active {
        transform: translateY(0);
        box_shadow: 0 2px 5px rgba(40, 167, 69, 0.2);
    }

    .chat-container {
        max-height: 65vh;
        overflow-y: auto;
        padding-right: 15px;
        margin-bottom: 25px;
    }

    .stChatMessage {
        animation: fadeIn 0.6s ease-out;
        border-radius: 18px;
        padding: 15px 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .stChatMessage[data-testid="stChatMessage"][data-state="rendered"] {
        background-color: #343a40;
        color: white;
        border-bottom-left-radius: 8px;
        align-self: flex-start;
        margin-right: 20%;
    }
    .st-chat-message-user > div {
        background-color: #d1e7dd;
        color: #155724;
        border-bottom-right-radius: 8px;
        align-self: flex-end;
        margin-left: 20%;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .source-documents {
        font-size: 0.9em;
        color: #6c757d;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px dashed #e9ecef;
        line-height: 1.5;
    }
    .source-documents a {
        color: #28a745;
        text-decoration: none;
    }
    .source-documents a:hover {
        text_decoration: underline;
    }

    .stAlert:not(section.main[data-testid="stSidebar"] .stAlert) {
        color: #343a40 !important;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        font-weight: 500;
    }
    .stAlert:not(section.main[data-testid="stSidebar"] .stAlert) div[data-testid="stMarkdownContainer"] {
        color: #343a40 !important;
    }

    section.main[data-testid="stSidebar"] > div {
        background-color: #1a202c;
        color: #e0e0e0;
        box-shadow: 6px 0 20px rgba(0,0,0,0.3);
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    section.main[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #e0e0e0 !important;
    }
    section.main[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #3a4750 0%, #2f3b43 100%);
        color: #e0e0e0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #218838 0%, #1e7e34 100%);
        box_shadow: 0 8px 20px rgba(40, 167, 69, 0.4);
        transform: translateY(-3px);
    }
    .stButton>button:active {
        transform: translateY(0);
        box_shadow: 0 2px 5px rgba(40, 167, 69, 0.2);
    }

    section.main[data-testid="stSidebar"] label[data-testid^="st"] {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    section.main[data-testid="stSidebar"] div[data-testid="stFileUploader"] span,
    section.main[data-testid="stSidebar"] div[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
        color: #e0e0e0 !important;
    }

    section.main[data-testid="stSidebar"] .stFileUploader button {
        background-color: #3a4750 !important;
        color: #e0e0e0 !important;
        border: 1px solid #4a5a6a !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #218838 0%, #1e7e34 100%);
        box-shadow: 0 8px 20px rgba(40, 167, 69, 0.4);
        transform: translateY(-3px);
    }
    .stButton>button:active {
        transform: translateY(0);
        box_shadow: 0 2px 5px rgba(40, 167, 69, 0.2);
    }

    section.main[data-testid="stSidebar"] .stAlert.info,
    section.main[data-testid="stSidebar"] .stAlert.success,
    section.main[data-testid="stSidebar"] .stAlert.warning,
    section.main[data-testid="stSidebar"] .stAlert.error {
        background-color: #f8f9fa !important;
        color: #343a40 !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    section.main[data-testid="stSidebar"] .stAlert div[data-testid="stMarkdownContainer"] p,
    section.main[data-testid="stSidebar"] .stAlert p,
    section.main[data-testid="stSidebar"] .stAlert span {
        color: #343a40 !important;
    }
    section.main[data-testid="stSidebar"] .stAlert svg {
        fill: #343a40 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --- LangChain Imports ---
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_ollama import ChatOllama, OllamaEmbeddings # Corrected import for OllamaEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever # MODIFIED: Added ContextualCompressionRetriever
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage # Added for multimodal input
from langchain import hub # Added for agent prompt


# --- Document Processor Imports ---
# Import the document_processor_with_image_embedding module directly
# to ensure consistent configuration for image embeddings.
from document_processor_with_image_embedding import (
    load_vectorstore,
    process_file,
    add_documents_to_vectorstore,
    save_manifest,
    load_manifest,
    generate_file_hash,
    embedding_model, # This is the text embedding model
    vision_embeddings_model, # This is the global instance of RealVisionEmbeddings
    IMAGE_VECTORSTORE_DIR_NAME # Keep this for consistency
)

# Setup UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO, # Set to INFO for general use, DEBUG for more verbosity
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# === Constants ===
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
LLM_MODEL_NAME = os.environ.get("MODEL", "gemma3:27b")
DEFAULT_UI_LANG = os.environ.get("UI_LANG", "en")

# Define document and FAISS paths based on roles
PROF_DOCS_PATH = os.getenv("PROF_DOCS_PATH", "/app/documents/professor_docs")
SHARED_DOCS_PATH = os.getenv("SHARED_DOCS_PATH", "/app/documents/shared_docs")
PROF_VECTORSTORE_PATH = os.getenv("PROF_VECTORSTORE_PATH", "/app/faiss_index/professor")
STUDENT_VECTORSTORE_PATH = os.getenv("STUDENT_VECTORSTORE_PATH", "/app/faiss_index/student")

# New constant for persistent extracted images root path (shared across all KBs)
# This directory should be mounted as a Docker volume to ensure persistence.
PERSISTENT_EXTRACTED_IMAGES_DIR = os.getenv("PERSISTENT_EXTRACTED_IMAGES_DIR", "/app/faiss_index/extracted_images")
logger.info(f"Set PERSISTENT_EXTRACTED_IMAGES_DIR to: {PERSISTENT_EXTRACTED_IMAGES_DIR}")

# Ensure these directories exist
os.makedirs(PROF_DOCS_PATH, exist_ok=True)
os.makedirs(SHARED_DOCS_PATH, exist_ok=True)
os.makedirs(PROF_VECTORSTORE_PATH, exist_ok=True)
os.makedirs(STUDENT_VECTORSTORE_PATH, exist_ok=True)
os.makedirs(PERSISTENT_EXTRACTED_IMAGES_DIR, exist_ok=True) # Ensure new persistent extracted images dir exists
logger.info(f"Ensured global persistent extracted images directory exists: {PERSISTENT_EXTRACTED_IMAGES_DIR}")

# Temporary directory for directly uploaded chat images (will be base64 encoded for LLM)
TEMP_CHAT_IMAGES_DIR = "/tmp/uploaded_chat_images"
os.makedirs(TEMP_CHAT_IMAGES_DIR, exist_ok=True)
logger.info(f"Set TEMP_CHAT_IMAGES_DIR to: {TEMP_CHAT_IMAGES_DIR}")


# Path to the local user credentials file
USERS_FILE = os.getenv("USERS_FILE", "/app/users.json")


# === UI Text Dictionary for Multilingual Support ===
UI_TEXTS = {
    "en": {
        "title": "IMFAA Chatbot for Aalen University",
        "caption": f"Powered by {LLM_MODEL_NAME} via Ollama",
        "initial_greeting": "Hello! I'm IMFAA bot, your AI assistant for Aalen University. How can I help you regarding documents and IMFAA?",
        "chat_input_placeholder": "Ask your question about Aalen University documents...",
        "thinking_status": "Thinking...",
        "rag_processing_status": "Processing your request with RAG...",
        "rag_retrieving_status": "Attempting to retrieve answer from documents (RAG)...",
        "rag_generating_response": "Generating response...",
        "rag_retrieval_complete": "RAG retrieval complete.",
        "rag_inconclusive": "I couldn't find a definitive answer to your question in the provided documents. Please try rephrasing or provide more context.",
        "rag_no_kb": "Knowledge base not loaded. Please ensure documents are placed and processed.",
        "general_chat_status": "Responding to general query...",
        "general_chat_uninitialized": "I am a specialized assistant for Aalen University documents. I can answer general questions if the LLM is fully initialized.",
        "unknown_query_classification": "I'm not sure how to classify that query. Please ask a question related to Aalen University documents, or a general question.",
        "internal_error": "I apologize, but I'm currently unable to process your request due to an internal error. Please try again later.",
        "sources_expander": "Source Used:",
        "doc_mgmt_header": "Document Management",
        "doc_mgmt_info": "Upload documents here to extend the chatbot's knowledge base. Supported formats: PDF, DOCX, PNG, JPG, JPEG, TIFF. Images will be OCR'd for text extraction. Professors can choose the access level for their uploads; student uploads automatically go to shared documents.",
        "upload_files_label": "Upload New Documents",
        "upload_doc_access": "Select document access level",
        "access_level_prof_only": "Professor Only (Private)",
        "access_level_shared": "Shared (Both Professor & Student)",
        "processing_docs_selected_level": "Processing documents with '{level}' access...",
        "process_button": "Process Documents for My Role",
        "saving_processing_status": "Processing documents for current role...",
        "kb_status_header": "Knowledge Base Status (Current Role)",
        "kb_loaded_ready": "Knowledge Base Loaded and Ready!",
        "faiss_entries": "Total chunks (FAISS):",
        "bm25_ready": "BM25 Retriever Ready!",
        "image_kb_ready": "Image Knowledge Base Ready!",
        "image_kb_not_loaded": "Image Knowledge Base Not Loaded.",
        "image_kb_empty_hint": "Image KB is empty or has few entries. Upload more image-containing documents to 'Shared' folder for better results.",
        "kb_not_loaded": "Knowledge Base Not Loaded. Please process documents to build it.",
        "clear_kb_button": "Clear Knowledge Base", # Renamed button
        "kb_cleared_instruction": "Knowledge base cleared. Please click 'Process Documents for My Role' to rebuild it.", # New instruction
        "clear_chat_button": "Clear Chat History",
        "ollama_status_header": "Ollama Connection Status",
        "ollama_connected": "Ollama Server: Connected",
        "ollama_disconnected": "Ollama Server: Getrennt (Check Docker logs)",
        "about_header": "About IMFAA Chatbot",
        "about_content": f"This chatbot is designed to assist you with information related to Aalen University, leveraging Retrieval Augmented Generation (RAG) from uploaded documents. It uses local LLMs ({LLM_MODEL_NAME} via Ollama) and is built with Streamlit. You can ask factual questions about the documents, request summaries, or extract metadata from files.",
        "loading_docs": "Loading documents...",
        "loaded_docs_chunking": "Loaded {num_docs} documents. Chunking and embedding (FAISS)...",
        "building_bm25": "Building BM25 retriever...",
        "kb_updated_success": "Knowledge base (FAISS & BM25) updated successfully!",
        "kb_update_failed": "Failed to update knowledge base. Check logs for details.",
        "error_processing_docs": "An error occurred during document processing:",
        "no_docs_found": "No documents found in the designated folders for processing.",
        "uploaded_docs_processed": "Documents processed and knowledge base updated!",
        "no_files_saved": "No new files were uploaded to process. Processing existing documents.",
        "image_chat_upload_label": "Or upload an image here (e.g., a flowchart) for direct analysis (not for RAG).",
        "image_query_rag_response_intro": "\n\n**Searching your knowledge base for similar research based on your query:**",
        "login_header": "Login",
        "email_label": "Email",
        "password_label": "Password",
        "login_button": "Login",
        "logout_button": "Logout",
        "logged_in_as": "Logged in as {email} ({role})",
        "login_error": "Login failed. Check email and password.",
        "auth_welcome_message": "Welcome! Please login to use the chatbot.",
        "user_role_professor": "Professor",
        "user_role_student": "Student",
        "user_mgmt_header": "User Management (Admin Only)",
        "user_list_fetching": "Fetching user list...",
        "user_list_error": "Error fetching user list.",
        "user_table_email": "Email",
        "user_table_uid": "User ID",
        "user_table_role": "Role",
        "user_update_role_button": "Update Role",
        "role_update_success": "User '{email}' role updated to '{role}' successfully!",
        "role_update_error": "Failed to update role for '{email}'. {error_detail}",
        "select_new_role": "Select new role",
        "similar_images_found": "\n\n**Here are some visually similar images found in the knowledge base:**",
        "no_similar_images": "No visually similar images found in the knowledge base. Please ensure images have been uploaded and processed.",
        "metadata_extraction_error_hint": "Note: Extracting author names from PDFs can be challenging for AI models, especially if the formatting is complex. Double-check the document if the results are incomplete.",
        "image_staged_for_upload": "Image staged for upload: {filename}",
        "image_staged_prompt": "Image '{filename}' uploaded. What would you like to do with it? (e.g., 'find similar images', 'explain this image')",
        "image_cleared_warning": "You had an image staged, but your query was not image-specific. Processing as a text query. The image has been cleared from staging."
    },
    "de": {
        "title": "IMFAA Chatbot für die Hochschule Aalen",
        "caption": f"Betrieben mit {LLM_MODEL_NAME} via Ollama",
        "initial_greeting": "Hallo! Ich bin der IMFAA-Bot, Ihr KI-Assistent für die Hochschule Aalen. Wie kann ich Ihnen bezüglich Dokumenten und IMFAA helfen?",
        "chat_input_placeholder": "Stellen Sie Ihre Frage zu Dokumenten der Hochschule Aalen...",
        "thinking_status": "Denke nach...",
        "rag_processing_status": "Verarbeite Ihre Anfrage mit RAG...",
        "rag_retrieving_status": "Versuche, die Antwort aus Dokumenten (RAG) abzurufen...",
        "rag_generating_response": "Antwort wird generiert...",
        "rag_retrieval_complete": "RAG-Abruf abgeschlossen.",
        "rag_inconclusive": "Ich konnte in den bereitgestellten Dokumenten keine eindeutige Antwort auf Ihre Frage finden. Bitte formulieren Sie Ihre Frage um oder geben Sie mehr Kontext an.",
        "rag_no_kb": "Wissensbasis nicht geladen. Bitte stellen Sie sicher, dass Dokumente platziert und verarbeitet werden.",
        "general_chat_status": "Antworte auf allgemeine Anfrage...",
        "general_chat_uninitialized": "Ich bin ein spezialisierter KI-Assistent für Dokumente der Hochschule Aalen. Ich kann allgemeine Fragen beantworten, wenn das LLM vollständig initialisiert ist.",
        "unknown_query_classification": "Ich bin mir nicht sicher, wie ich diese Anfrage klassifizieren soll. Bitte stellen Sie eine Frage zu Dokumenten der Hochschule Aalen oder eine allgemeine Frage.",
        "internal_error": "Entschuldigung, aber ich kann Ihre Anfrage aufgrund eines internen Fehlers derzeit nicht bearbeiten. Bitte versuchen Sie es später erneut.",
        "sources_expander": "Verwendete Quelle:",
        "doc_mgmt_header": "Dokumentenverwaltung",
        "doc_mgmt_info": "Laden Sie hier Dokumente hoch, um die Wissensbasis des Chatbots zu erweitern. Unterstützte Formate: PDF, DOCX, PNG, JPG, JPEG, TIFF. Bilder werden zur Textextraktion OCR-verarbeitet. Professoren können die Zugriffsebene für ihre Uploads wählen; Studenten-Uploads gehen automatisch an gemeinsame Dokumente.",
        "upload_files_label": "Neue Dokumente hochladen",
        "upload_doc_access": "Zugriffsebene des Dokuments auswählen",
        "access_level_prof_only": "Nur Professor (Privat)",
        "access_level_shared": "Geteilt (Professor & Student)",
        "processing_docs_selected_level": "Verarbeite Dokumente mit '{level}' access...",
        "process_button": "Dokumente für meine Rolle verarbeiten",
        "saving_processing_status": "Verarbeite Dokumente für die aktuelle Rolle...",
        "kb_status_header": "Status der Wissensbasis (Aktuelle Rolle)",
        "kb_loaded_ready": "Wissensbasis geladen und bereit!",
        "faiss_entries": "Gesamte Chunks (FAISS):",
        "bm25_ready": "BM25 Retriever bereit!",
        "image_kb_ready": "Bild-Wissensbasis bereit!",
        "image_kb_not_loaded": "Bild-Wissensbasis nicht geladen.",
        "image_kb_empty_hint": "Die Bild-Wissensbasis ist leer oder enthält nur wenige Einträge. Laden Sie weitere bildhaltige Dokumente in den Ordner 'Geteilt' hoch, um bessere Ergebnisse zu erzielen.",
        "kb_not_loaded": "Wissensbasis nicht geladen. Bitte verarbeiten Sie Dokumente, um sie aufzubauen.",
        "clear_kb_button": "Wissensbasis löschen", # Renamed button
        "kb_cleared_instruction": "Wissensbasis gelöscht. Bitte klicken Sie auf 'Dokumente für meine Rolle verarbeiten', um sie neu aufzubauen.", # New instruction
        "clear_chat_button": "Chatverlauf löschen",
        "ollama_status_header": "Ollama Verbindungsstatus",
        "ollama_connected": "Ollama Server: Verbunden",
        "ollama_disconnected": "Ollama Server: Getrennt (Überprüfen Sie die Docker-Logs)",
        "about_header": "Über den IMFAA Chatbot",
        "about_content": f"This chatbot is designed to assist you with information related to Aalen University, leveraging Retrieval Augmented Generation (RAG) from uploaded documents. It uses local LLMs ({LLM_MODEL_NAME} via Ollama) and is built with Streamlit. You can ask factual questions about the documents, request summaries, or extract metadata from files.",
        "loading_docs": "Lade Dokumente...",
        "loaded_docs_chunking": "Es wurden {num_docs} Dokumente geladen. Zerlege und bette ein (FAISS)...",
        "building_bm25": "BM25 Retriever wird aufgebaut...",
        "kb_updated_success": "Wissensbasis (FAISS & BM25) erfolgreich aktualisiert!",
        "kb_update_failed": "Fehler beim Aktualisieren der Wissensbasis. Überprüfen Sie die Logs für Details.",
        "error_processing_docs": "Ein Fehler ist bei der Dokumentenverarbeitung aufgetreten:",
        "no_docs_found": "Keine Dokumente in den vorgesehenen Ordnern zur Verarbeitung gefunden.",
        "uploaded_docs_processed": "Dokumente verarbeitet und Wissensbasis aktualisiert!",
        "no_files_saved": "Keine neuen Dateien zum Verarbeiten hochgeladen. Verarbeite vorhandene Dokumente.",
        "image_chat_upload_label": "Oder laden Sie hier ein Bild (z.B. ein Flussdiagramm) zur direkten Analyse hoch (nicht für RAG).",
        "image_query_rag_response_intro": "\n\n**Searching your knowledge base for similar research based on your query:**",
        "login_header": "Login",
        "email_label": "Email",
        "password_label": "Password",
        "login_button": "Login",
        "logout_button": "Logout",
        "logged_in_as": "Angemeldet als {email} ({role})",
        "login_error": "Login failed. Check email and password.",
        "auth_welcome_message": "Welcome! Please login to use the chatbot.",
        "user_role_professor": "Professor",
        "user_role_student": "Student",
        "user_mgmt_header": "User Management (Admin Only)",
        "user_list_fetching": "Fetching user list...",
        "user_list_error": "Error fetching user list.",
        "user_table_email": "Email",
        "user_table_uid": "User ID",
        "user_table_role": "Role",
        "user_update_role_button": "Update Role",
        "role_update_success": "User '{email}' role updated to '{role}' successfully!",
        "role_update_error": "Failed to update role for '{email}'. {error_detail}",
        "select_new_role": "Select new role",
        "similar_images_found": "\n\n**Here are some visually similar images found in the knowledge base:**",
        "no_similar_images": "Keine visuell ähnliche Bilder in der Wissensbasis gefunden. Bitte stellen Sie sicher, dass Bilder hochgeladen und verarbeitet werden.",
        "logged_in_as": "Angemeldet als {email} ({role})",
        "logout_button": "Abmelden",
        "metadata_extraction_error_hint": "Hinweis: Das Extrahieren von Autorennamen aus PDFs kann für KI-Modelle eine Herausforderung sein, insbesondere wenn die Formatierung komplex ist. Überprüfen Sie das Dokument, wenn die Ergebnisse unvollständig sind.",
        "image_staged_for_upload": "Bild zum Hochladen bereit: {filename}",
        "image_staged_prompt": "Bild '{filename}' hochgeladen. Was möchten Sie damit tun? (z.B. 'ähnliche Bilder finden', 'dieses Bild erklären')",
        "image_cleared_warning": "Sie hatten ein Bild bereitgestellt, aber Ihre Anfrage war nicht bildspezifisch. Wird als Textanfrage verarbeitet. Das Bild wurde aus der Bereitstellung entfernt."
    }
}

# === Embeddings ===
# `embedding_model` is the text embedding model (e.g., nomic-embed-text)
# `vision_embeddings_model` is the RealVisionEmbeddings instance for multimodal
embeddings = embedding_model 
image_embeddings = vision_embeddings_model # Use the directly imported global instance


# === Prompt Templates ===
# MODIFIED: Enhanced qa_prompt for better contextual reasoning
qa_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history", "language"],
    template="""You are a highly capable and specialized AI assistant for Aalen University's IMFAA program. Your primary role is to provide precise, in-depth, and scientifically-grounded answers based on a knowledge base of research papers and master's theses.

**Analysis of Provided Context:**
First, analyze the nature of the provided 'Context'.
- If the context is drawn from **multiple diverse documents**, your task is to **synthesize** a comprehensive overview. Identify common themes, compare findings, and present a broad picture of the research topic based on all available information.
- If the context is primarily from **one specific document**, your task is to provide a **detailed and focused** answer based on the specific findings, experiments, and conclusions within that single source.

**Context (retrieved documents and relevant text):**
{context}

---
**Chat History:**
{chat_history}

---
**User's Question:** {question}

**Instruction:**
- Your response MUST be in the following language: {language}.
- **Core Task**: Based on your analysis of the context (synthesis vs. focused), provide a detailed, accurate, and comprehensive answer strictly derived from the provided text. Explain scientific concepts clearly.
- When answering, adopt an academic and informative tone suitable for a research environment.
"""
)

rag_fusion_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant that generates multiple search queries based on a single input question.
Generate 3-5 different versions of the user's question to retrieve relevant documents from a vector database.
Each query should be on a new line.

Original question: {question}
"""
)

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Please provide a detailed and comprehensive summary of the following document text:

{text}
"""
)

class PaperMetadata(BaseModel):
    title: Optional[str] = Field(description="The title of the research paper.")
    authors: Optional[List[str]] = Field(description="List of authors of the paper, including first and last names where available. If no authors are found, return an empty list.")

metadata_parser = PydanticOutputParser(pydantic_object=PaperMetadata)

metadata_extraction_prompt = PromptTemplate(
    template="""Extract the title and all authors from the following research paper abstract or document content.
Pay close attention to all author names, including first and last names, and list them clearly.
If affiliations are present, also try to extract them alongside the author names if possible.
If you cannot find a specific piece of information, return null for that field (for title) or an empty list (for authors).

Document Content:
{abstract}

{format_instructions}""",
    input_variables=["abstract"],
    partial_variables={"format_instructions": metadata_parser.get_format_instructions()}
)

# NEW: Prompt for Query Classification (improved with examples)
query_classification_prompt = PromptTemplate(
    input_variables=["query"],
    template="""Classify the following user query into one of these EXACT categories:
- GREETING
- TOOL_SUMMARIZE
- TOOL_METADATA
- RAG_QUESTION
- GENERAL_CHAT
- IMAGE_QUERY
- EXPLAIN_IMAGE
- UNKNOWN

Respond with ONLY the category name. Do not include any other text or explanations.

--- EXAMPLES ---
Query: Hello
Category: GREETING

Query: hi there
Category: GREETING

Query: Summarize the document on Agent S2.
Category: TOOL_SUMMARIZE

Query: Who are the authors mentioned in the Agent S2 paper?
Category: TOOL_METADATA

Query: what is Materials?
Category: RAG_QUESTION

Query: tell me more about the compositional framework in Agent S2 paper.
Category: RAG_QUESTION

Query: What is Material?
Category: GENERAL_CHAT

Query: find similar images
Category: IMAGE_QUERY

Query: explain this image
Category: EXPLAIN_IMAGE

Query: what is in this flowchart?
Category: EXPLAIN_IMAGE
--- END EXAMPLES ---

Query: {query}
Category:
"""
)

# --- Ollama Readiness Check ---
def wait_for_ollama_to_be_ready(base_url: str, retries: int = 10, delay: int = 5):
    """
    Waits for the Ollama server to be ready by repeatedly checking its /api/tags endpoint.
    """
    for i in range(retries):
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is ready.")
                return True
        except requests.exceptions.ConnectionError:
            logger.warning(f"Ollama server not reachable. Retrying in {delay} seconds... (Attempt {i+1}/{retries})")
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
        time.sleep(delay)
    logger.error("Ollama server did not become ready after multiple attempts.")
    return False

# === LLM Initialization (Cached) ===
@st.cache_resource
def get_llms():
    with st.spinner("Initializing LLM and checking Ollama connection..."):
        if not wait_for_ollama_to_be_ready(OLLAMA_BASE_URL):
            st.error("Ollama not ready after multiple attempts. Please ensure Ollama is running and accessible. Check Docker logs.")
            st.stop()
        
        llm_instance = ChatOllama(
            model=LLM_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3, # Increased temperature for more elaborate responses
            num_predict=1024,
            top_k=40,
            top_p=0.9
        )
        
        classifier_llm_instance = ChatOllama(
            model=LLM_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=64,
            top_k=1,
            top_p=1.0
        )
    return llm_instance, classifier_llm_instance

# Call the cached function to get LLM instances
llm, classifier_llm = get_llms()


# Function to classify query type
@st.cache_data
def classify_query(query: str) -> str:
    try:
        raw_classification_output = (query_classification_prompt | classifier_llm | (lambda x: x.content)).invoke({"query": query})
        cleaned_classification = raw_classification_output.strip().upper()
        
        possible_categories = ["GREETING", "TOOL_SUMMARIZE", "TOOL_METADATA", "RAG_QUESTION", "GENERAL_CHAT", "IMAGE_QUERY", "EXPLAIN_IMAGE", "UNKNOWN"]
        
        found_category = "UNKNOWN"
        for category in possible_categories:
            if category in cleaned_classification:
                found_category = category
                break
        
        category_match = re.search(r'CATEGORY:\s*([A-Z_]+)', cleaned_classification)
        if category_match:
            extracted_category = category_match.group(1)
            if extracted_category in possible_categories:
                found_category = extracted_category
        
        logger.info(f"DEBUG: LLM raw classification output: '{raw_classification_output}'")
        logger.info(f"DEBUG: Cleaned classification: '{cleaned_classification}'")
        logger.info(f"DEBUG: Final classified category: '{found_category}'")

        return found_category
    except Exception as e:
        logger.error(f"Error classifying query: {e}", exc_info=True)
        return "UNKNOWN"

# === Contextual Compression Retriever Setup ===
@st.cache_resource
def get_compressor(_llm_instance):
    return LLMChainExtractor.from_llm(_llm_instance)

compressor = get_compressor(llm)

# === RAG-Fusion Query Generator Chain ===
@st.cache_resource
def get_query_generator_runnable(_llm_instance):
    return rag_fusion_prompt | _llm_instance | (lambda x: x.content)

query_generator_runnable = get_query_generator_runnable(llm)

# === Custom RAG-Fusion Retriever Class ===
class RagFusionRetriever(BaseRetriever):
    vector_store: FAISS
    query_generator: Runnable
    compressor: LLMChainExtractor
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        logger.info(f"Generating RAG-Fusion queries for: {query}")
        try:
            generated_queries_raw = self.query_generator.invoke({"question": query})
            generated_queries = [q.strip() for q in generated_queries_raw.split('\n') if q.strip()]
            if not generated_queries:
                generated_queries = [query]
            logger.debug(f"Generated queries: {generated_queries}")
        except Exception as e:
            logger.error(f"Error generating RAG-Fusion queries: {e}. Falling back to original query.")
            generated_queries = [query]

        all_retrieved_docs = {}
        for q in generated_queries:
            initial_docs = self.vector_store.similarity_search(q, k=self.k * 2)
            compressed_docs = self.compressor.compress_documents(initial_docs, q)
            for doc in compressed_docs:
                doc_id = doc.metadata.get("uuid")
                if doc_id not in all_retrieved_docs:
                    all_retrieved_docs[doc_id] = doc

        final_docs = list(all_retrieved_docs.values())
        return final_docs[:self.k]

# === Vector Store Initialization / Reloading Logic (for current role) ===
if 'vector_store_instance' not in st.session_state:
    st.session_state.vector_store_instance = None
if 'bm25_retriever_instance' not in st.session_state:
    st.session_state.bm25_retriever_instance = None
if 'hybrid_retriever_instance' not in st.session_state:
    st.session_state.hybrid_retriever_instance = None
if 'uploaded_files_in_session' not in st.session_state:
    st.session_state.uploaded_files_in_session = []

if 'image_vector_store_instance' not in st.session_state:
    st.session_state.image_vector_store_instance = None
if 'image_kb_total_entries' not in st.session_state: # New state for image KB entry count
    st.session_state.image_kb_total_entries = 0

# New session state variables for image handling
if 'uploaded_chat_image_file_obj' not in st.session_state:
    st.session_state.uploaded_chat_image_file_obj = None
if 'last_processed_chat_image_path' not in st.session_state:
    st.session_state.last_processed_chat_image_path = None


if 'professor_faiss_ready' not in st.session_state:
    st.session_state.professor_faiss_ready = False
    st.session_state.professor_faiss_total_chunks = 0
    st.session_state.professor_bm25_ready = False
if 'professor_image_kb_ready' not in st.session_state:
    st.session_state.professor_image_kb_ready = False
    st.session_state.professor_image_kb_total_entries = 0

if 'student_faiss_ready' not in st.session_state:
    st.session_state.student_faiss_ready = False
    st.session_state.student_faiss_total_chunks = 0
    st.session_state.student_bm25_ready = False
if 'student_image_kb_ready' not in st.session_state:
    st.session_state.student_image_kb_ready = False
    st.session_state.student_image_kb_total_entries = 0


if 'kb_update_status_message' not in st.session_state:
    st.session_state.kb_update_status_message = None
if 'kb_update_status_type' not in st.session_state:
    st.session_state.kb_update_status_type = None


# --- User Authentication Functions (Local File Based) ---
def load_users():
    """Loads user data from the users.json file."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)
        return []
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {USERS_FILE}. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"Error loading users from {USERS_FILE}: {e}")
        return []

def save_users(users_data):
    """Saves user data to the users.json file."""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving users to {USERS_FILE}: {e}")

def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticates a user against the local users.json file."""
    users = load_users()
    for user in users:
        if user.get("email") == email and user.get("password") == password:
            return user
    return None

def add_user_to_file(email: str, password: str, role: str):
    """Adds a new user to the users.json file."""
    users = load_users()
    if any(u.get("email") == email for u in users):
        raise ValueError("User with this email already exists.")
    
    users.append({"email": email, "password": password, "role": role})
    save_users(users)


def get_role_from_email_domain(email: str) -> str:
    if email.endswith("@hs-aalen.de"):
        return "professor"
    elif email.endswith("@studmail.htw-aalen.de"):
        return "student"
    return "unknown"

def _trigger_kb_update(user_role: str, force_rebuild: bool = False):
    """
    Manages the knowledge base update/rebuild process for a given role.
    This function now handles the comprehensive manifest management and document processing.
    """
    st.session_state.kb_update_status_message = None
    st.session_state.kb_update_status_type = None

    st.toast(f"Starting knowledge base update for {user_role}...", icon="ℹ️")
    logger.info(f"Triggering KB update for role: {user_role}, force_rebuild: {force_rebuild}")
    
    st.session_state[f"{user_role}_faiss_ready"] = False
    st.session_state[f"{user_role}_faiss_total_chunks"] = 0
    st.session_state[f"{user_role}_bm25_ready"] = False
    st.session_state[f"{user_role}_image_kb_ready"] = False
    st.session_state[f"{user_role}_image_kb_total_entries"] = 0 # Reset image KB entries

    try:
        document_folders_to_load = []
        target_text_vectorstore_path = ""
        
        if user_role == "professor":
            document_folders_to_load.append(PROF_DOCS_PATH)
            document_folders_to_load.append(SHARED_DOCS_PATH)
            target_text_vectorstore_path = PROF_VECTORSTORE_PATH
            logger.info(f"Professor KB paths: docs={PROF_DOCS_PATH}, shared={SHARED_DOCS_PATH}, vectorstore={PROF_VECTORSTORE_PATH}")
            
        elif user_role == "student":
            document_folders_to_load.append(SHARED_DOCS_PATH)
            target_text_vectorstore_path = STUDENT_VECTORSTORE_PATH
            logger.info(f"Student KB paths: shared={SHARED_DOCS_PATH}, vectorstore={STUDENT_VECTORSTORE_PATH}")
        else:
            st.session_state.kb_update_status_message = "Invalid user role detected. Cannot load knowledge base."
            st.session_state.kb_update_status_type = "error"
            logger.error(f"Invalid user role: {user_role}")
            st.session_state.vector_store_instance = None
            st.session_state.bm25_retriever_instance = None
            st.session_state.hybrid_retriever_instance = None
            st.session_state.image_vector_store_instance = None
            return

        # Use the new PERSISTENT_EXTRACTED_IMAGES_DIR for images extracted from documents
        os.makedirs(PERSISTENT_EXTRACTED_IMAGES_DIR, exist_ok=True)
        logger.info(f"Ensured global persistent extracted images directory exists: {PERSISTENT_EXTRACTED_IMAGES_DIR}")

        # The image vector store path is now relative to the text vector store path
        global_image_vectorstore_path = os.path.join(target_text_vectorstore_path, IMAGE_VECTORSTORE_DIR_NAME)
        os.makedirs(global_image_vectorstore_path, exist_ok=True)
        logger.info(f"Ensured global image vector store directory exists: {global_image_vectorstore_path}")

        # --- Handle Force Rebuild (Clear Only) ---
        if force_rebuild:
            st.toast(f"Attempting to clear existing vector store(s) for {user_role} and images...", icon="🗑️")
            logger.info(f"Force rebuild initiated. Clearing vector stores for {user_role}.")
            
            # CRITICAL: Explicitly set session state objects to None to release references
            st.session_state.vector_store_instance = None
            st.session_state.bm25_retriever_instance = None
            st.session_state.hybrid_retriever_instance = None
            st.session_state.image_vector_store_instance = None
            
            # CRITICAL: Clear Streamlit's resource and data caches to release references
            # This is crucial for FAISS files to be deletable.
            st.cache_resource.clear()
            st.cache_data.clear()
            logger.info("Streamlit caches cleared to release file handles.")
            
            gc.collect() # Force garbage collection again after clearing caches
            time.sleep(0.5) # Give OS time to release file locks

            def safe_rmtree(path_to_remove):
                for i in range(5):
                    try:
                        if os.path.exists(path_to_remove):
                            logger.info(f"Removing directory: {path_to_remove}")
                            shutil.rmtree(path_to_remove)
                        logger.info(f"Successfully cleared {path_to_remove} on attempt {i+1}")
                        return True
                    except PermissionError as e:
                        logger.error(f"Attempt {i+1} to clear {path_to_remove} failed: Permission denied. Error: {e}", exc_info=True)
                        st.session_state.kb_update_status_message = f"Permission Denied: Cannot clear '{os.path.basename(path_to_remove)}'. Please check Docker volume permissions."
                        st.session_state.kb_update_status_type = "error"
                        return False
                    except OSError as e:
                        if e.errno == 16: # errno 16 is 'Device or resource busy'
                            logger.warning(f"Attempt {i+1} to clear {path_to_remove} failed: Device or resource busy. Retrying in 1 second...")
                            time.sleep(1)
                        else:
                            logger.error(f"Failed to clear {path_to_remove} due to unexpected OS error: {e}", exc_info=True)
                            raise
                logger.error(f"Failed to clear {path_to_remove} after multiple retries due to Device or resource busy. This might require a manual restart of the application.")
                st.session_state.kb_update_status_message = f"Failed to clear {os.path.basename(path_to_remove)}. Manual intervention might be needed."
                st.session_state.kb_update_status_type = "warning"
                return False

            text_kb_cleared = safe_rmtree(target_text_vectorstore_path)
            # Image KB is also tied to the role's vectorstore path
            image_kb_cleared = safe_rmtree(global_image_vectorstore_path)
            
            if not text_kb_cleared:
                st.session_state.kb_update_status_message = UI_TEXTS[st.session_state.ui_lang]["kb_update_failed"] + " (Failed to clear old text KB)"
                st.session_state.kb_update_status_type = "error"
                return
            
            if not image_kb_cleared:
                 st.session_state.kb_update_status_message = f"Failed to clear old image knowledge base. Image search might be inconsistent."
                 st.session_state.kb_update_status_type = "warning"
            
            # Recreate directories after clearing
            os.makedirs(target_text_vectorstore_path, exist_ok=True)
            os.makedirs(global_image_vectorstore_path, exist_ok=True)
            os.makedirs(PERSISTENT_EXTRACTED_IMAGES_DIR, exist_ok=True) # Ensure it exists after any potential manual deletion
            time.sleep(0.5)
            
            # Set status message for clearing and return, do not proceed with rebuild here
            st.session_state.kb_update_status_message = UI_TEXTS[st.session_state.ui_lang]["kb_cleared_instruction"]
            st.session_state.kb_update_status_type = "info"
            st.toast("Knowledge base cleared!", icon="✅")
            return # IMPORTANT: Exit after clearing for force_rebuild

        # --- Incremental Processing Logic (only if not force_rebuild) ---
        
        all_current_files_for_role = {}
        for doc_folder in document_folders_to_load:
            if not os.path.exists(doc_folder):
                logger.warning(f"Document source folder '{doc_folder}' does not exist. Creating it.")
                os.makedirs(doc_folder, exist_ok=True)
                continue
            for root, _, files in os.walk(doc_folder):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    file_hash = generate_file_hash(filepath)
                    last_modified = os.path.getmtime(filepath)
                    all_current_files_for_role[filepath] = {"hash": file_hash, "last_modified": last_modified}

        manifest_path = os.path.join(target_text_vectorstore_path, "processed_documents_manifest.json")
        processed_manifest = load_manifest(manifest_path)

        # Load existing vector stores *after* potential clearing
        existing_text_vector_store = load_vectorstore(target_text_vectorstore_path, embeddings)
        # Use the global `vision_embeddings_model` instance for image vector store
        existing_image_vector_store = load_vectorstore(global_image_vectorstore_path, vision_embeddings_model) 
        
        needs_full_rebuild_due_to_missing_index = (existing_text_vector_store is None and not os.path.exists(target_text_vectorstore_path + "/index.faiss"))

        new_or_modified_text_docs_for_vs = []
        new_or_modified_image_docs_for_vs = []

        files_to_process = {}
        if needs_full_rebuild_due_to_missing_index: # Only trigger full rebuild if index is missing
            logger.info("Full rebuild needed due to missing text index: processing all current files.")
            files_to_process = all_current_files_for_role
            processed_manifest = {} # Clear manifest for full rebuild
        else:
            for filepath, current_meta in all_current_files_for_role.items():
                # FIX 1: Use .get() to safely access keys in the manifest to prevent KeyError.
                # This checks if the file is new, or if its hash or modification time has changed.
                if filepath not in processed_manifest or \
                   processed_manifest.get(filepath, {}).get("hash") != current_meta.get("hash") or \
                   processed_manifest.get(filepath, {}).get("last_modified") != current_meta.get("last_modified"):
                    files_to_process[filepath] = current_meta
        
        if not files_to_process and needs_full_rebuild_due_to_missing_index:
            st.session_state.kb_update_status_message = UI_TEXTS[st.session_state.ui_lang]["no_docs_found"] + " Cannot build KB without documents."
            st.session_state.kb_update_status_type = "warning"
            logger.warning("No documents found to build KB, and full rebuild was needed.")
            return

        for filepath, current_meta in files_to_process.items():
            logger.info(f"Processing file for KB: {os.path.basename(filepath)}")
            st.toast(f"Processing: {os.path.basename(filepath)}", icon="⏳") 

            # Pass the new PERSISTENT_EXTRACTED_IMAGES_DIR to process_file
            process_file_result = process_file(filepath, PERSISTENT_EXTRACTED_IMAGES_DIR)
            text_docs_from_file = []
            extracted_image_paths = []

            if isinstance(process_file_result, tuple) and len(process_file_result) == 2:
                text_docs_from_file, extracted_image_paths = process_file_result
            elif isinstance(process_file_result, list):
                text_docs_from_file = process_file_result
                extracted_image_paths = []
                logger.warning(f"process_file returned only text documents for {filepath}. No images extracted.")
            else:
                logger.error(f"process_file returned unexpected result for {filepath}. Result: {process_file_result}")
                continue
            
            if text_docs_from_file:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunked_text_docs = text_splitter.split_documents(text_docs_from_file)
                new_or_modified_text_docs_for_vs.extend(chunked_text_docs)
                logger.info(f"Split {len(text_docs_from_file)} raw text documents into {len(chunked_text_docs)} chunks for {os.path.basename(filepath)}.")
            else:
                logger.warning(f"No text content found or extracted from {os.path.basename(filepath)}.")
            
            for img_path in extracted_image_paths:
                img_doc = Document(
                    page_content=f"Visually represented by image: {os.path.basename(img_path)}", # Updated page_content
                    metadata={
                        "source": filepath, # Original document source
                        "image_path": img_path, # This will now point to the persistent directory
                        "type": "extracted_image",
                        "uuid": str(uuid.uuid4()),
                        "timestamp": time.time()
                    }
                )
                new_or_modified_image_docs_for_vs.append(img_doc)
            
            # FIX 2: Removed a stray empty string that caused a SyntaxError.
            processed_manifest[filepath] = {
                "hash": current_meta["hash"],
                "last_modified": current_meta["last_modified"],
                "processed_at": time.time(),
                "extracted_images": extracted_image_paths
            }

        if not needs_full_rebuild_due_to_missing_index: # Only prune manifest if not a full rebuild
            files_to_delete_from_manifest = [
                filepath for filepath in processed_manifest if filepath not in all_current_files_for_role
            ]
            for filepath in files_to_delete_from_manifest:
                logger.info(f"File {os.path.basename(filepath)} deleted from source. Removing from manifest.")
                logger.warning(f"Manual removal of embeddings for deleted file {os.path.basename(filepath)} is not implemented. Consider full rebuild if many files are deleted.")
                del processed_manifest[filepath]

        if new_or_modified_text_docs_for_vs:
            st.toast(f"Adding {len(new_or_modified_text_docs_for_vs)} text documents to KB...", icon="✍️")
            st.session_state.vector_store_instance = add_documents_to_vectorstore(
                new_or_modified_text_docs_for_vs,
                target_text_vectorstore_path,
                embeddings, # Use the text embeddings model
                existing_text_vector_store
            )
        elif existing_text_vector_store:
            st.session_state.vector_store_instance = existing_text_vector_store
            logger.info("No new text documents to add, but existing text vector store was loaded.")
        else:
            st.session_state.vector_store_instance = None
            logger.info("No new text documents to add and no existing text vector store to load.")


        if new_or_modified_image_docs_for_vs:
            st.toast(f"Adding {len(new_or_modified_image_docs_for_vs)} image documents to image KB...", icon="🖼️")
            
            st.session_state.image_vector_store_instance = add_documents_to_vectorstore(
                new_or_modified_image_docs_for_vs,
                global_image_vectorstore_path,
                vision_embeddings_model, # Pass the global RealVisionEmbeddings instance
                existing_image_vector_store
            )
        elif existing_image_vector_store:
            st.session_state.image_vector_store_instance = existing_image_vector_store
            logger.info("No new image documents to add, but existing image vector store was loaded.")
        else:
            st.session_state.image_vector_store_instance = None
            logger.info("No new image documents to add and no existing image vector store to load.")

        save_manifest(processed_manifest, manifest_path)

        current_faiss_chunks = 0
        if st.session_state.vector_store_instance:
            current_faiss_chunks = st.session_state.vector_store_instance.index.ntotal if st.session_state.vector_store_instance.index else 0
            logger.info(f"Final FAISS store for {user_role} has {current_faiss_chunks} chunks.")
            st.session_state[f"{user_role}_faiss_ready"] = True
            st.session_state[f"{user_role}_faiss_total_chunks"] = current_faiss_chunks
        else:
            st.session_state[f"{user_role}_faiss_ready"] = False
            st.session_state[f"{user_role}_faiss_total_chunks"] = 0

        image_kb_entries = 0
        if st.session_state.image_vector_store_instance:
            # Check if docstore exists and is not empty to count entries.
            # FAISS.index.ntotal is for the main vector, not necessarily number of documents.
            # The docstore is a better indicator of actual stored documents.
            if hasattr(st.session_state.image_vector_store_instance, 'docstore') and \
               st.session_state.image_vector_store_instance.docstore is not None and \
               hasattr(st.session_state.image_vector_store_instance.docstore, '_dict'):
                image_kb_entries = len(st.session_state.image_vector_store_instance.docstore._dict)
            else:
                # Fallback for older FAISS versions or if docstore is not directly accessible
                # This might not be accurate for number of *documents* but gives an idea of index size
                image_kb_entries = st.session_state.image_vector_store_instance.index.ntotal if st.session_state.image_vector_store_instance.index else 0
            
            st.session_state[f"{user_role}_image_kb_ready"] = True
            st.session_state[f"{user_role}_image_kb_total_entries"] = image_kb_entries
            logger.info(f"Image KB for {user_role} is ready with {image_kb_entries} entries.")
        else:
            st.session_state[f"{user_role}_image_kb_ready"] = False
            st.session_state[f"{user_role}_image_kb_total_entries"] = 0
            logger.info(f"Image KB for {user_role} is NOT ready (0 entries).")

        unique_docs_for_bm25 = []
        if st.session_state.vector_store_instance:
            seen_uuids = set()
            for doc_uuid, doc in st.session_state.vector_store_instance.docstore._dict.items():
                if doc_uuid not in seen_uuids:
                    unique_docs_for_bm25.append(doc)
                    seen_uuids.add(doc_uuid)
            
            if unique_docs_for_bm25:
                st.toast(UI_TEXTS[st.session_state.ui_lang]["building_bm25"], icon="🔍")
                st.session_state.bm25_retriever_instance = BM25Retriever.from_documents(unique_docs_for_bm25)
                st.session_state.bm25_retriever_instance.k = 5 # RESTORED: Increased k value
                logger.info(f"BM25 retriever built with {len(unique_docs_for_bm25)} documents.")
                st.session_state[f"{user_role}_bm25_ready"] = True
            else:
                logger.warning(f"No documents available to build BM25 retriever for {user_role}.")
                st.session_state.bm25_retriever_instance = None
                st.session_state[f"{user_role}_bm25_ready"] = False
        else:
            st.session_state.bm25_retriever_instance = None
            st.session_state[f"{user_role}_bm25_ready"] = False


        if st.session_state.vector_store_instance and st.session_state.bm25_retriever_instance:
            st.session_state.hybrid_retriever_instance = EnsembleRetriever(
                retrievers=[st.session_state.bm25_retriever_instance, st.session_state.vector_store_instance.as_retriever(search_kwargs={"k": 5})], # RESTORED: Increased k value
                weights=[0.4, 0.6] # RESTORED: Adjusted weights
            )
            st.session_state.kb_update_status_message = UI_TEXTS[st.session_state.ui_lang]["kb_updated_success"]
            st.session_state.kb_update_status_type = "success"
            logger.info(f"Hybrid retriever successfully initialized for {user_role}.")
        elif st.session_state.vector_store_instance: # Only FAISS is available
            st.session_state.hybrid_retriever_instance = st.session_state.vector_store_instance.as_retriever(search_kwargs={"k": 5}) # RESTORED: Increased k value
            st.session_state.kb_update_status_message = f"Only FAISS retriever available for {user_role}. Hybrid not possible."
            st.session_state.kb_update_status_type = "warning"
            logger.warning(f"Only FAISS retriever available for {user_role}. BM25 could not be built.")
        else:
            st.session_state.hybrid_retriever_instance = None
            st.session_state.kb_update_status_message = UI_TEXTS[st.session_state.ui_lang]["kb_not_loaded"] + " (No text vector store could be built)."
            st.session_state.kb_update_status_type = "warning"
            logger.error(f"No text vector store could be built or loaded for {user_role}. KB is not ready.")

    except Exception as e:
        st.session_state.kb_update_status_message = f"{UI_TEXTS[st.session_state.ui_lang]['error_processing_docs']} {e}"
        st.session_state.kb_update_status_type = "error"
        logger.error(f"Error building/loading KB for role {user_role}: {e}", exc_info=True)
        st.session_state.vector_store_instance = None
        st.session_state.bm25_retriever_instance = None
        st.session_state.hybrid_retriever_instance = None
        st.session_state.image_vector_store_instance = None
        st.session_state[f"{user_role}_faiss_ready"] = False
        st.session_state[f"{user_role}_faiss_total_chunks"] = 0
        st.session_state[f"{user_role}_bm25_ready"] = False
        st.session_state[f"{user_role}_image_kb_ready"] = False
        st.session_state[f"{user_role}_image_kb_total_entries"] = 0
    

def build_or_load_knowledge_base_for_role(user_role: str, uploaded_files_session: Optional[List] = None, access_level_for_new_uploads: Optional[str] = None):
    """
    Main entry point for building/loading KB. Handles saving new uploads then triggers internal update.
    """
    if uploaded_files_session:
        logger.info(f"Processing {len(uploaded_files_session)} newly uploaded files with access level: {access_level_for_new_uploads}")
        
        for uploaded_file in uploaded_files_session:
            if user_role == "professor":
                target_dir_for_save = PROF_DOCS_PATH if access_level_for_new_uploads == "professor_only" else SHARED_DOCS_PATH
            elif user_role == "student":
                target_dir_for_save = SHARED_DOCS_PATH
            else:
                target_dir_for_save = SHARED_DOCS_PATH
            
            file_save_path = os.path.join(target_dir_for_save, uploaded_file.name)
            os.makedirs(os.path.dirname(file_save_path), exist_ok=True)
            with open(file_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"Saved uploaded file: {file_save_path}")
        st.session_state.uploaded_files_in_session = []
        st.success(UI_TEXTS[st.session_state.ui_lang]["uploaded_docs_processed"])

    _trigger_kb_update(st.session_state.user_role) # No force_rebuild here, it's for incremental updates


# === Session State Initialization ===
if 'ui_lang' not in st.session_state:
    st.session_state.ui_lang = DEFAULT_UI_LANG

if "messages_en" not in st.session_state:
    st.session_state.messages_en = [{"role": "assistant", "content": UI_TEXTS["en"]["initial_greeting"]}]
if "messages_de" not in st.session_state: # Corrected this line
    st.session_state.messages_de = [{"role": "assistant", "content": UI_TEXTS["de"]["initial_greeting"]}]

if 'memory_en' not in st.session_state:
    st.session_state.memory_en = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, input_key="question", output_key="answer"
    )
if 'memory_de' not in st.session_state:
    st.session_state.memory_de = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, input_key="question", output_key="answer"
    )

if 'vector_store_instance' not in st.session_state:
    st.session_state.vector_store_instance = None
if 'bm25_retriever_instance' not in st.session_state:
    st.session_state.bm25_retriever_instance = None
if 'hybrid_retriever_instance' not in st.session_state:
    st.session_state.hybrid_retriever_instance = None
if 'uploaded_files_in_session' not in st.session_state:
    st.session_state.uploaded_files_in_session = []
if 'image_vector_store_instance' not in st.session_state:
    st.session_state.image_vector_store_instance = None
if 'image_kb_total_entries' not in st.session_state: # New state for image KB entry count
    st.session_state.image_kb_total_entries = 0
# Removed 'uploaded_chat_image_file' from session state
if 'uploaded_chat_image_file_obj' not in st.session_state:
    st.session_state.uploaded_chat_image_file_obj = None
if 'last_processed_chat_image_path' not in st.session_state:
    st.session_state.last_processed_chat_image_path = None

if 'professor_faiss_ready' not in st.session_state:
    st.session_state.professor_faiss_ready = False
    st.session_state.professor_faiss_total_chunks = 0
    st.session_state.professor_bm25_ready = False
if 'professor_image_kb_ready' not in st.session_state:
    st.session_state.professor_image_kb_ready = False
    st.session_state.professor_image_kb_total_entries = 0

if 'student_faiss_ready' not in st.session_state:
    st.session_state.student_faiss_ready = False
    st.session_state.student_faiss_total_chunks = 0
    st.session_state.student_bm25_ready = False
if 'student_image_kb_ready' not in st.session_state:
    st.session_state.student_image_kb_ready = False
    st.session_state.student_image_kb_total_entries = 0

if 'kb_update_status_message' not in st.session_state:
    st.session_state.kb_update_status_message = None
if 'kb_update_status_type' not in st.session_state:
    st.session_state.kb_update_status_type = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.user_role = None
    st.session_state.user_id = None
    
    
# --- App Title and Caption ---
st.title(UI_TEXTS[st.session_state.ui_lang]["title"])
st.caption(UI_TEXTS[st.session_state.ui_lang]["caption"])

# --- Login Section ---
if not st.session_state.logged_in:
    st.markdown(f"## {UI_TEXTS[st.session_state.ui_lang]['login_header']}")
    
    email = st.text_input(UI_TEXTS[st.session_state.ui_lang]['email_label'], key="login_email")
    password = st.text_input(UI_TEXTS[st.session_state.ui_lang]['password_label'], type="password", key="login_password")

    if st.button(UI_TEXTS[st.session_state.ui_lang]['login_button'], key="do_login"):
        user_data = authenticate_user(email, password)
        if user_data:
            st.session_state.logged_in = True
            st.session_state.user_email = user_data["email"]
            st.session_state.user_role = user_data["role"]
            st.session_state.user_id = user_data["email"]
            
            st.success(UI_TEXTS[st.session_state.ui_lang]['logged_in_as'].format(email=st.session_state.user_email, role=st.session_state.user_role))
            
            # Trigger initial KB load/build for the logged-in user's role
            _trigger_kb_update(st.session_state.user_role)
            st.rerun()
        else:
            st.error(UI_TEXTS[st.session_state.ui_lang]['login_error'])
            logger.warning(f"Login attempt failed for email: {email}")
    
    st.info(UI_TEXTS[st.session_state.ui_lang]["auth_welcome_message"])


else:
    current_lang = st.session_state.ui_lang
    current_messages = st.session_state[f"messages_{current_lang}"]
    current_memory = st.session_state[f"memory_{current_lang}"]

    st.sidebar.markdown(f"**{UI_TEXTS[st.session_state.ui_lang]['logged_in_as'].format(email=st.session_state.user_email, role=st.session_state.user_role)}**")
    if st.sidebar.button(UI_TEXTS[st.session_state.ui_lang]['logout_button'], key="do_logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.session_state.messages_en = [{"role": "assistant", "content": UI_TEXTS["en"]["initial_greeting"]}]
        st.session_state.messages_de = [{"role": "assistant", "content": UI_TEXTS["de"]["initial_greeting"]}]
        st.session_state.memory_en.clear()
        st.session_state.memory_de.clear()
        st.session_state.vector_store_instance = None
        st.session_state.bm25_retriever_instance = None
        st.session_state.hybrid_retriever_instance = None
        st.session_state.image_vector_store_instance = None
        st.session_state.professor_faiss_ready = False
        st.session_state.professor_faiss_total_chunks = 0
        st.session_state.professor_bm25_ready = False
        st.session_state.professor_image_kb_ready = False
        st.session_state.professor_image_kb_total_entries = 0
        st.session_state.student_faiss_ready = False
        st.session_state.student_faiss_total_chunks = 0
        st.session_state.student_bm25_ready = False
        st.session_state.student_image_kb_ready = False
        st.session_state.student_image_kb_total_entries = 0
        st.session_state.kb_update_status_message = None
        st.session_state.kb_update_status_type = None
        # Clear staged image and delete file on logout
        if st.session_state.last_processed_chat_image_path and os.path.exists(st.session_state.last_processed_chat_image_path):
            os.remove(st.session_state.last_processed_chat_image_path)
            logger.info(f"Cleaned up temporary staged image on logout: {st.session_state.last_processed_chat_image_path}")
        st.session_state.uploaded_chat_image_file_obj = None
        st.session_state.last_processed_chat_image_path = None

        st.rerun()

    # --- Sidebar Elements ---
    with st.sidebar:
        st.selectbox(
            "Select Language",
            options=["en", "de"],
            format_func=lambda x: "English" if x == "en" else "Deutsch",
            key="ui_lang",
        )

        st.markdown("---")

        # --- Document Management Section ---
        st.markdown(f"### {UI_TEXTS[st.session_state.ui_lang]['doc_mgmt_header']}")
        st.info(UI_TEXTS[st.session_state.ui_lang]['doc_mgmt_info'])

        uploaded_files = st.file_uploader(
            UI_TEXTS[st.session_state.ui_lang]["upload_files_label"],
            type=["pdf", "docx", "png", "jpg", "jpeg", "tiff"],
            accept_multiple_files=True,
            key="doc_uploader",
        )
        
        if uploaded_files:
            st.session_state.uploaded_files_in_session = uploaded_files

        doc_access_level = None
        if st.session_state.user_role == "professor":
            doc_access_level = st.radio(
                UI_TEXTS[st.session_state.ui_lang]["upload_doc_access"],
                (UI_TEXTS[st.session_state.ui_lang]["access_level_prof_only"], UI_TEXTS[st.session_state.ui_lang]["access_level_shared"]),
                key="doc_access_radio",
            )
        else:
            doc_access_level = UI_TEXTS[st.session_state.ui_lang]["access_level_shared"]
            st.info(f"Student uploads automatically go to: **{doc_access_level}**")

        if st.button(UI_TEXTS[st.session_state.ui_lang]["process_button"], key="process_docs_button"):
            if doc_access_level:
                internal_access_level = "professor_only" if doc_access_level == UI_TEXTS[st.session_state.ui_lang]["access_level_prof_only"] else "shared"
                
                with st.spinner(UI_TEXTS[st.session_state.ui_lang]["processing_docs_selected_level"].format(level=doc_access_level)):
                    build_or_load_knowledge_base_for_role(
                        st.session_state.user_role,
                        st.session_state.uploaded_files_in_session,
                        internal_access_level
                    )
                st.rerun()
            else:
                st.warning("Please select a document access level.")
        
        if st.session_state.kb_update_status_message:
            if st.session_state.kb_update_status_type == "success":
                st.success(st.session_state.kb_update_status_message)
            elif st.session_state.kb_update_status_type == "warning":
                st.warning(st.session_state.kb_update_status_message)
            elif st.session_state.kb_update_status_type == "error":
                st.error(st.session_state.kb_update_status_message)
            elif st.session_state.kb_update_status_type == "info": # New info type for cleared KB
                st.info(st.session_state.kb_update_status_message)
            st.session_state.kb_update_status_message = None
            st.session_state.kb_update_status_type = None

        st.markdown("---")

        # --- Knowledge Base Status Section ---
        st.markdown(f"### {UI_TEXTS[st.session_state.ui_lang]['kb_status_header']}")

        current_role_faiss_ready = st.session_state.get(f"{st.session_state.user_role}_faiss_ready", False)
        current_role_faiss_chunks = st.session_state.get(f"{st.session_state.user_role}_faiss_total_chunks", 0)
        current_role_bm25_ready = st.session_state.get(f"{st.session_state.user_role}_bm25_ready", False)
        current_role_image_kb_ready = st.session_state.get(f"{st.session_state.user_role}_image_kb_ready", False)
        current_role_image_kb_entries = st.session_state.get(f"{st.session_state.user_role}_image_kb_total_entries", 0)


        if current_role_faiss_ready:
            st.success(UI_TEXTS[st.session_state.ui_lang]["kb_loaded_ready"])
            st.write(f"- {UI_TEXTS[st.session_state.ui_lang]['faiss_entries']} {current_role_faiss_chunks}")
            if current_role_bm25_ready:
                st.write(f"- {UI_TEXTS[st.session_state.ui_lang]['bm25_ready']}")
            else:
                st.warning(f"- BM25 Retriever Not Loaded. Only FAISS will be used for search.")
        else:
            st.warning(UI_TEXTS[st.session_state.ui_lang]["kb_not_loaded"])
        
        if current_role_image_kb_ready:
            st.success(UI_TEXTS[st.session_state.ui_lang]["image_kb_ready"])
            st.write(f"- Image KB entries: {current_role_image_kb_entries}")
            if current_role_image_kb_entries == 0:
                st.info(UI_TEXTS[st.session_state.ui_lang]["image_kb_empty_hint"])
        else:
            st.warning(UI_TEXTS[st.session_state.ui_lang]["image_kb_not_loaded"])
            st.info(UI_TEXTS[st.session_state.ui_lang]["image_kb_empty_hint"])


        # Changed button: "Clear Knowledge Base"
        if st.button(UI_TEXTS[st.session_state.ui_lang]["clear_kb_button"], key="clear_kb_button"):
            with st.spinner("Clearing knowledge base..."):
                _trigger_kb_update(st.session_state.user_role, force_rebuild=True) # This now only clears
            st.rerun()

        st.markdown("---")

        # --- Clear Chat History Button ---
        if st.button(UI_TEXTS[st.session_state.ui_lang]["clear_chat_button"], key="clear_chat"):
            st.session_state[f"messages_{current_lang}"] = [{"role": "assistant", "content": UI_TEXTS[current_lang]["initial_greeting"]}]
            current_memory.clear()
            # Clear staged image and delete file on chat clear
            if st.session_state.last_processed_chat_image_path and os.path.exists(st.session_state.last_processed_chat_image_path):
                os.remove(st.session_state.last_processed_chat_image_path)
                logger.info(f"Cleaned up temporary staged image on chat clear: {st.session_state.last_processed_chat_image_path}")
            st.session_state.uploaded_chat_image_file_obj = None
            st.session_state.last_processed_chat_image_path = None
            st.rerun()

        st.markdown("---")

        # --- Ollama Connection Status ---
        st.markdown(f"### {UI_TEXTS[st.session_state.ui_lang]['ollama_status_header']}")
        try:
            ollama_status = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if ollama_status.status_code == 200:
                st.success(UI_TEXTS[st.session_state.ui_lang]["ollama_connected"])
            else:
                st.error(UI_TEXTS[st.session_state.ui_lang]["ollama_disconnected"])
        except requests.exceptions.ConnectionError:
            st.error(UI_TEXTS[st.session_state.ui_lang]["ollama_disconnected"])
        except Exception as e:
            st.error(f"Ollama Status Error: {e}")

        st.markdown("---")

        # --- About Section ---
        st.markdown(f"### {UI_TEXTS[st.session_state.ui_lang]['about_header']}")
        st.info(UI_TEXTS[st.session_state.ui_lang]['about_content'])


    # Display chat messages from history
    chat_container = st.container()
    with chat_container:
        for message in current_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander(UI_TEXTS[st.session_state.ui_lang]["sources_expander"]):
                        for source in message["sources"]:
                            st.markdown(source)


    # Input elements for user query and image upload
    # Image uploader directly above the chat input
    uploaded_chat_image = st.file_uploader(
        UI_TEXTS[st.session_state.ui_lang]["image_chat_upload_label"],
        type=["png", "jpg", "jpeg", "tiff"],
        key="chat_image_uploader",
        disabled=not st.session_state.logged_in
    )

    # Handle initial image upload: save to temp and acknowledge
    if uploaded_chat_image and st.session_state.uploaded_chat_image_file_obj != uploaded_chat_image:
        # If a previous image was staged, clean it up before saving the new one
        if st.session_state.last_processed_chat_image_path and os.path.exists(st.session_state.last_processed_chat_image_path):
            os.remove(st.session_state.last_processed_chat_image_path)
            logger.info(f"Cleaned up previous temporary staged image: {st.session_state.last_processed_chat_image_path}")

        # Save new uploaded chat image to temporary directory
        os.makedirs(TEMP_CHAT_IMAGES_DIR, exist_ok=True)
        temp_image_path = os.path.join(TEMP_CHAT_IMAGES_DIR, uploaded_chat_image.name)
        
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_chat_image.getbuffer())
        logger.info(f"Saved uploaded chat image temporarily to: {temp_image_path}")

        st.session_state.last_processed_chat_image_path = temp_image_path
        st.session_state.uploaded_chat_image_file_obj = uploaded_chat_image # Store the object to detect changes

        # Add an assistant message to chat history to reflect the staging
        assistant_message_content = UI_TEXTS[st.session_state.ui_lang]["image_staged_prompt"].format(filename=uploaded_chat_image.name)
        current_messages.append({"role": "assistant", "content": assistant_message_content})
        
        st.rerun() # Rerun to clear the file uploader and update UI

    # Display staged image info if an image is already staged from a previous run
    if st.session_state.last_processed_chat_image_path and not uploaded_chat_image:
        st.info(UI_TEXTS[st.session_state.ui_lang]["image_staged_for_upload"].format(filename=os.path.basename(st.session_state.last_processed_chat_image_path)))

    user_query = st.chat_input(UI_TEXTS[st.session_state.ui_lang]["chat_input_placeholder"], key="user_chat_input", disabled=not st.session_state.logged_in)
    logger.info(f"Streamlit chat_input value on rerun: '{user_query}'") # ADDED DEBUG LOG HERE

    @st.cache_resource(hash_funcs={ChatOllama: lambda _: None, FAISS: lambda _: None, EnsembleRetriever: lambda _: None, BM25Retriever: lambda _: None, ConversationBufferMemory: lambda _: None, ContextualCompressionRetriever: lambda _: None})
    def get_langchain_components(_llm_instance, hybrid_retriever_instance, _memory_instance, _compressor_instance):
        
        # MODIFIED: Setup the Contextual Compression Retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_compressor_instance,
            base_retriever=hybrid_retriever_instance
        )

        qa_chain_instance = ConversationalRetrievalChain.from_llm(
            llm=_llm_instance,
            retriever=compression_retriever, # MODIFIED: Use the compression retriever
            memory=_memory_instance,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True
        )

        summarize_chain_instance = load_summarize_chain(_llm_instance, chain_type="stuff", prompt=summarization_prompt)
        metadata_extraction_chain_instance = metadata_extraction_prompt | _llm_instance | metadata_parser

        simple_question_llm_chain_instance = PromptTemplate(
            input_variables=["question", "chat_history"], # Changed 'input' to 'question'
            template="""You are a helpful and friendly AI assistant from Aalen University.
            
            Based on the user's input, respond appropriately:
            - If the user says "Hi", "Hello", or similar greetings, respond with a friendly greeting and identify yourself as the IMFAA bot.
            - If the user asks "How are you?", respond politely and positively (e.g., "I'm doing well, thank you for asking!").
            - If the user says "Thank you" or expresses gratitude, respond with a polite acknowledgment (e.g., "You're welcome!", "Glad I could help!").
            - For any other general question related to documents (e.g., "what is the current gap in research at IMFAA", "How AI could be used for Materials?", "What is the Material"), provide a concise and helpful answer based on any information based on the docment give your suggestion.
            - If you don't know the answer to a general question, politely state that you don't have that information.
            - if the user is asking anything non document question, Politely give reply in very short sentance

            Respond in the same language as the user's question.

            Chat History:
            {chat_history}
            Question: {question}
            """
        ) | _llm_instance | (lambda x: x.content)

        # === Tool Implementations ===
        def summarize_document_tool(document_query: str) -> str:
            """
            Summarizes content of a document found in the knowledge base based on a query.
            Input must be a query string (e.g., "Agent S2 paper" or "report on AI").
            """
            if not hybrid_retriever_instance:
                logger.error("Summarize tool: Hybrid retriever not initialized.")
                return "Error: The knowledge base is not loaded. Please process documents first."
            
            with st.spinner("Retrieving document content for summarization..."):
                # Use the hybrid retriever to find relevant document chunks
                retrieved_docs = hybrid_retriever_instance.get_relevant_documents(document_query)
            
            if not retrieved_docs:
                logger.warning(f"Summarize tool: No relevant documents found for query: '{document_query}'")
                return f"Error: No relevant documents found for '{document_query}' in the knowledge base. Please try a different query."
            
            # Concatenate content from retrieved documents
            full_document_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Limit the total content to avoid exceeding LLM context window for summarization
            summarization_content = full_document_content[:30000] 

            try:
                with st.spinner(UI_TEXTS[st.session_state.ui_lang]["thinking_status"]):
                    summary = summarize_chain_instance.invoke({"input_documents": [Document(page_content=summarization_content)]})['output_text']
                return summary
            except Exception as e:
                logger.error(f"Error during summarization of document content for query '{document_query}': {e}", exc_info=True)
                return f"Error during summarization of the document content: {e}"

        def extract_paper_metadata_tool(document_query: str) -> str:
            """
            Extracts title and authors from a document found in the knowledge base based on a query.
            Input must be a query string (e.g., "Agent S2 paper" or "report on AI").
            """
            if not hybrid_retriever_instance:
                logger.error("Metadata tool: Hybrid retriever not initialized.")
                return "Error: The knowledge base is not loaded. Please process documents first."

            with st.spinner("Retrieving document content for metadata extraction..."):
                retrieved_docs = hybrid_retriever_instance.get_relevant_documents(document_query)
            
            if not retrieved_docs:
                logger.warning(f"Metadata tool: No relevant documents found for query: '{document_query}'")
                return f"Error: No relevant documents found for '{document_query}' in the knowledge base. Please try a different query."

            # MODIFIED: Use only the content of the *first* retrieved document for more focused extraction
            metadata_content = retrieved_docs[0].page_content

            try:
                with st.spinner(UI_TEXTS[st.session_state.ui_lang]["thinking_status"]):
                    logger.info(f"Attempting metadata extraction on content snippet (first 500 chars): {metadata_content[:500]}...")
                    
                    raw_llm_output = (metadata_extraction_prompt | _llm_instance).invoke({"abstract": metadata_content})
                    
                    logger.info(f"Raw LLM output for metadata extraction: {raw_llm_output.content}")
                    
                    metadata_obj = metadata_parser.parse(raw_llm_output.content)
                return metadata_obj.model_dump_json(indent=2) + f"\n\n{UI_TEXTS[st.session_state.ui_lang]['metadata_extraction_error_hint']}"
            except Exception as e:
                logger.error(f"Error extracting metadata from document content for query '{document_query}': {e}", exc_info=True)
                return f"Error extracting metadata from the document content: {e}. Raw LLM output might be malformed. Check logs for details. {UI_TEXTS[st.session_state.ui_lang]['metadata_extraction_error_hint']}"

        # === Define Tools ===
        tools_instance = [
            Tool(
                name="summarize_document", # Renamed for clarity
                func=summarize_document_tool,
                description="Summarize content of a document found in the knowledge base. Input must be a query string (e.g., 'Agent S2 paper' or 'report on AI')."
            ),
            Tool(
                name="extract_paper_metadata", # Renamed for clarity
                func=extract_paper_metadata_tool,
                description="Extract title and authors from a document found in the knowledge base. Input must be a query string (e.g., 'Agent S2 paper' or 'report on AI')."
            ),
        ]
        
        # === Initialize Agent (Updated Method) ===
        # Pull the ReAct prompt from the hub
        prompt = hub.pull("hwchase17/react-chat")
        
        # Create the ReAct agent
        agent = create_react_agent(llm, tools_instance, prompt)

        # Create the agent executor
        agent_executor_instance = AgentExecutor(
            agent=agent, 
            tools=tools_instance, 
            verbose=True,
            handle_parsing_errors=True
        )

        return qa_chain_instance, summarize_chain_instance, metadata_extraction_chain_instance, simple_question_llm_chain_instance, agent_executor_instance

    qa_chain, summarize_chain, metadata_extraction_chain, simple_question_llm_chain, agent_executor = (None, None, None, None, None)
    if st.session_state.get('hybrid_retriever_instance'):
        qa_chain, summarize_chain, metadata_extraction_chain, simple_question_llm_chain, agent_executor = get_langchain_components(
            llm, st.session_state.hybrid_retriever_instance, current_memory, compressor
        )
        
    if user_query:
        user_content_display = user_query
        if st.session_state.last_processed_chat_image_path:
            user_content_display += (f"\n\n**[Referring to uploaded image: {os.path.basename(st.session_state.last_processed_chat_image_path)}]** ")
        
        current_messages.append({"role": "user", "content": user_content_display})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_content_display, unsafe_allow_html=True)
    
            with st.chat_message("assistant"):
                response_content = ""
                should_stream = True
                response_placeholder = st.empty()
                
                similar_images_found = []
                sources_to_display = []
        
                with st.spinner(UI_TEXTS[st.session_state.ui_lang]["thinking_status"]):
                    try:
                        staged_image_path = st.session_state.last_processed_chat_image_path
            
                        if staged_image_path:
                            query_type = classify_query(user_query)
                            logger.info(f"Classified query '{user_query}' with staged image as: {query_type}")
            
                            if query_type == "IMAGE_QUERY":
                                logger.info(f"Processing IMAGE_QUERY for staged image: {staged_image_path}")
                                if vision_embeddings_model and st.session_state.image_vector_store_instance:
                                    with st.spinner("Generating image embedding and searching for similar images..."):
                                        try:
                                            if not os.path.exists(staged_image_path):
                                                logger.error(f"Staged image file not found for embedding: {staged_image_path}. Skipping embedding.")
                                                response_content += UI_TEXTS[st.session_state.ui_lang]["no_similar_images"] + " (Staged image file not found)\n\n"
                                                st.session_state.last_processed_chat_image_path = None
                                                st.session_state.uploaded_chat_image_file_obj = None
                                            else:
                                                uploaded_image_embedding_vector = vision_embeddings_model.embed_image(staged_image_path)
                                                if uploaded_image_embedding_vector is not None:
                                                    logger.info(f"Generated embedding for staged image. Vector length: {len(uploaded_image_embedding_vector)}")
                                                    similar_image_docs = st.session_state.image_vector_store_instance.similarity_search_by_vector(uploaded_image_embedding_vector, k=3)
                                                    
                                                    unique_similar_images = set()
                                                    for doc in similar_image_docs:
                                                        img_path_from_doc = doc.metadata.get("image_path")
                                                        if img_path_from_doc and os.path.exists(img_path_from_doc):
                                                            unique_similar_images.add(img_path_from_doc)
                                                            logger.info(f"Found similar image doc: {img_path_from_doc}")
                                                        else:
                                                            logger.warning(f"Similar image document found with invalid/missing 'image_path' metadata: {doc.metadata.get('image_path', 'N/A')}")
                                                    similar_images_found = list(unique_similar_images)
            
                                                    if similar_images_found:
                                                        logger.info(f"Found {len(similar_images_found)} unique visually similar images for display.")
                                                        response_content += UI_TEXTS[st.session_state.ui_lang]["similar_images_found"] + "\n"
                                                    else:
                                                        logger.info("No visually similar images found in the knowledge base.")
                                                        response_content += UI_TEXTS[st.session_state.ui_lang]["no_similar_images"] + "\n\n"
                                                else:
                                                    response_content += UI_TEXTS[st.session_state.ui_lang]["no_similar_images"] + " (Could not generate embedding for staged image)\n\n"
                                                    logger.warning("Failed to generate embedding for staged image for similarity search (embedding was None).")
                                        except Exception as img_embed_e:
                                            logger.error(f"Error during image embedding or similarity search for staged image: {img_embed_e}", exc_info=True)
                                            response_content += f"Error performing image similarity search: {img_embed_e}\n\n"
                                else:
                                    response_content += UI_TEXTS[st.session_state.ui_lang]["no_similar_images"] + " (Image KB not initialized or vision model not loaded)\n\n"
                                    logger.warning("Image vector store not initialized or vision embedding model not loaded for similarity search with staged image.")
                                
                                if os.path.exists(staged_image_path):
                                    os.remove(staged_image_path)
                                    logger.info(f"Cleaned up temporary staged image: {staged_image_path}")
                                st.session_state.last_processed_chat_image_path = None
                                st.session_state.uploaded_chat_image_file_obj = None
            
                            elif query_type == "EXPLAIN_IMAGE":
                                logger.info(f"Processing EXPLAIN_IMAGE query for staged image: {staged_image_path}")
                                try:
                                    language_name = "German" if current_lang == 'de' else "English"
                                    
                                    with st.spinner("Analyzing image and generating explanation..."):
                                        if not os.path.exists(staged_image_path):
                                            logger.error(f"Staged image file not found for explanation: {staged_image_path}. Cannot explain.")
                                            response_content = f"Error: Staged image file not found at {staged_image_path}. Please re-upload the image."
                                        else:
                                            with open(staged_image_path, "rb") as f:
                                                image_bytes = f.read()
                                            
                                            mime_type, _ = mimetypes.guess_type(staged_image_path)
                                            if not mime_type:
                                                mime_type = "image/jpeg"
                                            
                                            base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                                            logger.info(f"Image {os.path.basename(staged_image_path)} loaded and base64 encoded. Size: {len(base64_encoded_image)} bytes.")
                                            
                                            human_message_content = [
                                                {"type": "text", "text": user_query},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:{mime_type};base64,{base64_encoded_image}"
                                                    }
                                                },
                                            ]
                                            logger.info(f"Sending multimodal message to LLM for explanation. Text: '{user_query[:50]}...', Image: {os.path.basename(staged_image_path)}")

                                            explanation_response = llm.invoke([HumanMessage(content=human_message_content)])
                                            image_explanation_text = explanation_response.content
                                            response_content = image_explanation_text
                                            logger.info(f"LLM explanation for image: '{image_explanation_text[:100]}...'")

                                    if st.session_state.hybrid_retriever_instance and qa_chain:
                                        with st.spinner("Searching knowledge base for related information..."):
                                            # MODIFIED: Create a more contextual RAG query
                                            rag_query_from_explanation = f"Based on the user's query '{user_query}' and an image showing '{image_explanation_text[:150]}', what relevant information is in the documents?"
                                            logger.info(f"Performing RAG with improved query derived from image explanation: '{rag_query_from_explanation}'")

                                            qa_response = qa_chain.invoke({
                                                "question": rag_query_from_explanation,
                                                "language": language_name
                                            })
                                            retrieved_answer = qa_response.get('answer', '').strip()
                                            
                                            if retrieved_answer and not ("NO_ANSWER_IN_CONTEXT" in retrieved_answer.upper() or len(retrieved_answer) < 50 or "I don't know" in retrieved_answer.lower()):
                                                response_content += f"\n\n**{UI_TEXTS[st.session_state.ui_lang]['image_query_rag_response_intro']}**\n"
                                                response_content += retrieved_answer
                                                if qa_response.get('source_documents'):
                                                    logger.info(f"Retrieved {len(qa_response['source_documents'])} documents for RAG part of image explanation.")
                                                    for doc in qa_response['source_documents']:
                                                        source_path = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                                                        sources_to_display.append(f"{source_path}")
                                                        logger.info(f"Image RAG Document Source: {source_path}")
                                            else:
                                                logger.info("RAG inconclusive for image explanation, no additional document info appended.")
                                    else:
                                        logger.warning("Hybrid retriever or QA chain not initialized for RAG part of image explanation.")

                                except Exception as explain_e:
                                    logger.error(f"Error explaining image or performing RAG: {explain_e}", exc_info=True)
                                    response_content = f"Error explaining the image: {explain_e}"
                                finally:
                                    if os.path.exists(staged_image_path):
                                        os.remove(staged_image_path)
                                        logger.info(f"Cleaned up temporary staged image: {staged_image_path}")
                                    st.session_state.last_processed_chat_image_path = None
                                    st.session_state.uploaded_chat_image_file_obj = None
            
                            else: # Non-image query with a staged image
                                response_content += UI_TEXTS[st.session_state.ui_lang]["image_cleared_warning"] + "\n\n"
                                if os.path.exists(staged_image_path):
                                    os.remove(staged_image_path)
                                    logger.info(f"Cleaned up temporary staged image: {staged_image_path}")
                                st.session_state.last_processed_chat_image_path = None
                                st.session_state.uploaded_chat_image_file_obj = None
                                # Fall through to process the text query
                                query_type = classify_query(user_query) # Re-classify without image context
                                # Fallthrough to text-only processing
                                if "RAG_QUESTION" in query_type: # Add fallthrough logic here
                                    # This block is now reachable from the image handling section
                                    if not st.session_state.hybrid_retriever_instance or qa_chain is None:
                                        response_content += UI_TEXTS[st.session_state.ui_lang]["rag_no_kb"]
                                        st.warning(response_content)
                                        should_stream = False
                                    else:
                                        # ... (RAG logic as below)
                                        pass # Let it flow to the main text processing block
                                else: # Let other query types fall through
                                    pass


                        # This block handles text-only queries and fallthrough from image handling
                        if not staged_image_path or (staged_image_path and query_type not in ["IMAGE_QUERY", "EXPLAIN_IMAGE"]):
                            query_type = classify_query(user_query) # Ensure classification is fresh if falling through
                            logger.info(f"Processing as text-only query. Type: {query_type}")
                            
                            if query_type == "GREETING":
                                llm_response = simple_question_llm_chain.invoke({"question": user_query, "chat_history": current_memory.chat_memory.messages})
                                response_content += llm_response
                                current_memory.save_context({"question": user_query}, {"answer": llm_response})
                                should_stream = True
            
                            elif "TOOL_SUMMARIZE" in query_type or "TOOL_METADATA" in query_type:
                                if agent_executor:
                                    with st.spinner(UI_TEXTS[st.session_state.ui_lang]["thinking_status"]):
                                        agent_response_dict = agent_executor.invoke({
                                            "input": user_query,
                                            "chat_history": current_memory.chat_memory.messages
                                        })
                                        agent_output = agent_response_dict.get('output', "Error: Could not get a response from the agent.")
                                        response_content += agent_output
                                        current_memory.save_context({"question": user_query}, {"answer": agent_output})
                                    should_stream = True
                                else:
                                    response_content += "The tool execution agent is not initialized."
                                    should_stream = False
            
                            elif "RAG_QUESTION" in query_type:
                                if not st.session_state.hybrid_retriever_instance or qa_chain is None:
                                    response_content += UI_TEXTS[st.session_state.ui_lang]["rag_no_kb"]
                                    st.warning(response_content)
                                    should_stream = False
                                else:
                                    with st.status(UI_TEXTS[st.session_state.ui_lang]["rag_processing_status"], expanded=True) as status:
                                        status.update(label=UI_TEXTS[st.session_state.ui_lang]["rag_retrieving_status"], state="running")
                                        
                                        language_name = "German" if current_lang == 'de' else "English"
                                        qa_response = qa_chain.invoke({
                                            "question": user_query,
                                            "language": language_name
                                        })
                                        retrieved_answer = qa_response.get('answer', '').strip()
                                        
                                        if qa_response.get('source_documents'):
                                            logger.info(f"Retrieved {len(qa_response['source_documents'])} documents for RAG query.")
                                            for doc in qa_response['source_documents']:
                                                source_path = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                                                sources_to_display.append(f"{source_path}")
                                                logger.info(f"RAG Document Source: {source_path}")
                                        else:
                                            logger.info("No documents retrieved for RAG query.")
                        
                                        status.update(label="Response generated!", state="complete", expanded=False)
                        
                                        if "NO_ANSWER_IN_CONTEXT" in retrieved_answer.upper() or len(retrieved_answer) < 50 or "I don't know" in retrieved_answer.lower():
                                            response_content += UI_TEXTS[st.session_state.ui_lang]["rag_inconclusive"]
                                            should_stream = False
                                            sources_to_display = []
                                        else:
                                            response_content += retrieved_answer
                                            should_stream = True
            
                            elif "GENERAL_CHAT" in query_type:
                                if simple_question_llm_chain:
                                    with st.status(UI_TEXTS[st.session_state.ui_lang]["general_chat_status"], expanded=False):
                                        llm_response = simple_question_llm_chain.invoke({"question": user_query, "chat_history": current_memory.chat_memory.messages})
                                        response_content += llm_response
                                        current_memory.save_context({"question": user_query}, {"answer": llm_response})
                                    should_stream = True
                                else:
                                    response_content += UI_TEXTS[st.session_state.ui_lang]["general_chat_uninitialized"]
                                    should_stream = False
            
                            elif "IMAGE_QUERY" in query_type or "EXPLAIN_IMAGE" in query_type:
                                response_content += "Please upload an image first to use this feature."
                                should_stream = False
            
                            else: # UNKNOWN
                                response_content += UI_TEXTS[st.session_state.ui_lang]["unknown_query_classification"]
                                should_stream = False
            
                    except Exception as e:
                        logger.error(f"Error during query processing: {e}", exc_info=True)
                        error_message = UI_TEXTS[st.session_state.ui_lang]["internal_error"]
                        st.error(error_message)
                        response_content = error_message
                        should_stream = False
                        sources_to_display = []
            
                if should_stream:
                    full_response = ""
                    response_words = response_content.split(" ")
                    for word in response_words:
                        full_response += word + " "
                        response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                        time.sleep(0.02)
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.markdown(response_content)
                
                if similar_images_found:
                    st.markdown(UI_TEXTS[st.session_state.ui_lang]["similar_images_found"])
                    cols = st.columns(3)
                    unique_displayed_images = set()
                    for i, img_path in enumerate(similar_images_found):
                        if img_path not in unique_displayed_images and i < 3:
                            logger.info(f"Attempting to display image from path: {img_path}")
                            try:
                                if os.path.exists(img_path):
                                    cols[i].image(img_path, caption=os.path.basename(img_path), width=150)
                                    unique_displayed_images.add(img_path)
                                else:
                                    logger.error(f"Image file not found for display: {img_path}")
                                    cols[i].warning(f"Image not found: {os.path.basename(img_path)}")
                            except Exception as e:
                                logger.error(f"Could not display {os.path.basename(img_path)}: {e}", exc_info=True)
                                cols[i].warning(f"Could not display {os.path.basename(img_path)}: {e}")

                # MODIFIED: Limit sources to display to only the first one if any exist and remove duplicates
                unique_sources = list(dict.fromkeys(sources_to_display))
                if unique_sources:
                    sources_to_display = [unique_sources[0]]
                    source_expander_placeholder = st.empty() 
                    with source_expander_placeholder.expander(UI_TEXTS[st.session_state.ui_lang]["sources_expander"]):
                        for source in sources_to_display:
                            st.markdown(source)
        
                current_messages.append({"role": "assistant", "content": response_content, "sources": sources_to_display, "similar_images": similar_images_found})
