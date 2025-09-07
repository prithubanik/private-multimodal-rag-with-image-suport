import os
import logging
import hashlib
import json
import time
import sys
import uuid
import shutil # Added for file operations like copying
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime # Import datetime

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredImageLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings # Corrected import for OllamaEmbeddings
from langchain_core.embeddings import Embeddings # Corrected import for Embeddings abstract base class
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from PIL import Image # Used for checking image dimensions
import io # Import io for Image.open(io.BytesIO(...))

import ollama # Import the ollama client
import requests # Import for catching connection errors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
TEXT_EMBEDDING_MODEL_NAME = os.environ.get("TEXT_EMBEDDING_MODEL", "nomic-embed-text")
# Define the multimodal model name you intend to use for vision (e.g., LLaVA, gemma3:2b, gemma3:27b)
MULTIMODAL_OLLAMA_MODEL_NAME = os.environ.get("MULTIMODAL_OLLAMA_MODEL", "gemma3:27b")
IMAGE_VECTORSTORE_DIR_NAME = "images_faiss_index" # Renamed for clarity and consistency with previous suggestions

# Define a minimum dimension for extracted images to be considered valid
MIN_IMAGE_DIMENSION = 150 # Pixels (e.g., 150x150 pixels minimum for width and height)

# --- Custom JSON Encoder for Datetime Objects ---
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects and custom objects with __dict__."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # If the object has a __dict__ (like TypedDicts or other custom classes),
        # convert it to a plain dictionary and then let the base JSONEncoder handle it.
        # We do NOT recursively call self.default on its items here;
        # the base encoder will handle the recursion for the new dictionary's contents.
        if hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, list, dict, type(None))):
            return obj.__dict__
        
        # For namedtuples or similar structures that have _asdict()
        if hasattr(obj, '_asdict'):
            return obj._asdict()

        # Let the base class default method handle all other basic types
        # (str, int, float, bool, None, dict, list, tuple)
        # and raise TypeError for truly unsupported types.
        return json.JSONEncoder.default(self, obj)

# --- Helper function to check if an Ollama model is ready ---
def is_ollama_model_ready(model_name: str, client: ollama.Client) -> bool:
    """Checks if a given Ollama model is ready for use by attempting to 'show' it."""
    try:
        logger.info(f"Checking readiness of model '{model_name}' using ollama.show()...")
        client.show(model_name)
        logger.info(f"Model '{model_name}' is ready.")
        return True
    except ollama.ResponseError as re:
        logger.warning(f"Model '{model_name}' not yet ready (ollama.show failed): {re}")
        return False
    except requests.exceptions.ConnectionError as ce:
        logger.warning(f"Could not connect to Ollama server while checking model '{model_name}' readiness: {ce}")
        return False
    except Exception as e:
        logger.warning(f"An unexpected error occurred while checking model '{model_name}' readiness: {e}", exc_info=True)
        return False

# --- Text Embedding Model Initialization ---
# Increased retry logic with exponential backoff for text embedding model initialization
text_embedding_model_initialized = False
max_retries_text_embedding = 30 # Increased retries
initial_delay_text_embedding = 2 # seconds
max_delay_text_embedding = 30 # seconds

ollama_client_for_pull = ollama.Client(host=OLLAMA_BASE_URL) # Dedicated client for pull operations

for i in range(max_retries_text_embedding):
    try:
        # First, try to ensure the model is ready using ollama.show()
        if not is_ollama_model_ready(TEXT_EMBEDDING_MODEL_NAME, ollama_client_for_pull):
            # If not ready, attempt to pull it
            logger.info(f"Attempting to pull '{TEXT_EMBEDDING_MODEL_NAME}' as a fallback (Attempt {i+1}/{max_retries_text_embedding}).")
            try:
                ollama_client_for_pull.pull(TEXT_EMBEDDING_MODEL_NAME)
                logger.info(f"Successfully initiated pull for '{TEXT_EMBEDDING_MODEL_NAME}'. Waiting for it to be ready...")
                # Give it a moment after pulling before retrying the readiness check
                time.sleep(5) 
                if not is_ollama_model_ready(TEXT_EMBEDDING_MODEL_NAME, ollama_client_for_pull):
                    raise ValueError(f"Model '{TEXT_EMBEDDING_MODEL_NAME}' still not ready after pull attempt.")
            except Exception as pull_e:
                logger.error(f"Failed to pull '{TEXT_EMBEDDING_MODEL_NAME}': {pull_e}", exc_info=True)
                current_delay = min(max_delay_text_embedding, initial_delay_text_embedding * (2 ** i))
                time.sleep(current_delay)
                continue # Continue to next retry attempt

        # If model is ready, proceed with OllamaEmbeddings
        embedding_model = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        # Perform a test query to ensure the text embedding model is functional
        test_text_embedding = embedding_model.embed_query("test query")
        if not isinstance(test_text_embedding, list) or not test_text_embedding:
            raise ValueError("Text embedding model returned an invalid or empty embedding.")
        logger.info(f"Text embedding model '{TEXT_EMBEDDING_MODEL_NAME}' initialized and tested successfully. Vector length: {len(test_text_embedding)}")
        text_embedding_model_initialized = True
        break # Exit loop if successful
    except requests.exceptions.ConnectionError as ce:
        current_delay = min(max_delay_text_embedding, initial_delay_text_embedding * (2 ** i))
        logger.warning(f"Attempt {i+1}/{max_retries_text_embedding}: Could not connect to Ollama server for text embedding: {ce}. Retrying in {current_delay} seconds...")
        time.sleep(current_delay)
    except Exception as e:
        current_delay = min(max_delay_text_embedding, initial_delay_text_embedding * (2 ** i))
        logger.warning(f"Attempt {i+1}/{max_retries_text_embedding}: An unexpected error occurred during text embedding model initialization: {e}. Retrying in {current_delay} seconds...", exc_info=True)
        time.sleep(current_delay)

if not text_embedding_model_initialized:
    logger.error(f"Failed to initialize text embedding model '{TEXT_EMBEDDING_MODEL_NAME}' after multiple attempts. Exiting.")
    raise SystemExit(f"Text embedding model initialization failed. Please ensure Ollama is running and '{TEXT_EMBEDDING_MODEL_NAME}' is pulled.")


# --- Real Vision Embedding Model Implementation ---
class RealVisionEmbeddings(Embeddings): # Inherit from Embeddings
    """
    A class to integrate a real vision embedding model, prioritizing Ollama multimodal models.
    This class now conforms to LangChain's Embeddings interface.
    It generates an image description and then embeds that description using a text embedder.
    """
    def __init__(self):
        self.ollama_client = None
        self.multimodal_model_name = MULTIMODAL_OLLAMA_MODEL_NAME
        # Fallback text embedder for image descriptions and text queries
        self.text_embedder = embedding_model # Use the globally initialized text embedding model
        self._load_model() # Call load model after initializing text_embedder

    def _load_model(self):
        """
        Load Ollama client and verify the multimodal model is available.
        Includes retry logic with exponential backoff.
        """
        logger.info(f"Attempting to initialize Ollama client at {OLLAMA_BASE_URL} for multimodal model '{self.multimodal_model_name}'.")
        
        max_retries_multimodal = 20
        initial_delay_multimodal = 2
        max_delay_multimodal = 30

        for i in range(max_retries_multimodal):
            try:
                self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
                
                # Add a longer delay to ensure Ollama is fully up and models are loaded
                time.sleep(5) # Increased sleep time

                # Verify the multimodal model is available using is_ollama_model_ready
                if not is_ollama_model_ready(self.multimodal_model_name, self.ollama_client):
                    # If not ready, attempt to pull it
                    logger.info(f"Attempting to pull '{self.multimodal_model_name}' as a fallback (Attempt {i+1}/{max_retries_multimodal}).")
                    try:
                        # FIX: Corrected to pull the multimodal model, not the text embedding model
                        ollama_client_for_pull.pull(self.multimodal_model_name) 
                        logger.info(f"Successfully initiated pull for '{self.multimodal_model_name}'. Waiting for it to be ready...")
                        time.sleep(5) # Give it a moment after pulling
                        if not is_ollama_model_ready(self.multimodal_model_name, self.ollama_client):
                            raise ValueError(f"Model '{self.multimodal_model_name}' still not ready after pull attempt.")
                    except Exception as pull_e:
                        logger.error(f"Failed to pull '{self.multimodal_model_name}': {pull_e}", exc_info=True)
                        current_delay = min(max_delay_multimodal, initial_delay_multimodal * (2 ** i))
                        time.sleep(current_delay)
                        continue # Continue to next retry attempt

                # If model is ready, proceed
                models_response = self.ollama_client.list() # Get list for logging
                loggable_models_response = dict(models_response) 
                logger.info(f"Attempt {i+1}/{max_retries_multimodal}: Raw Ollama models list response: {json.dumps(loggable_models_response, indent=2, cls=DateTimeEncoder)}")
                
                # Corrected: Use 'model' key instead of 'name'
                available_models = [m.get('model') for m in models_response.get('models', []) if m.get('model')]
                logger.info(f"Attempt {i+1}/{max_retries_multimodal}: Parsed Ollama models available: {available_models}")

                model_found = False
                for model_name_in_list in available_models:
                    if self.multimodal_model_name == model_name_in_list or \
                       self.multimodal_model_name.split(':')[0] == model_name_in_list.split(':')[0]:
                        model_found = True
                        break

                if model_found:
                    logger.info(f"Ollama client initialized and multimodal model '{self.multimodal_model_name}' found.")
                    
                    # Try generating a dummy response to warm up and verify
                    try:
                        dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                        warmup_response = self.ollama_client.generate(
                            model=self.multimodal_model_name,
                            prompt="Describe this empty image.",
                            images=[dummy_image_b64]
                        )
                        logger.info(f"Multimodal model '{self.multimodal_model_name}' warmed up successfully. Response snippet: {warmup_response['response'][:50]}...")
                    except Exception as warmup_e:
                        logger.warning(f"Failed to warm up multimodal model '{self.multimodal_model_name}': {warmup_e}. Image embedding might still work but could be slow initially.")
                    return # Model loaded successfully, exit method
                else:
                    current_delay = min(max_delay_multimodal, initial_delay_multimodal * (2 ** i))
                    logger.warning(f"Attempt {i+1}/{max_retries_multimodal}: Multimodal Ollama model '{self.multimodal_model_name}' not found among available models. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay) # Wait before next retry
            except requests.exceptions.ConnectionError as ce:
                current_delay = min(max_delay_multimodal, initial_delay_multimodal * (2 ** i))
                logger.warning(f"Attempt {i+1}/{max_retries_multimodal}: Could not connect to Ollama server at {OLLAMA_BASE_URL}: {ce}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
            except Exception as e:
                current_delay = min(max_delay_multimodal, initial_delay_multimodal * (2 ** i))
                logger.warning(f"Attempt {i+1}/{max_retries_multimodal}: An unexpected error occurred during multimodal model initialization: {e}. Retrying in {current_delay} seconds...", exc_info=True)
                time.sleep(current_delay)
        
        # If loop finishes without success
        logger.error(f"Failed to load Ollama multimodal model '{self.multimodal_model_name}' after multiple attempts. Falling back to dummy embeddings.")
        self.ollama_client = None # Disable if model not found

    def embed_image(self, image_path: str) -> Optional[List[float]]:
        """
        Generates an embedding for a given image path.
        This method is primarily called by LangChain's FAISS when it needs to embed the
        'page_content' of a Document. For image documents, their page_content will be
        a textual description generated elsewhere. This method then uses the text_embedder
        to embed that description.
        """
        # This method is expected to embed text (the image description), not the raw image.
        # The actual image description generation happens in process_documents_incrementally.
        logger.debug(f"RealVisionEmbeddings.embed_image called (internally by FAISS). Delegating to text_embedder.")
        # The 'image_path' here is actually the page_content (description) passed by FAISS
        return self.text_embedder.embed_query(image_path) 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text documents.
        This is primarily used by FAISS when it needs to embed the 'page_content' of Documents.
        For image documents stored in FAISS, their 'page_content' is a textual description.
        """
        logger.debug(f"RealVisionEmbeddings.embed_documents called for {len(texts)} text documents. Delegating to text_embedder.")
        return self.text_embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text query.
        Used when a text query needs to be compared against image embeddings (e.g., "find images of cars").
        """
        logger.debug("RealVisionEmbeddings.embed_query called. Delegating to text_embedder for query.")
        return self.text_embedder.embed_query(text)


# Global instance for image embeddings
# This will try to load the multimodal model when the module is imported
vision_embeddings_model = RealVisionEmbeddings()


# --- Document Processing Functions ---

def generate_file_hash(filepath: str) -> str:
    """Generates an MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_file(filepath: str, extracted_images_output_dir: str) -> Tuple[List[Document], List[str]]:
    """
    Loads and processes a single document file (PDF, DOCX, Image).
    Extracts text and images, returning LangChain Document objects and paths to extracted images.
    """
    logger.info(f"Attempting to load document: {filepath}")
    documents = []
    extracted_image_paths = []
    file_extension = os.path.splitext(filepath)[1].lower()

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            # Extract images from PDF using PyMuPDF (fitz)
            try:
                import fitz # Ensure fitz is imported if used here
                pdf_document = fitz.open(filepath)
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Check image dimensions before saving
                        try:
                            # Use PIL to get image dimensions from bytes
                            img_pil = Image.open(io.BytesIO(image_bytes))
                            width, height = img_pil.size
                        except Exception as img_read_e:
                            logger.warning(f"Could not read image dimensions from bytes for {os.path.basename(filepath)} (Page {page_num+1}, Index {img_index}): {img_read_e}. Skipping image due to read error.")
                            continue # Skip this image if its dimensions cannot be read

                        if width >= MIN_IMAGE_DIMENSION and height >= MIN_IMAGE_DIMENSION:
                            image_filename = f"image_{os.path.basename(filepath).replace('.', '_')}_{page_num+1}_{img_index}.{image_ext}"
                            image_save_path = os.path.join(extracted_images_output_dir, image_filename)
                            
                            os.makedirs(extracted_images_output_dir, exist_ok=True)

                            with open(image_save_path, "wb") as f:
                                f.write(image_bytes)
                            logger.info(f"Extracted image from PDF: {image_save_path} ({width}x{height})")
                            extracted_image_paths.append(image_save_path) # Add to the list of extracted image paths
                        else:
                            logger.info(f"Skipping small image from PDF: {os.path.basename(filepath)} (Page {page_num+1}, Index {img_index}, {width}x{height} - below {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})")

            except ImportError:
                logger.warning("PyMuPDF (fitz) not installed. PDF image extraction skipped. Install with `pip install PyMuPDF`.")
            except Exception as e:
                logger.error(f"Error extracting images from PDF {filepath}: {e}", exc_info=True)

        elif file_extension == ".docx":
            loader = Docx2txtLoader(filepath) # Changed from UnstructuredWordDocumentLoader
            documents = loader.load()
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff"]:
            # For image files, use UnstructuredImageLoader which performs OCR
            loader = UnstructuredImageLoader(filepath)
            documents = loader.load()
            # For image files, the image itself is the source for the image vector store
            if documents:
                # For direct image files, we also apply the dimension filter
                try:
                    img_pil = Image.open(filepath)
                    width, height = img_pil.size
                    if width >= MIN_IMAGE_DIMENSION and height >= MIN_IMAGE_DIMENSION:
                        # Copy the original image to the extracted_images_output_dir
                        # to ensure all images for embedding are in a central, persistent location.
                        unique_img_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_{str(uuid.uuid4())[:8]}{file_extension}"
                        target_image_path = os.path.join(extracted_images_output_dir, unique_img_filename)
                        shutil.copy(filepath, target_image_path)
                        extracted_image_paths.append(target_image_path)
                        logger.info(f"Copied direct image file for embedding: {target_image_path} ({width}x{height})")
                    else:
                        logger.info(f"Skipping small direct image file: {filepath} ({width}x{height} - below {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})")
                except Exception as img_read_e:
                    logger.warning(f"Could not read dimensions for direct image file {filepath}: {img_read_e}. Skipping image for embedding.")
        else:
            logger.warning(f"Unsupported file type: {filepath}")
            return [], []

        # Add source metadata and UUID to each document
        for doc in documents:
            doc.metadata["source"] = filepath
            if "uuid" not in doc.metadata:
                doc.metadata["uuid"] = str(uuid.uuid4())
            doc.metadata["last_modified"] = os.path.getmtime(filepath)

        logger.info(f"Loaded {len(documents)} raw documents from {filepath}. Extracted {len(extracted_image_paths)} images (after filtering).")
        return documents, extracted_image_paths
    except Exception as e:
        logger.error(f"Error loading or processing {filepath}: {e}", exc_info=True)
        return [], []

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits a list of documents into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} raw text documents into {len(chunks)} chunks.")
    return chunks

def load_vectorstore(path: str, embeddings: Any) -> Optional[FAISS]:
    """Loads an existing FAISS vector store from the given path."""
    index_file_path = os.path.join(path, "index.faiss") # Construct the full path to the index file
    if os.path.exists(index_file_path): # Check if the index file itself exists
        try:
            logger.info(f"Attempting to load FAISS index from: {path}")
            vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Successfully loaded FAISS index from: {path}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {path}: {e}", exc_info=True)
            return None
    else:
        logger.info(f"FAISS index file not found at: {index_file_path}. Returning None.")
        return None

def add_documents_to_vectorstore(
    documents: List[Document],
    vectorstore_path: str,
    embeddings_model: Embeddings, # Type hint for clarity: expects a LangChain Embeddings object
    existing_vector_store: Optional[FAISS] = None
) -> FAISS:
    """Adds documents to an existing vector store or creates a new one."""
    if not documents and existing_vector_store is None:
        logger.info(f"No documents provided and no existing vector store. Creating a new, empty vector store at {vectorstore_path}.")
        # FAISS.from_documents requires at least one document to initialize.
        # Create a dummy document just for initialization.
        dummy_doc = Document(page_content="initialization_placeholder", metadata={"source": "internal_init"})
        vector_store = FAISS.from_documents([dummy_doc], embeddings_model)
        # Save it to ensure the index.faiss file is created
        vector_store.save_local(vectorstore_path)
        logger.info(f"Successfully created empty FAISS index at: {vectorstore_path}")
        return vector_store
    
    if existing_vector_store:
        if documents:
            logger.info(f"Adding {len(documents)} new documents to existing vector store at {vectorstore_path}.")
            existing_vector_store.add_documents(documents)
            logger.info(f"Finished adding {len(documents)} documents to existing FAISS index.")
        else:
            logger.info(f"No new documents to add to existing vector store at {vectorstore_path}.")
        vector_store = existing_vector_store
    else: # No existing_vector_store, but documents are present
        logger.info(f"Creating new vector store with {len(documents)} documents at {vectorstore_path}.")
        logger.info(f"Starting embedding and creating new FAISS index with {len(documents)} documents.")
        vector_store = FAISS.from_documents(documents, embeddings_model)
        logger.info(f"Finished embedding and creating new FAISS index with {len(documents)} documents.")
    
    logger.info(f"Saving vector store to: {vectorstore_path}...")
    vector_store.save_local(vectorstore_path)
    logger.info(f"Vector store successfully saved to: {vectorstore_path}")
    return vector_store

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Loads the processed documents manifest."""
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from manifest at {manifest_path}. Starting with empty manifest.")
            return {}
    return {}

def save_manifest(manifest: Dict[str, Any], manifest_path: str):
    """Saves the processed documents manifest."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifest saved to {manifest_path}")

def process_documents_incrementally(
    document_folder: str,
    vectorstore_path: str,
    embeddings: Embeddings, # This is the text embedding model (OllamaEmbeddings instance)
    extracted_images_output_dir: str,
    force_rebuild: bool = False # Added force_rebuild parameter
) -> Tuple[Optional[FAISS], Optional[FAISS]]: # Changed return type hint
    """
    Processes documents from a folder incrementally, updating the FAISS vector store.
    Handles both text and image documents, and updates a manifest file.
    If force_rebuild is True, all documents are re-processed regardless of changes.
    Returns a tuple of (text_vector_store, image_vector_store).
    """
    manifest_path = os.path.join(vectorstore_path, "processed_documents_manifest.json")
    processed_manifest = load_manifest(manifest_path)

    if force_rebuild:
        logger.info(f"Force rebuild initiated for {document_folder}. Clearing existing manifest and attempting to remove old FAISS indexes.")
        processed_manifest = {} # Clear manifest to force re-processing
        # Attempt to remove old FAISS indexes for a clean rebuild
        if os.path.exists(vectorstore_path):
            try:
                shutil.rmtree(vectorstore_path)
                logger.info(f"Removed old text FAISS index directory: {vectorstore_path}")
            except OSError as e:
                logger.warning(f"Error removing old text FAISS index directory {vectorstore_path}: {e}")
        
        image_vectorstore_path_for_clear = os.path.join(vectorstore_path, IMAGE_VECTORSTORE_DIR_NAME)
        if os.path.exists(image_vectorstore_path_for_clear):
            try:
                shutil.rmtree(image_vectorstore_path_for_clear)
                logger.info(f"Removed old image FAISS index directory: {image_vectorstore_path_for_clear}")
            except OSError as e:
                logger.warning(f"Error removing old image FAISS index directory {image_vectorstore_path_for_clear}: {e}")
        
        # Recreate the base vectorstore_path if it was removed
        os.makedirs(vectorstore_path, exist_ok=True)


    new_or_modified_text_docs = []
    new_or_modified_image_paths = [] # Store paths for image embedding

    # Load existing text vector store
    existing_text_vector_store = load_vectorstore(vectorstore_path, embeddings)

    # Initialize image vector store (always load/create with the vision embedder)
    image_vectorstore_path = os.path.join(vectorstore_path, IMAGE_VECTORSTORE_DIR_NAME) 
    os.makedirs(image_vectorstore_path, exist_ok=True) # Ensure directory exists

    existing_image_vector_store = load_vectorstore(image_vectorstore_path, vision_embeddings_model)
    logger.info(f"DEBUG: existing_image_vector_store after load_vectorstore (initial): {existing_image_vector_store is None}")
    
    # If the image vector store couldn't be loaded, create an empty one
    if existing_image_vector_store is None:
        logger.info(f"Image vector store at {image_vectorstore_path} not found or failed to load. Creating a new empty one.")
        # Call add_documents_to_vectorstore with an empty list to create a new empty store
        existing_image_vector_store = add_documents_to_vectorstore(
            [], # Empty list of documents
            image_vectorstore_path,
            vision_embeddings_model,
            None # No existing store to pass
        )
        if existing_image_vector_store is None:
            logger.error(f"Failed to create an empty image vector store at {image_vectorstore_path}. Image search will not work.")
            return existing_text_vector_store, None # Return None for image VS if creation failed
        else:
            logger.info(f"Successfully created empty image vector store at {image_vectorstore_path}.")


    # Track files found in current scan to detect deletions
    current_files_in_folder = set()

    logger.info(f"Processing documents from folder for incremental update: {document_folder}")
    for root, _, files in os.walk(document_folder):
        for filename in files:
            filepath = os.path.join(root, filename)
            current_files_in_folder.add(filepath)
            file_hash = generate_file_hash(filepath)
            last_modified = os.path.getmtime(filepath)

            # Check if file has been processed and not modified, UNLESS force_rebuild is True
            if not force_rebuild and \
               filepath in processed_manifest and \
               processed_manifest[filepath]["hash"] == file_hash and \
               processed_manifest[filepath]["last_modified"] == last_modified:
                logger.debug(f"Skipping unchanged file: {filepath}")
                continue

            logger.info(f"Processing new/modified file: {filename}")
            
            # Process the file and get LangChain Document objects and extracted image paths
            docs_from_file, extracted_imgs = process_file(filepath, extracted_images_output_dir)
            
            if not docs_from_file and not extracted_imgs:
                logger.warning(f"Could not process {filepath}. Skipping.")
                continue

            # Extend the list of new/modified text documents and image paths
            new_or_modified_text_docs.extend(chunk_documents(docs_from_file))
            new_or_modified_image_paths.extend(extracted_imgs)

            # Update manifest for this file
            processed_manifest[filepath] = {
                "hash": file_hash,
                "last_modified": last_modified,
                "chunks_count": len(chunk_documents(docs_from_file)), # Recalculate for manifest
                "processed_at": time.time(),
                "extracted_images": extracted_imgs
            }

    # Remove deleted files from manifest and vector store (manual removal not implemented for embeddings)
    files_to_delete = [f for f in processed_manifest if f not in current_files_in_folder]
    for filepath in files_to_delete:
        logger.info(f"File {filepath} deleted. Removing from manifest.")
        logger.warning(f"Manual removal of embeddings for deleted file {filepath} is not implemented. Consider full rebuild if many files are deleted.")
        del processed_manifest[filepath]

    # Add all new/modified text documents to the text vector store
    if new_or_modified_text_docs:
        existing_text_vector_store = add_documents_to_vectorstore(
            new_or_modified_text_docs,
            vectorstore_path,
            embeddings,
            existing_text_vector_store
        )
    else:
        logger.info(f"No new text documents to add to vector store at {vectorstore_path}.")


    # Now, existing_image_vector_store should always be a valid FAISS object (either loaded or newly created empty)
    # Proceed to add new/modified image documents to it
    if new_or_modified_image_paths:
        logger.info(f"Adding {len(new_or_modified_image_paths)} new image documents to image vector store.")
        image_docs_for_vs = []
        for img_path in new_or_modified_image_paths:
            image_description = ""
            try:
                if vision_embeddings_model.ollama_client:
                    with open(img_path, "rb") as f:
                        image_bytes = f.read()
                    response = vision_embeddings_model.ollama_client.generate(
                        model=vision_embeddings_model.multimodal_model_name,
                        prompt="Describe this image in detail for retrieval purposes, focusing on key objects, text, and overall content.",
                        images=[image_bytes],
                        options={"num_predict": 128}
                    )
                    image_description = response['response'].strip()
                    logger.info(f"Generated description for {os.path.basename(img_path)}: {image_description[:100]}...")
                else:
                    logger.warning(f"Vision embedding model client not available. Cannot generate description for {img_path}.")
                    image_description = f"Visual content of image from {os.path.basename(img_path)} (description unavailable)."
            except Exception as e:
                logger.error(f"Error generating description for image {img_path}: {e}", exc_info=True)
                image_description = f"Visual content of image from {os.path.basename(img_path)} (description generation failed)."


            # Create a Langchain Document for each image for storage in FAISS
            image_doc = Document(
                # Use the generated description as the page_content
                page_content=image_description,
                metadata={
                    "source": img_path,
                    "type": "image_embedding",
                    "image_path": img_path, # Store the actual image path here
                    "image_description": image_description, # Store the description explicitly
                    "uuid": str(uuid.uuid4()),
                    "timestamp": time.time()
                }
            )
            image_docs_for_vs.append(image_doc)
        
        try:
            # Add documents to the now-guaranteed-to-exist image vector store
            existing_image_vector_store.add_documents(image_docs_for_vs)
            existing_image_vector_store.save_local(image_vectorstore_path) # Save changes
            logger.info(f"Successfully added new image documents and saved vector store at {image_vectorstore_path}.")
        except Exception as e:
            logger.error(f"Error during adding/saving new image documents to {image_vectorstore_path}: {e}", exc_info=True)
            # If adding new documents fails, the existing_image_vector_store might still be valid,
            # but its content might be incomplete. We'll keep it as is, but log the error.
    else:
        logger.info("No new image documents to add to existing image vector store.")


    save_manifest(processed_manifest, manifest_path)
    logger.info(f"Document processing complete. Total processed files in manifest: {len(processed_manifest)}")
    logger.info(f"Final state: Text VS is {'not None' if existing_text_vector_store else 'None'}, Image VS is {'not None' if existing_image_vector_store else 'None'}")
    return existing_text_vector_store, existing_image_vector_store # Return both vector stores

# === Main Execution for Demonstration (Optional) ===
if __name__ == "__main__":
    print("Running document_processor.py in standalone mode for testing.")

    # Define the application's expected document and FAISS paths
    APP_PROF_DOCS_DIR = "/app/documents/professor_docs"
    APP_SHARED_DOCS_DIR = "/app/documents/shared_docs"
    APP_PROF_FAISS_DIR = "/app/faiss_index/professor"
    APP_STUDENT_FAISS_DIR = "/app/faiss_index/student"
    APP_GLOBAL_EXTRACTED_IMAGES_DIR = "/app/faiss_index/extracted_images" # This is where raw images are stored

    # Ensure these directories exist for testing
    os.makedirs(APP_PROF_DOCS_DIR, exist_ok=True)
    os.makedirs(APP_SHARED_DOCS_DIR, exist_ok=True)
    os.makedirs(APP_PROF_FAISS_DIR, exist_ok=True)
    os.makedirs(APP_STUDENT_FAISS_DIR, exist_ok=True)
    os.makedirs(APP_GLOBAL_EXTRACTED_IMAGES_DIR, exist_ok=True)
    
    # Create dummy PDF, DOCX, and Image files in the *application's expected paths*
    dummy_pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj 3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj 4 0 obj<</Length 44>>stream\nBT /F1 24 Tf 100 700 Td (Hello from PDF!) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000055 00000 n\n0000000107 00000 n\n0000000196 00000 n\ntrailer<</Size 5/Root 1 0 R>>startxref\n296\n%%EOF"
    with open(os.path.join(APP_PROF_DOCS_DIR, "test_prof_doc.pdf"), "wb") as f:
        f.write(dummy_pdf_content)
    print(f"Created dummy PDF: {os.path.join(APP_PROF_DOCS_DIR, 'test_prof_doc.pdf')}")
    
    with open(os.path.join(APP_SHARED_DOCS_DIR, "test_shared_doc.txt"), "w") as f:
        f.write("This is a shared document. It contains information about Aalen University research. " * 10) # Make it longer for chunking
    print(f"Created dummy TXT: {os.path.join(APP_SHARED_DOCS_DIR, 'test_shared_doc.txt')}")
    
    # Create a dummy image file (100x100 is too small for MIN_IMAGE_DIMENSION, should be skipped)
    dummy_small_image_path = os.path.join(APP_SHARED_DOCS_DIR, "test_small_image.png")
    Image.new('RGB', (100, 100), color = 'blue').save(dummy_small_image_path)
    print(f"Created dummy small image: {dummy_small_image_path}")

    # Create a dummy large image file (will be processed)
    dummy_large_image_path = os.path.join(APP_SHARED_DOCS_DIR, "test_large_image.png")
    Image.new('RGB', (200, 200), color = 'green').save(dummy_large_image_path)
    print(f"Created dummy large image: {dummy_large_image_path}")


    # Simulate processing for Professor role
    print("\n--- Processing for Professor Role ---")
    processed_text_faiss_prof, processed_image_faiss_prof = process_documents_incrementally( # Unpack both return values
        document_folder=APP_PROF_DOCS_DIR,
        vectorstore_path=APP_PROF_FAISS_DIR, # Use APP_PROF_FAISS_DIR for text
        embeddings=embedding_model,
        extracted_images_output_dir=APP_GLOBAL_EXTRACTED_IMAGES_DIR, # All extracted images go here
        force_rebuild=True # Force rebuild for testing
    )
    if processed_text_faiss_prof:
        print(f"Professor Text FAISS index total entries: {processed_text_faiss_prof.index.ntotal}")
        # Load image FAISS using the explicitly defined prof_image_faiss_dir
        # Note: image_vectorstore_path is handled internally by process_documents_incrementally
        # We need to manually load it here for testing purposes.
        prof_image_faiss_dir = os.path.join(APP_PROF_FAISS_DIR, IMAGE_VECTORSTORE_DIR_NAME)
        # Use the returned image FAISS object directly, or reload if necessary for consistency in test block
        image_faiss_prof = processed_image_faiss_prof # Use the returned object
        if image_faiss_prof:
            print(f"Professor Image FAISS index total entries: {image_faiss_prof.index.ntotal}")
            # Simulate a query for similar images
            # Need to get a path to an extracted image from GLOBAL_EXTRACTED_IMAGES_DIR
            # The PDF extraction creates image_test_prof_doc_pdf_1_0.png
            query_image_path_from_pdf = os.path.join(APP_GLOBAL_EXTRACTED_IMAGES_DIR, "image_test_prof_doc_pdf_1_0.png")
            if image_faiss_prof.index.ntotal > 0 and os.path.exists(query_image_path_from_pdf):
                print("\nSimulating image similarity search for a PDF-extracted image...")
                dummy_embedding = vision_embeddings_model.embed_query("A diagram of a process flow") # Query with text
                if dummy_embedding:
                    similar_image_docs = image_faiss_prof.similarity_search_by_vector(dummy_embedding, k=2)
                    print("Found similar images (paths):")
                    for doc in similar_image_docs:
                        print(f"- {doc.metadata.get('image_path', 'N/A')}")
                        print(f"  (Original Document: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Type: {doc.metadata.get('type')})")
                        print(f"  (Description: {doc.page_content[:50]}...)") # Print the description
                else:
                    print("Failed to get dummy embedding for image similarity search (PDF-extracted image).")
            else:
                print(f"No suitable query image found for similarity search for Professor ({query_image_path_from_pdf}).")
        else:
            print("Professor Image FAISS index not loaded.")
    else:
        print("Professor Text FAISS index not created/loaded.")


    # Simulate processing for Student role (only shared docs)
    print("\n--- Processing for Student Role ---")
    processed_text_faiss_student, processed_image_faiss_student = process_documents_incrementally( # Unpack both return values
        document_folder=APP_SHARED_DOCS_DIR,
        vectorstore_path=APP_STUDENT_FAISS_DIR, # Use APP_STUDENT_FAISS_DIR for text
        embeddings=embedding_model,
        extracted_images_output_dir=APP_GLOBAL_EXTRACTED_IMAGES_DIR, # All extracted images go here
        force_rebuild=True # Force rebuild for testing
    )
    if processed_text_faiss_student:
        print(f"Student Text FAISS index total entries: {processed_text_faiss_student.index.ntotal}")
        student_image_faiss_dir = os.path.join(APP_STUDENT_FAISS_DIR, IMAGE_VECTORSTORE_DIR_NAME)
        # Use the returned image FAISS object directly, or reload if necessary for consistency in test block
        image_faiss_student = processed_image_faiss_student # Use the returned object
        if image_faiss_student:
            print(f"Student Image FAISS index total entries: {image_faiss_student.index.ntotal}")
            # Simulate a query for the large dummy image
            if image_faiss_student.index.ntotal > 0 and os.path.exists(dummy_large_image_path):
                print("\nSimulating image similarity search for a direct large image...")
                copied_dummy_large_image_path = None
                # Find the actual path where it was copied
                for root, _, files in os.walk(APP_GLOBAL_EXTRACTED_IMAGES_DIR):
                    for f_name in files:
                        if "test_large_image" in f_name:
                            copied_dummy_large_image_path = os.path.join(root, f_name)
                            break
                    if copied_dummy_large_image_path:
                        break

                if copied_dummy_large_image_path:
                    dummy_embedding = vision_embeddings_model.embed_query("A green square") # Query with text
                    if dummy_embedding:
                        similar_image_docs = image_faiss_student.similarity_search_by_vector(dummy_embedding, k=2)
                        print("Found similar images (paths):")
                        for doc in similar_image_docs:
                            print(f"- {doc.metadata.get('image_path', 'N/A')}")
                            print(f"  (Original Document: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Type: {doc.metadata.get('type')})")
                            print(f"  (Description: {doc.page_content[:50]}...)") # Print the description
                    else:
                        print("Failed to get dummy embedding for image similarity search (direct large image).")
                else:
                    print("Could not find copied dummy large image for similarity search.")
            else:
                print("No suitable query image found for similarity search for Student.")
        else:
            print("Student Image FAISS index not loaded.")
    else:
        print("Student Text FAISS index not created/loaded.")
