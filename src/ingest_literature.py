#!/usr/bin/env python3
"""
Literature Ingestion Module for Geothermal Digital Twin AI

Processes PDFs in knowledge/corpus/ directory to extract:
1. Text content → chunk → embed → ChromaDB (text_emb)
2. Images/figures → extract → OCR captions → CLIP embed → ChromaDB (image_emb)

Dependencies:
- PyPDF2: PDF text extraction
- PyMuPDF (fitz): Advanced PDF processing and image extraction  
- tiktoken: Text chunking
- openai: Text embeddings
- chromadb: Vector database
- pytesseract: OCR for captions
- open_clip: Image embeddings
- PIL: Image processing

Usage:
    python src/ingest_literature.py [--rebuild]
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time

# Core processing
import PyPDF2
import fitz  # PyMuPDF
import tiktoken
from PIL import Image, ImageEnhance
import pytesseract

# AI/ML libraries
import openai
import chromadb
import open_clip
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CORPUS_DIR = Path("knowledge/corpus")
IMAGES_DIR = Path("knowledge/images")
TEXT_EMB_DIR = Path("knowledge/text_emb")
IMAGE_EMB_DIR = Path("knowledge/image_emb")

# Text processing parameters
MAX_CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200    # tokens
ENCODING_MODEL = "cl100k_base"  # GPT-4 encoding

# Image processing parameters
MIN_IMAGE_SIZE = (100, 100)  # Minimum width, height
MAX_IMAGE_SIZE = (2048, 2048)  # Maximum for processing
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

# ChromaDB collections
TEXT_COLLECTION = "geothermal_texts"
IMAGE_COLLECTION = "geothermal_images"


class LiteratureIngester:
    """Main class for processing literature and building knowledge base."""
    
    def __init__(self):
        """Initialize the ingester with required models and databases."""
        self.setup_directories()
        self.encoding = tiktoken.get_encoding(ENCODING_MODEL)
        
        # Initialize ChromaDB
        self.setup_chromadb()
        
        # Initialize CLIP for image embeddings
        self.setup_clip()
        
        # Processed files tracking
        self.processed_files = self.load_processed_files()
        
    def setup_directories(self):
        """Create required directories."""
        for dir_path in [CORPUS_DIR, IMAGES_DIR, TEXT_EMB_DIR, IMAGE_EMB_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure ready")
    
    def setup_chromadb(self):
        """Initialize ChromaDB collections."""
        try:
            # Initialize client
            self.chroma_client = chromadb.PersistentClient(path=str(TEXT_EMB_DIR))
            
            # Get or create text collection
            try:
                self.text_collection = self.chroma_client.get_collection(TEXT_COLLECTION)
                logger.info(f"Loaded existing text collection: {TEXT_COLLECTION}")
            except:
                self.text_collection = self.chroma_client.create_collection(
                    name=TEXT_COLLECTION,
                    metadata={"description": "Geothermal literature text embeddings"}
                )
                logger.info(f"Created new text collection: {TEXT_COLLECTION}")
            
            # Initialize image client 
            self.image_client = chromadb.PersistentClient(path=str(IMAGE_EMB_DIR))
            
            # Get or create image collection
            try:
                self.image_collection = self.image_client.get_collection(IMAGE_COLLECTION)
                logger.info(f"Loaded existing image collection: {IMAGE_COLLECTION}")
            except:
                self.image_collection = self.image_client.create_collection(
                    name=IMAGE_COLLECTION,
                    metadata={"description": "Geothermal literature image embeddings"}
                )
                logger.info(f"Created new image collection: {IMAGE_COLLECTION}")
                
        except Exception as e:
            logger.error(f"ChromaDB setup failed: {e}")
            raise
    
    def setup_clip(self):
        """Initialize CLIP model for image embeddings."""
        try:
            # Load CLIP model
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained='openai'
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.clip_model = self.clip_model.to(self.device)
            
            logger.info(f"CLIP model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"CLIP setup failed: {e}")
            raise
    
    def load_processed_files(self) -> Dict[str, float]:
        """Load tracking of previously processed files."""
        processed_file = TEXT_EMB_DIR / "processed_files.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed files tracking: {e}")
        return {}
    
    def save_processed_files(self):
        """Save tracking of processed files."""
        processed_file = TEXT_EMB_DIR / "processed_files.json"
        try:
            with open(processed_file, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save processed files tracking: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file content to detect changes."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Could not hash file {file_path}: {e}")
            return ""
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file needs processing (new or modified)."""
        file_key = str(file_path.relative_to(CORPUS_DIR))
        current_hash = self.get_file_hash(file_path)
        
        if file_key not in self.processed_files:
            return True
            
        return self.processed_files[file_key] != current_hash
    
    def extract_text_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_and_images(self, pdf_path: Path) -> Tuple[str, List[Dict]]:
        """
        Extract text and images from PDF using PyMuPDF.
        
        Returns:
            Tuple of (text_content, image_data_list)
        """
        text_content = ""
        images_data = []
        
        try:
            # Open PDF
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                
                # Extract text
                page_text = page.get_text()
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                # Extract images
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_doc, xref)
                        
                        # Skip if too small or wrong format
                        if pix.width < MIN_IMAGE_SIZE[0] or pix.height < MIN_IMAGE_SIZE[1]:
                            pix = None
                            continue
                        if pix.n - pix.alpha < 3:  # Skip grayscale
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Generate image filename
                        img_filename = f"{pdf_path.stem}_p{page_num+1}_img{img_index+1}.png"
                        img_path = IMAGES_DIR / img_filename
                        
                        # Save image
                        pix.save(img_path)
                        
                        # Extract caption (text near image)
                        caption = self.extract_image_caption(page, img, page_text)
                        
                        # Store image data
                        images_data.append({
                            'path': img_path,
                            'source_pdf': pdf_path.name,
                            'page': page_num + 1,
                            'caption': caption,
                            'size': (pix.width, pix.height)
                        })
                        
                        pix = None  # Free memory
                        logger.debug(f"Extracted image: {img_filename}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num} of {pdf_path}: {e}")
                        continue
            
            pdf_doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            # Fallback to PyPDF2 for text
            text_content = self.extract_text_pypdf2(pdf_path)
        
        return text_content, images_data
    
    def extract_image_caption(self, page, img_info, page_text: str) -> str:
        """
        Extract caption for an image by finding nearby text.
        Simple heuristic: look for text containing 'Fig', 'Figure', 'Chart', etc.
        """
        try:
            # Get image position
            img_bbox = page.get_image_bbox(img_info)
            
            # Search for caption keywords in nearby text
            caption_keywords = ['fig', 'figure', 'chart', 'graph', 'image', 'photo', 'diagram']
            
            # Split page text into lines and search for captions
            lines = page_text.lower().split('\n')
            for line in lines:
                if any(keyword in line for keyword in caption_keywords):
                    # Return first reasonable caption found
                    if len(line.strip()) > 10 and len(line.strip()) < 200:
                        return line.strip()
            
            return f"Image from {page.number + 1}"
            
        except Exception as e:
            logger.debug(f"Caption extraction failed: {e}")
            return "Extracted image"
    
    def chunk_text(self, text: str, source_info: Dict) -> List[Dict]:
        """
        Split text into chunks with overlap for embedding.
        
        Args:
            text: Text to chunk
            source_info: Metadata about source document
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        
        chunks = []
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(tokens):
            # Calculate end index
            end_idx = min(start_idx + MAX_CHUNK_SIZE, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                start_idx = end_idx
                continue
            
            # Create chunk metadata
            chunk_data = {
                'id': f"{source_info['file_id']}_chunk_{chunk_id}",
                'text': chunk_text,
                'source': source_info['filename'],
                'source_type': 'pdf',
                'chunk_index': chunk_id,
                'token_count': len(chunk_tokens),
                'file_path': source_info['file_path']
            }
            
            chunks.append(chunk_data)
            
            # Move start index with overlap
            start_idx = end_idx - CHUNK_OVERLAP
            chunk_id += 1
        
        return chunks
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text."""
        try:
            # Check if OpenAI API key is set
            if not openai.api_key:
                logger.error("OpenAI API key not set")
                return []
            
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return []
    
    def get_image_embedding(self, image_path: Path, caption: str = "") -> List[float]:
        """Get CLIP embedding for image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Resize if too large
            if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
                image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Preprocess for CLIP
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_embedding = image_features.cpu().numpy().flatten().tolist()
            
            return image_embedding
            
        except Exception as e:
            logger.error(f"CLIP embedding failed for {image_path}: {e}")
            return []
    
    def process_pdf(self, pdf_path: Path) -> bool:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if processing succeeded
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            # Extract text and images
            text_content, images_data = self.extract_text_and_images(pdf_path)
            
            if not text_content.strip() and not images_data:
                logger.warning(f"No content extracted from {pdf_path}")
                return False
            
            # Process text content
            if text_content.strip():
                source_info = {
                    'filename': pdf_path.name,
                    'file_path': str(pdf_path),
                    'file_id': pdf_path.stem
                }
                
                # Chunk text
                chunks = self.chunk_text(text_content, source_info)
                logger.info(f"Created {len(chunks)} text chunks from {pdf_path.name}")
                
                # Process chunks and add to ChromaDB
                for chunk in chunks:
                    # Get embedding
                    embedding = self.get_text_embedding(chunk['text'])
                    if embedding:
                        # Add to ChromaDB
                        self.text_collection.add(
                            ids=[chunk['id']],
                            embeddings=[embedding],
                            documents=[chunk['text']],
                            metadatas=[{
                                'source': chunk['source'],
                                'source_type': chunk['source_type'],
                                'chunk_index': chunk['chunk_index'],
                                'token_count': chunk['token_count'],
                                'file_path': chunk['file_path']
                            }]
                        )
                
            # Process images
            if images_data:
                logger.info(f"Processing {len(images_data)} images from {pdf_path.name}")
                
                for img_data in images_data:
                    # Get image embedding
                    embedding = self.get_image_embedding(img_data['path'], img_data['caption'])
                    if embedding:
                        # Add to image ChromaDB
                        img_id = f"{pdf_path.stem}_{img_data['page']}_img_{img_data['path'].stem}"
                        
                        self.image_collection.add(
                            ids=[img_id],
                            embeddings=[embedding],
                            documents=[img_data['caption']],  # Use caption as document
                            metadatas=[{
                                'source_pdf': img_data['source_pdf'],
                                'page': img_data['page'],
                                'caption': img_data['caption'],
                                'image_path': str(img_data['path']),
                                'size_w': img_data['size'][0],
                                'size_h': img_data['size'][1]
                            }]
                        )
            
            # Mark as processed
            file_key = str(pdf_path.relative_to(CORPUS_DIR))
            self.processed_files[file_key] = self.get_file_hash(pdf_path)
            
            logger.info(f"Successfully processed {pdf_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            return False
    
    def process_all_pdfs(self, rebuild: bool = False) -> Dict[str, int]:
        """
        Process all PDFs in corpus directory.
        
        Args:
            rebuild: If True, reprocess all files regardless of cache
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting literature ingestion...")
        
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0
        }
        
        # Find all PDF files
        pdf_files = list(CORPUS_DIR.rglob("*.pdf"))
        stats['total_files'] = len(pdf_files)
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                # Check if processing needed
                if not rebuild and not self.should_process_file(pdf_path):
                    logger.info(f"Skipping {pdf_path.name} (already processed)")
                    stats['skipped_files'] += 1
                    continue
                
                # Process the PDF
                if self.process_pdf(pdf_path):
                    stats['processed_files'] += 1
                else:
                    stats['failed_files'] += 1
                
                # Save progress periodically
                if (stats['processed_files'] + stats['failed_files']) % 5 == 0:
                    self.save_processed_files()
                
            except Exception as e:
                logger.error(f"Unexpected error processing {pdf_path}: {e}")
                stats['failed_files'] += 1
        
        # Save final state
        self.save_processed_files()
        
        logger.info(f"Literature ingestion complete: {stats}")
        return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest literature into knowledge base")
    parser.add_argument("--rebuild", action="store_true", 
                       help="Rebuild all embeddings (ignore cache)")
    
    args = parser.parse_args()
    
    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize and run ingester
        ingester = LiteratureIngester()
        stats = ingester.process_all_pdfs(rebuild=args.rebuild)
        
        # Print summary
        print(f"\n=== Literature Ingestion Summary ===")
        print(f"Total files found: {stats['total_files']}")
        print(f"Files processed: {stats['processed_files']}")
        print(f"Files skipped: {stats['skipped_files']}")
        print(f"Files failed: {stats['failed_files']}")
        
        if stats['failed_files'] > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()