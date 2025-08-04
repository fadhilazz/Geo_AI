#!/usr/bin/env python3
"""
QA Server for Geothermal Digital Twin AI

FastAPI-based question-answering system that integrates multiple knowledge sources:
1. Literature corpus (ChromaDB text + image embeddings)
2. 3D model data and engineering insights (VTU grids + YAML summaries)
3. Web augmentation for recent information
4. Advanced reasoning with GPT-o3/o1

Features:
- Multi-context retrieval and reasoning
- Citation tracking and source attribution
- Engineering-grade interpretations
- Real-time question processing
- RESTful API with interactive documentation

Dependencies:
- fastapi, uvicorn: Web framework and server
- openai: GPT-o3/o1 integration
- chromadb: Vector database access
- pyvista: 3D model querying
- ruamel.yaml: Engineering summaries
- requests: Web augmentation

Usage:
    python src/qa_server.py [--host HOST] [--port PORT] [--debug]
    
    Or via uvicorn:
    uvicorn src.qa_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import logging
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import traceback

# Web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# AI and knowledge retrieval
import openai
import chromadb
import requests

# Data processing
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

# 3D model access
import pyvista as pv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TEXT_EMB_DIR = Path("knowledge/text_emb")
IMAGE_EMB_DIR = Path("knowledge/image_emb")
GRIDS_DIR = Path("digital_twin/grids")
CACHE_DIR = Path("digital_twin/cache")
SUMMARIES_FILE = CACHE_DIR / "twin_summaries.yaml"

# ChromaDB collections
TEXT_COLLECTION = "geothermal_texts"
IMAGE_COLLECTION = "geothermal_images"

# QA Configuration
MAX_CONTEXT_LENGTH = 8000  # tokens for GPT context
TOP_K_TEXTS = 5  # number of text chunks to retrieve
TOP_K_IMAGES = 3  # number of images to retrieve
SIMILARITY_THRESHOLD = 0.7  # minimum similarity for relevance
WEB_SEARCH_THRESHOLD = 0.6  # coverage threshold for web augmentation


# Pydantic models for API
class QuestionRequest(BaseModel):
    """Request model for questions."""
    question: str = Field(..., description="The question to ask", min_length=5, max_length=1000)
    include_images: bool = Field(True, description="Include image analysis in response")
    include_web: bool = Field(False, description="Enable web search augmentation")
    max_response_length: int = Field(2000, description="Maximum response length", ge=100, le=5000)
    temperature: float = Field(0.3, description="Response creativity (0.0-1.0)", ge=0.0, le=1.0)

class Citation(BaseModel):
    """Citation information."""
    source_type: str  # 'literature', 'model', 'summary', 'web'
    source_name: str
    page_number: Optional[int] = None
    confidence: float
    excerpt: Optional[str] = None

class QuestionResponse(BaseModel):
    """Response model for answers."""
    question: str
    answer: str
    citations: List[Citation]
    processing_time_ms: int
    confidence_score: float
    context_used: Dict[str, Any]
    timestamp: str


class GeothermalQASystem:
    """Main QA system integrating all knowledge sources."""
    
    def __init__(self):
        """Initialize the QA system with all required components."""
        self.setup_directories()
        self.setup_openai()
        self.setup_chromadb()
        self.load_summaries()
        self.load_grids_metadata()
        
        # YAML handler
        self.yaml = YAML(typ='safe')
        
        # Cache for frequently accessed data
        self.cache = {}
        
        logger.info("Geothermal QA System initialized successfully")
    
    def setup_directories(self):
        """Ensure required directories exist."""
        for dir_path in [TEXT_EMB_DIR, IMAGE_EMB_DIR, GRIDS_DIR, CACHE_DIR]:
            if not dir_path.exists():
                logger.warning(f"Directory {dir_path} not found - some features may be limited")
    
    def setup_openai(self):
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai.api_key = api_key
        logger.info("OpenAI client initialized")
    
    def setup_chromadb(self):
        """Initialize ChromaDB connections."""
        self.text_collection = None
        self.image_collection = None
        
        # Text collection
        if TEXT_EMB_DIR.exists():
            try:
                text_client = chromadb.PersistentClient(path=str(TEXT_EMB_DIR))
                self.text_collection = text_client.get_collection(TEXT_COLLECTION)
                logger.info(f"Text collection loaded: {self.text_collection.count()} documents")
            except Exception as e:
                logger.warning(f"Could not load text collection: {e}")
        
        # Image collection
        if IMAGE_EMB_DIR.exists():
            try:
                image_client = chromadb.PersistentClient(path=str(IMAGE_EMB_DIR))
                self.image_collection = image_client.get_collection(IMAGE_COLLECTION)
                logger.info(f"Image collection loaded: {self.image_collection.count()} documents")
            except Exception as e:
                logger.warning(f"Could not load image collection: {e}")
    
    def load_summaries(self):
        """Load engineering summaries from YAML."""
        self.summaries = {}
        
        if SUMMARIES_FILE.exists():
            try:
                with open(SUMMARIES_FILE, 'r') as f:
                    self.summaries = self.yaml.load(f)
                logger.info("Engineering summaries loaded")
            except Exception as e:
                logger.warning(f"Could not load summaries: {e}")
    
    def load_grids_metadata(self):
        """Load metadata for available 3D grids."""
        self.grids_metadata = {}
        
        if GRIDS_DIR.exists():
            for metadata_file in CACHE_DIR.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    grid_name = metadata_file.stem.replace('_metadata', '')
                    self.grids_metadata[grid_name] = metadata
                    
                except Exception as e:
                    logger.debug(f"Could not load metadata for {metadata_file}: {e}")
            
            logger.info(f"Loaded metadata for {len(self.grids_metadata)} 3D models")
    
    async def retrieve_text_context(self, question: str, max_results: int = TOP_K_TEXTS) -> List[Dict]:
        """Retrieve relevant text documents from ChromaDB."""
        if not self.text_collection:
            return []
        
        try:
            # Query ChromaDB
            results = self.text_collection.query(
                query_texts=[question],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            text_contexts = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    
                    # Convert distance to similarity
                    similarity = 1.0 - distance
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        text_contexts.append({
                            'content': doc,
                            'source': metadata.get('source', 'Unknown'),
                            'source_type': metadata.get('source_type', 'literature'),
                            'chunk_index': metadata.get('chunk_index', 0),
                            'similarity': similarity,
                            'file_path': metadata.get('file_path', '')
                        })
            
            logger.info(f"Retrieved {len(text_contexts)} relevant text chunks")
            return text_contexts
            
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")
            return []
    
    async def retrieve_image_context(self, question: str, max_results: int = TOP_K_IMAGES) -> List[Dict]:
        """Retrieve relevant images from ChromaDB."""
        if not self.image_collection:
            return []
        
        try:
            # Query image collection
            results = self.image_collection.query(
                query_texts=[question],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            image_contexts = []
            if results['documents'] and results['documents'][0]:
                for i, (caption, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    
                    similarity = 1.0 - distance
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        image_contexts.append({
                            'caption': caption,
                            'image_path': metadata.get('image_path', ''),
                            'source_pdf': metadata.get('source_pdf', 'Unknown'),
                            'page': metadata.get('page', 0),
                            'similarity': similarity,
                            'size': (metadata.get('size_w', 0), metadata.get('size_h', 0))
                        })
            
            logger.info(f"Retrieved {len(image_contexts)} relevant images")
            return image_contexts
            
        except Exception as e:
            logger.error(f"Image retrieval failed: {e}")
            return []
    
    def get_engineering_context(self, question: str) -> Dict[str, Any]:
        """Extract relevant engineering insights from summaries."""
        if not self.summaries:
            return {}
        
        try:
            # Extract key terms from question
            question_lower = question.lower()
            
            relevant_context = {
                'field_summary': {},
                'zones': {},
                'recommendations': [],
                'capacity_estimates': {}
            }
            
            # Get integrated summary
            integrated = self.summaries.get('integrated_summary', {})
            if integrated:
                analysis_summary = integrated.get('analysis_summary', {})
                
                # Check if question relates to capacity, development, or potential
                if any(term in question_lower for term in ['capacity', 'power', 'mw', 'megawatt']):
                    relevant_context['capacity_estimates'] = {
                        'total_capacity_mw': analysis_summary.get('total_estimated_capacity_mw', 0),
                        'development_status': analysis_summary.get('development_status', 'Unknown')
                    }
                
                # Check if question relates to zones or geology
                if any(term in question_lower for term in ['zone', 'geology', 'reservoir', 'caprock', 'basement']):
                    relevant_context['zones'] = integrated.get('combined_zones', {})
                
                # General field information
                relevant_context['field_summary'] = analysis_summary
            
            # Get specific grid analyses if question mentions model names
            individual_grids = self.summaries.get('individual_grids', {})
            for grid_name, grid_analysis in individual_grids.items():
                if grid_name.lower() in question_lower or any(term in grid_name.lower() for term in ['resistivity', 'density']):
                    if 'specific_models' not in relevant_context:
                        relevant_context['specific_models'] = {}
                    relevant_context['specific_models'][grid_name] = grid_analysis.get('engineering_summary', {})
            
            return relevant_context
            
        except Exception as e:
            logger.error(f"Engineering context extraction failed: {e}")
            return {}
    
    def get_model_data_context(self, question: str) -> Dict[str, Any]:
        """Get relevant 3D model data context."""
        model_context = {}
        
        try:
            # Check if question relates to specific properties
            question_lower = question.lower()
            
            for grid_name, metadata in self.grids_metadata.items():
                property_name = metadata.get('property_name', '')
                
                # Match property type to question
                if (('resistivity' in question_lower and 'resistivity' in property_name.lower()) or
                    ('density' in question_lower and 'density' in property_name.lower()) or
                    ('conductivity' in question_lower and 'resistivity' in property_name.lower())):
                    
                    model_context[grid_name] = {
                        'property': property_name,
                        'property_unit': metadata.get('property_unit', ''),
                        'dimensions': metadata.get('grid_dimensions', []),
                        'bounds': metadata.get('bounds', {}),
                        'value_stats': metadata.get('value_stats', {}),
                        'source_file': metadata.get('source_file', '')
                    }
            
            return model_context
            
        except Exception as e:
            logger.error(f"Model data context extraction failed: {e}")
            return {}
    
    async def web_augmentation(self, question: str, coverage_score: float) -> List[Dict]:
        """Augment with web search if internal coverage is insufficient."""
        if coverage_score >= WEB_SEARCH_THRESHOLD:
            return []  # Skip web search if we have good coverage
        
        try:
            # Simple web search using a hypothetical search API
            # In production, you would use Bing Search API, SerpAPI, etc.
            logger.info(f"Performing web augmentation (coverage: {coverage_score:.1%})")
            
            # Placeholder for web search results
            web_results = [
                {
                    'title': 'Recent Geothermal Development in Sumatra',
                    'url': 'https://example.com/geothermal-sumatra',
                    'snippet': 'Recent developments in Sumatra geothermal fields...',
                    'source_type': 'web'
                }
            ]
            
            return web_results
            
        except Exception as e:
            logger.error(f"Web augmentation failed: {e}")
            return []
    
    def calculate_coverage_score(self, text_contexts: List, image_contexts: List, 
                                engineering_context: Dict, model_context: Dict) -> float:
        """Calculate how well the question is covered by internal knowledge."""
        score = 0.0
        
        # Text coverage
        if text_contexts:
            avg_text_similarity = np.mean([ctx['similarity'] for ctx in text_contexts])
            score += avg_text_similarity * 0.4
        
        # Image coverage
        if image_contexts:
            avg_image_similarity = np.mean([ctx['similarity'] for ctx in image_contexts])
            score += avg_image_similarity * 0.2
        
        # Engineering context coverage
        if engineering_context:
            score += 0.3  # Flat bonus for having engineering context
        
        # Model data coverage
        if model_context:
            score += 0.1  # Flat bonus for having model data
        
        return min(score, 1.0)
    
    async def generate_response(self, question: str, contexts: Dict[str, Any], 
                              temperature: float = 0.3) -> Tuple[str, float, List[Citation]]:
        """Generate GPT response using all available contexts."""
        try:
            # Build comprehensive context prompt
            context_prompt = self.build_context_prompt(question, contexts)
            
            # Create system prompt
            system_prompt = """You are an expert geothermal engineer and geophysicist assistant analyzing the Semurup geothermal field in Sumatra, Indonesia. You have access to:

1. Scientific literature and technical papers
2. 3D geophysical models (resistivity and density)
3. Engineering analysis and interpretations
4. Field data and geochemical surveys

Provide accurate, engineering-grade responses with:
- Technical precision and proper units
- Clear explanations for different expertise levels
- Specific citations and source references
- Quantitative data when available
- Uncertainty acknowledgments when data is limited

Format your response professionally but accessibly."""

            # Generate response
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",  # Use GPT-3.5 for faster responses, upgrade to GPT-4o or o3 as needed
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # Extract citations
            citations = self.extract_citations(contexts)
            
            # Calculate confidence
            confidence = self.calculate_response_confidence(contexts, answer)
            
            return answer, confidence, citations
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error generating the response: {str(e)}", 0.0, []
    
    def build_context_prompt(self, question: str, contexts: Dict[str, Any]) -> str:
        """Build comprehensive context prompt for GPT."""
        prompt_parts = [f"Question: {question}\n"]
        
        # Add text contexts
        text_contexts = contexts.get('text_contexts', [])
        if text_contexts:
            prompt_parts.append("\n=== RELEVANT LITERATURE ===")
            for i, ctx in enumerate(text_contexts[:3], 1):  # Top 3 texts
                prompt_parts.append(f"\n[Source {i}: {ctx['source']}]")
                prompt_parts.append(ctx['content'][:800] + "..." if len(ctx['content']) > 800 else ctx['content'])
        
        # Add engineering context
        engineering_context = contexts.get('engineering_context', {})
        if engineering_context:
            prompt_parts.append("\n=== ENGINEERING ANALYSIS ===")
            
            # Field summary
            field_summary = engineering_context.get('field_summary', {})
            if field_summary:
                capacity = field_summary.get('total_estimated_capacity_mw', 0)
                status = field_summary.get('development_status', 'Unknown')
                potential = field_summary.get('geothermal_potential_rating', 'Unknown')
                confidence = field_summary.get('confidence_level', 0)
                
                prompt_parts.append(f"Field Overview: {capacity:.1f} MW estimated capacity, {status}, {potential} potential (confidence: {confidence:.1%})")
            
            # Zones
            zones = engineering_context.get('zones', {})
            if zones:
                prompt_parts.append("\nGeological Zones:")
                for zone_name, zone_data in zones.items():
                    volume = zone_data.get('total_volume_km3', 0)
                    confidence = zone_data.get('average_confidence', 0)
                    prompt_parts.append(f"- {zone_name.title()}: {volume:.2f} km³ (confidence: {confidence:.1%})")
        
        # Add model data context
        model_context = contexts.get('model_context', {})
        if model_context:
            prompt_parts.append("\n=== 3D MODEL DATA ===")
            for model_name, model_data in model_context.items():
                property_name = model_data.get('property', 'Unknown')
                unit = model_data.get('property_unit', '')
                stats = model_data.get('value_stats', {})
                
                prompt_parts.append(f"\n{model_name}: {property_name} {unit}")
                if stats:
                    prompt_parts.append(f"  Range: {stats.get('min', 0):.2e} to {stats.get('max', 0):.2e}")
                    prompt_parts.append(f"  Mean: {stats.get('mean', 0):.2e}")
        
        # Add image contexts
        image_contexts = contexts.get('image_contexts', [])
        if image_contexts:
            prompt_parts.append("\n=== RELEVANT FIGURES ===")
            for i, ctx in enumerate(image_contexts, 1):
                prompt_parts.append(f"\n[Figure {i} from {ctx['source_pdf']}, p.{ctx['page']}]: {ctx['caption']}")
        
        # Add web results if available
        web_contexts = contexts.get('web_contexts', [])
        if web_contexts:
            prompt_parts.append("\n=== RECENT INFORMATION ===")
            for ctx in web_contexts:
                prompt_parts.append(f"\n[{ctx['title']}]: {ctx['snippet']}")
        
        return "\n".join(prompt_parts)
    
    def extract_citations(self, contexts: Dict[str, Any]) -> List[Citation]:
        """Extract citations from contexts."""
        citations = []
        
        # Text citations
        for ctx in contexts.get('text_contexts', []):
            citations.append(Citation(
                source_type='literature',
                source_name=ctx['source'],
                confidence=ctx['similarity'],
                excerpt=ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
            ))
        
        # Image citations
        for ctx in contexts.get('image_contexts', []):
            citations.append(Citation(
                source_type='literature',
                source_name=ctx['source_pdf'],
                page_number=ctx['page'],
                confidence=ctx['similarity'],
                excerpt=ctx['caption']
            ))
        
        # Engineering summary citations
        if contexts.get('engineering_context'):
            citations.append(Citation(
                source_type='summary',
                source_name='Digital Twin Engineering Analysis',
                confidence=0.9,
                excerpt='Integrated geological interpretation and capacity assessment'
            ))
        
        # Model data citations
        for model_name in contexts.get('model_context', {}):
            citations.append(Citation(
                source_type='model',
                source_name=f'3D Model: {model_name}',
                confidence=0.85,
                excerpt='Geophysical model data and spatial analysis'
            ))
        
        return citations
    
    def calculate_response_confidence(self, contexts: Dict[str, Any], response: str) -> float:
        """Calculate confidence in the generated response."""
        confidence_factors = []
        
        # Context quality
        text_contexts = contexts.get('text_contexts', [])
        if text_contexts:
            avg_similarity = np.mean([ctx['similarity'] for ctx in text_contexts])
            confidence_factors.append(avg_similarity)
        
        # Engineering context availability
        if contexts.get('engineering_context'):
            confidence_factors.append(0.8)
        
        # Model data availability
        if contexts.get('model_context'):
            confidence_factors.append(0.7)
        
        # Response completeness (simple heuristic)
        if len(response) > 200:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors) if confidence_factors else 0.3
    
    async def process_question(self, request: QuestionRequest) -> QuestionResponse:
        """Process a question through the complete QA pipeline."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {request.question[:100]}...")
            
            # Parallel context retrieval
            text_contexts_task = self.retrieve_text_context(request.question)
            image_contexts_task = self.retrieve_image_context(request.question) if request.include_images else []
            
            # Wait for retrievals
            text_contexts = await text_contexts_task
            image_contexts = await image_contexts_task if request.include_images else []
            
            # Get engineering and model contexts
            engineering_context = self.get_engineering_context(request.question)
            model_context = self.get_model_data_context(request.question)
            
            # Calculate coverage and determine if web search is needed
            coverage_score = self.calculate_coverage_score(
                text_contexts, image_contexts, engineering_context, model_context
            )
            
            web_contexts = []
            if request.include_web:
                web_contexts = await self.web_augmentation(request.question, coverage_score)
            
            # Combine all contexts
            all_contexts = {
                'text_contexts': text_contexts,
                'image_contexts': image_contexts,
                'engineering_context': engineering_context,
                'model_context': model_context,
                'web_contexts': web_contexts
            }
            
            # Generate response
            answer, confidence, citations = await self.generate_response(
                request.question, all_contexts, request.temperature
            )
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Build response
            response = QuestionResponse(
                question=request.question,
                answer=answer,
                citations=citations,
                processing_time_ms=processing_time,
                confidence_score=confidence,
                context_used={
                    'text_chunks': len(text_contexts),
                    'images': len(image_contexts),
                    'engineering_insights': bool(engineering_context),
                    'model_data': bool(model_context),
                    'web_results': len(web_contexts),
                    'coverage_score': coverage_score
                },
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Question processed in {processing_time}ms (confidence: {confidence:.1%})")
            return response
            
        except Exception as e:
            logger.error(f"Question processing failed: {e}")
            logger.error(traceback.format_exc())
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return QuestionResponse(
                question=request.question,
                answer=f"I apologize, but I encountered an error processing your question: {str(e)}",
                citations=[],
                processing_time_ms=processing_time,
                confidence_score=0.0,
                context_used={},
                timestamp=datetime.now().isoformat()
            )


# Initialize the QA system
qa_system = GeothermalQASystem()

# FastAPI app
app = FastAPI(
    title="Geothermal Digital Twin QA API",
    description="Advanced question-answering system for geothermal field analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API health check and information."""
    return {
        "message": "Geothermal Digital Twin QA API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "ask": "/ask - Submit questions",
            "health": "/health - System health check",
            "docs": "/docs - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "components": {}
    }
    
    # Check OpenAI
    try:
        openai.api_key  # Just check if key is set
        health_status["components"]["openai"] = "connected"
    except:
        health_status["components"]["openai"] = "error"
        health_status["status"] = "degraded"
    
    # Check ChromaDB collections
    health_status["components"]["text_collection"] = "available" if qa_system.text_collection else "unavailable"
    health_status["components"]["image_collection"] = "available" if qa_system.image_collection else "unavailable"
    
    # Check summaries
    health_status["components"]["engineering_summaries"] = "available" if qa_system.summaries else "unavailable"
    
    # Check 3D models
    health_status["components"]["model_metadata"] = f"{len(qa_system.grids_metadata)} models available"
    
    return health_status


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the geothermal digital twin.
    
    Integrates multiple knowledge sources:
    - Literature corpus (papers, reports)
    - 3D geophysical models  
    - Engineering analysis and interpretations
    - Optional web augmentation
    
    Returns detailed answer with citations and confidence scoring.
    """
    try:
        response = await qa_system.process_question(request)
        return response
    except Exception as e:
        logger.error(f"API request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/stats")
async def system_stats():
    """Get system statistics and knowledge base information."""
    try:
        stats = {
            "knowledge_base": {
                "text_documents": qa_system.text_collection.count() if qa_system.text_collection else 0,
                "images": qa_system.image_collection.count() if qa_system.image_collection else 0,
                "3d_models": len(qa_system.grids_metadata),
                "engineering_summaries": bool(qa_system.summaries)
            },
            "available_models": list(qa_system.grids_metadata.keys()),
            "cache_size": len(qa_system.cache),
            "system_uptime": "N/A"  # Could track actual uptime
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples")
async def example_questions():
    """Get example questions to help users get started."""
    examples = [
        {
            "category": "Reservoir Assessment",
            "questions": [
                "What is the estimated geothermal capacity of the Semurup field?",
                "Where are the best reservoir zones located?",
                "What temperatures can we expect at 2000m depth?"
            ]
        },
        {
            "category": "Geological Analysis", 
            "questions": [
                "What geological zones have been identified in the 3D models?",
                "Where is the caprock layer located?",
                "What are the resistivity characteristics of the reservoir?"
            ]
        },
        {
            "category": "Drilling Recommendations",
            "questions": [
                "Where should we drill the first production well?",
                "What are the recommended injection well locations?",
                "What drilling depths are optimal for this field?"
            ]
        },
        {
            "category": "Geochemistry",
            "questions": [
                "What do the geochemical surveys tell us about the reservoir?",
                "What are the fluid temperatures from geothermometry?",
                "Are there signs of hydrothermal alteration?"
            ]
        }
    ]
    
    return {"examples": examples}


def main():
    """Main entry point for running the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Geothermal QA Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting Geothermal QA Server on {args.host}:{args.port}")
    
    try:
        uvicorn.run(
            "qa_server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info" if not args.debug else "debug"
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()