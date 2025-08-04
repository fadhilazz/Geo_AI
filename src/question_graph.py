#!/usr/bin/env python3
"""
Question Graph Module for Geothermal Digital Twin AI

Advanced question routing and classification system using semantic graphs.
Creates a knowledge graph of geothermal topics with intelligent question routing,
similarity search, and query optimization.

Features:
1. Semantic question graph with NetworkX
2. Ultra-fast similarity search with FAISS
3. Intelligent question routing and expansion
4. Topic clustering and classification
5. Related question suggestions
6. Query optimization and caching
7. Machine learning-based intent recognition

Dependencies:
- networkx: Graph operations and analysis
- faiss: High-performance similarity search
- openai: Embeddings generation
- scikit-learn: Clustering and classification
- numpy, scipy: Numerical operations

Usage:
    python src/question_graph.py [--build] [--update] [--query "question"]
"""

import os
import sys
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import hashlib

# Graph and similarity search
import networkx as nx
import faiss
import numpy as np

# Machine learning
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# AI embeddings
import openai

# Progress tracking
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("digital_twin/cache")
GRAPH_FILE = CACHE_DIR / "question_graph.pkl"
FAISS_INDEX_FILE = CACHE_DIR / "question_index.faiss"
EMBEDDINGS_FILE = CACHE_DIR / "question_embeddings.npy"
METADATA_FILE = CACHE_DIR / "question_metadata.json"
TOPICS_FILE = CACHE_DIR / "geothermal_topics.json"

# Configuration
EMBEDDING_DIMENSION = 1536  # OpenAI ada-002 embedding dimension
MAX_QUESTIONS_PER_TOPIC = 50
SIMILARITY_THRESHOLD = 0.75
TOP_K_SIMILAR = 10


class GeothermalQuestionGraph:
    """Advanced question graph for geothermal domain knowledge."""
    
    def __init__(self):
        """Initialize the question graph system."""
        self.setup_directories()
        self.graph = nx.DiGraph()
        self.faiss_index = None
        self.question_embeddings = None
        self.question_metadata = {}
        self.topics = {}
        self.topic_clusters = {}
        
        # ML components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.intent_classifier = None
        
        # Cache for performance
        self.embedding_cache = {}
        self.similarity_cache = {}
        
        logger.info("Question Graph system initialized")
    
    def setup_directories(self):
        """Create required directories."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def setup_openai(self):
        """Initialize OpenAI for embeddings."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = api_key
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text with caching."""
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    
    def define_geothermal_topics(self) -> Dict[str, Dict]:
        """Define comprehensive geothermal domain topics and questions."""
        topics = {
            'reservoir_assessment': {
                'name': 'Reservoir Assessment',
                'description': 'Geothermal reservoir characterization and evaluation',
                'keywords': ['reservoir', 'capacity', 'temperature', 'permeability', 'porosity'],
                'sample_questions': [
                    "What is the estimated geothermal capacity of the field?",
                    "What are the reservoir temperatures at different depths?",
                    "How permeable is the reservoir rock?",
                    "What is the reservoir pressure and fluid composition?",
                    "What are the heat flow characteristics?",
                    "How large is the reservoir volume?",
                    "What is the sustainable production rate?",
                    "What are the reservoir recharge mechanisms?",
                    "How does reservoir permeability vary spatially?",
                    "What are the fluid flow patterns in the reservoir?"
                ],
                'priority': 1.0,
                'complexity': 'high'
            },
            
            'geological_zones': {
                'name': 'Geological Zones',
                'description': 'Identification and characterization of geological structures',
                'keywords': ['geology', 'zones', 'caprock', 'basement', 'fractures', 'alteration'],
                'sample_questions': [
                    "What geological zones have been identified?",
                    "Where is the caprock layer located?",
                    "How thick is the caprock seal?",
                    "What are the characteristics of the basement rocks?",
                    "Where are the main fracture zones?",
                    "What evidence exists for hydrothermal alteration?",
                    "What is the structural geology of the area?",
                    "How do fault systems affect fluid flow?",
                    "What are the rock types in each zone?",
                    "How does geological structure control permeability?"
                ],
                'priority': 0.9,
                'complexity': 'high'
            },
            
            'drilling_operations': {
                'name': 'Drilling Operations',
                'description': 'Well planning, drilling targets, and operational considerations',
                'keywords': ['drilling', 'wells', 'production', 'injection', 'targets', 'locations'],
                'sample_questions': [
                    "Where should we drill the first production well?",
                    "What are the recommended injection well locations?",
                    "What drilling depths are optimal?",
                    "What drilling hazards should we expect?",
                    "How should wells be completed?",
                    "What casing programs are recommended?",
                    "What are the expected drilling costs?",
                    "How many wells are needed for field development?",
                    "What drilling fluid should be used?",
                    "What are the optimal well trajectories?"
                ],
                'priority': 0.8,
                'complexity': 'medium'
            },
            
            'geophysical_data': {
                'name': 'Geophysical Data',
                'description': 'Geophysical surveys and 3D model interpretation',
                'keywords': ['geophysics', 'resistivity', 'density', 'magnetotelluric', '3d model'],
                'sample_questions': [
                    "What do the resistivity models tell us?",
                    "How do we interpret the MT data?",
                    "What are the density contrasts in the subsurface?",
                    "Where are the conductive zones?",
                    "What geophysical anomalies are present?",
                    "How reliable are the 3D models?",
                    "What is the resolution of the geophysical data?",
                    "How do geophysical properties relate to geology?",
                    "What are the limitations of the survey?",
                    "How do we validate geophysical interpretations?"
                ],
                'priority': 0.7,
                'complexity': 'high'
            },
            
            'geochemistry': {
                'name': 'Geochemistry',
                'description': 'Fluid geochemistry and geothermometry analysis',
                'keywords': ['geochemistry', 'fluid', 'temperature', 'composition', 'geothermometry'],
                'sample_questions': [
                    "What do geochemical surveys reveal about temperatures?",
                    "What is the fluid composition of hot springs?",
                    "What geothermometers are most reliable?",
                    "Are there signs of fluid mixing?",
                    "What are the gas compositions?",
                    "How do fluid chemistries vary spatially?",
                    "What do isotope analyses indicate?",
                    "Are there corrosive components in the fluids?",
                    "What are the scaling tendencies?",
                    "How do fluid chemistries change over time?"
                ],
                'priority': 0.6,
                'complexity': 'medium'
            },
            
            'field_development': {
                'name': 'Field Development',
                'description': 'Economic assessment and development planning',
                'keywords': ['development', 'economics', 'planning', 'power plant', 'infrastructure'],
                'sample_questions': [
                    "What is the development timeline for the field?",
                    "What are the estimated development costs?",
                    "What power plant size is optimal?",
                    "What infrastructure is required?",
                    "What are the economic risks?",
                    "What is the projected power output?",
                    "What are the environmental considerations?",
                    "What permits and approvals are needed?",
                    "What are the financing options?",
                    "What is the expected project IRR?"
                ],
                'priority': 0.5,
                'complexity': 'medium'
            },
            
            'environmental_impact': {
                'name': 'Environmental Impact',
                'description': 'Environmental considerations and impact assessment',  
                'keywords': ['environment', 'impact', 'sustainability', 'emissions', 'water'],
                'sample_questions': [
                    "What are the environmental impacts of development?",
                    "How can we minimize ecological disruption?",
                    "What are the water resource implications?",
                    "What are the induced seismicity risks?",
                    "How do we handle waste fluids?",
                    "What are the air quality impacts?",
                    "How do we protect local ecosystems?",
                    "What monitoring is required?",
                    "What mitigation measures are needed?",
                    "How do we ensure sustainable operation?"
                ],
                'priority': 0.4,
                'complexity': 'medium'
            },
            
            'risk_assessment': {
                'name': 'Risk Assessment',
                'description': 'Technical and commercial risk evaluation',
                'keywords': ['risk', 'uncertainty', 'probability', 'technical', 'commercial'],
                'sample_questions': [
                    "What are the main technical risks?",
                    "How do we quantify resource uncertainty?",
                    "What are the drilling risks?",
                    "What commercial risks should be considered?",
                    "How do we assess reservoir performance risk?",
                    "What are the regulatory risks?",
                    "How do we mitigate development risks?",
                    "What contingency plans are needed?",
                    "How do risks change over project phases?",
                    "What insurance coverage is recommended?"
                ],
                'priority': 0.3,
                'complexity': 'high'
            }
        }
        
        return topics
    
    def build_question_graph(self) -> nx.DiGraph:
        """Build comprehensive question graph with topics and relationships."""
        logger.info("Building question graph...")
        
        # Get topic definitions
        topics = self.define_geothermal_topics()
        
        # Add topic nodes
        for topic_id, topic_data in topics.items():
            self.graph.add_node(
                topic_id,
                type='topic',
                name=topic_data['name'],
                description=topic_data['description'],
                keywords=topic_data['keywords'],
                priority=topic_data['priority'],
                complexity=topic_data['complexity']
            )
            
            # Add question nodes for each topic
            for i, question in enumerate(topic_data['sample_questions']):
                question_id = f"{topic_id}_q{i+1}"
                
                self.graph.add_node(
                    question_id,
                    type='question',
                    text=question,
                    topic=topic_id,
                    complexity=topic_data['complexity'],
                    evaluation=True if i < 3 else False  # Mark first 3 as evaluation questions
                )
                
                # Connect question to topic
                self.graph.add_edge(topic_id, question_id, relationship='contains')
        
        # Add topic relationships based on keyword overlap
        topic_nodes = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'topic']
        
        for i, topic1 in enumerate(topic_nodes):
            for topic2 in topic_nodes[i+1:]:
                keywords1 = set(self.graph.nodes[topic1]['keywords'])
                keywords2 = set(self.graph.nodes[topic2]['keywords'])
                
                # Calculate keyword overlap
                overlap = len(keywords1.intersection(keywords2))
                similarity = overlap / len(keywords1.union(keywords2)) if keywords1.union(keywords2) else 0
                
                if similarity > 0.2:  # Add edge if significant overlap
                    self.graph.add_edge(topic1, topic2, 
                                      relationship='related',
                                      similarity=similarity,
                                      overlap=overlap)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def build_faiss_index(self):
        """Build FAISS index for ultra-fast similarity search."""
        logger.info("Building FAISS index...")
        
        # Get all questions
        question_nodes = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'question']
        
        if not question_nodes:
            logger.warning("No questions found for indexing")
            return
        
        # Setup OpenAI
        self.setup_openai()
        
        # Generate embeddings for all questions
        embeddings = []
        metadata = []
        
        with tqdm(question_nodes, desc="Generating embeddings") as pbar:
            for question_id in pbar:
                question_text = self.graph.nodes[question_id]['text']
                pbar.set_description(f"Embedding: {question_text[:50]}...")
                
                embedding = self.get_embedding(question_text)
                embeddings.append(embedding)
                
                metadata.append({
                    'id': question_id,
                    'text': question_text,
                    'topic': self.graph.nodes[question_id]['topic'],
                    'complexity': self.graph.nodes[question_id]['complexity'],
                    'evaluation': self.graph.nodes[question_id]['evaluation']
                })
        
        # Convert to numpy array
        embeddings_array = np.vstack(embeddings).astype(np.float32)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        index.add(embeddings_array)
        
        # Store components
        self.faiss_index = index
        self.question_embeddings = embeddings_array
        self.question_metadata = {i: meta for i, meta in enumerate(metadata)}
        
        logger.info(f"FAISS index built with {index.ntotal} questions")
    
    def find_similar_questions(self, query: str, k: int = TOP_K_SIMILAR) -> List[Dict]:
        """Find similar questions using FAISS index."""
        if not self.faiss_index:
            logger.warning("FAISS index not available")
            return []
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query).reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search index
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.question_metadata):
                    metadata = self.question_metadata[idx]
                    results.append({
                        'question_id': metadata['id'],
                        'text': metadata['text'],
                        'topic': metadata['topic'],
                        'similarity': float(score),
                        'complexity': metadata['complexity'],
                        'evaluation': metadata['evaluation']
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def classify_question_intent(self, question: str) -> Dict[str, Any]:
        """Classify question intent and extract key information."""
        question_lower = question.lower()
        
        # Intent classification rules
        intent_patterns = {
            'capacity_inquiry': ['capacity', 'power', 'mw', 'megawatt', 'generation'],
            'location_inquiry': ['where', 'location', 'drill', 'site', 'target'],
            'depth_inquiry': ['depth', 'deep', 'how deep', 'drilling depth'],
            'temperature_inquiry': ['temperature', 'hot', 'heat', 'thermal'],
            'geological_inquiry': ['geology', 'rock', 'formation', 'zone', 'caprock'],
            'technical_inquiry': ['how', 'what', 'explain', 'describe', 'technical'],
            'risk_inquiry': ['risk', 'uncertainty', 'problem', 'challenge'],
            'economic_inquiry': ['cost', 'economics', 'price', 'investment', 'irr'],
            'environmental_inquiry': ['environment', 'impact', 'sustainability', 'emission']
        }
        
        # Score each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in question_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Determine primary intent
        primary_intent = max(intent_scores.keys(), key=intent_scores.get) if intent_scores else 'general_inquiry'
        
        # Extract question type
        question_type = 'what'  # default
        if question_lower.startswith(('where', 'which location')):
            question_type = 'where'
        elif question_lower.startswith(('how', 'how to')):
            question_type = 'how'
        elif question_lower.startswith('why'):
            question_type = 'why'
        elif question_lower.startswith('when'):
            question_type = 'when'
        elif '?' not in question:
            question_type = 'statement'
        
        # Extract complexity indicators
        complexity_indicators = {
            'high': ['complex', 'detailed', 'comprehensive', 'analysis', 'modeling'],
            'medium': ['explain', 'describe', 'compare', 'evaluate'],
            'low': ['what is', 'simple', 'basic', 'quick']
        }
        
        complexity = 'medium'  # default
        for level, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                complexity = level
                break
        
        return {
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'question_type': question_type,
            'complexity': complexity,
            'requires_quantitative': any(term in question_lower 
                                       for term in ['how much', 'how many', 'capacity', 'temperature', 'depth']),
            'requires_spatial': any(term in question_lower 
                                  for term in ['where', 'location', 'map', 'zone']),
            'requires_literature': any(term in question_lower 
                                     for term in ['literature', 'studies', 'research', 'papers'])
        }
    
    def route_question(self, question: str) -> Dict[str, Any]:
        """Route question to appropriate processing pipeline."""
        # Classify intent
        intent_info = self.classify_question_intent(question)
        
        # Find similar questions
        similar_questions = self.find_similar_questions(question, k=5)
        
        # Determine best topic match
        topic_scores = {}
        for similar in similar_questions:
            topic = similar['topic']
            similarity = similar['similarity']
            
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(similarity)
        
        # Calculate average similarity per topic
        topic_rankings = {}
        for topic, scores in topic_scores.items():
            topic_rankings[topic] = {
                'average_similarity': np.mean(scores),
                'max_similarity': np.max(scores),
                'question_count': len(scores)
            }
        
        # Determine best topic
        best_topic = None
        if topic_rankings:
            best_topic = max(topic_rankings.keys(), 
                           key=lambda t: topic_rankings[t]['average_similarity'])
        
        # Generate routing recommendations
        routing = {
            'question': question,
            'intent_analysis': intent_info,
            'best_topic': best_topic,
            'topic_rankings': topic_rankings,
            'similar_questions': similar_questions,
            'processing_recommendations': {
                'use_literature': intent_info['requires_literature'] or intent_info['complexity'] == 'high',
                'use_3d_models': intent_info['requires_spatial'] or intent_info['requires_quantitative'],
                'use_engineering_summaries': intent_info['primary_intent'] in ['capacity_inquiry', 'technical_inquiry'],
                'response_depth': intent_info['complexity'],
                'include_visualizations': intent_info['requires_spatial']
            },
            'estimated_processing_time': self.estimate_processing_time(intent_info),
            'confidence': self.calculate_routing_confidence(similar_questions, topic_rankings)
        }
        
        return routing
    
    def estimate_processing_time(self, intent_info: Dict) -> float:
        """Estimate processing time based on question complexity."""
        base_time = 2.0  # seconds
        
        complexity_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        }
        
        multiplier = complexity_multipliers.get(intent_info['complexity'], 1.0)
        
        # Add time for special requirements
        if intent_info['requires_quantitative']:
            multiplier += 0.5
        if intent_info['requires_spatial']:
            multiplier += 0.3
        if intent_info['requires_literature']:
            multiplier += 0.7
        
        return base_time * multiplier
    
    def calculate_routing_confidence(self, similar_questions: List, topic_rankings: Dict) -> float:
        """Calculate confidence in question routing."""
        if not similar_questions:
            return 0.3
        
        # Base confidence on top similarity score
        top_similarity = similar_questions[0]['similarity'] if similar_questions else 0.0
        
        # Boost confidence if multiple similar questions exist
        similarity_boost = min(len(similar_questions) * 0.1, 0.3)
        
        # Boost confidence if topic is well-represented
        topic_boost = 0.0
        if topic_rankings:
            best_topic_data = max(topic_rankings.values(), key=lambda x: x['average_similarity'])
            if best_topic_data['question_count'] >= 3:
                topic_boost = 0.2
        
        confidence = min(top_similarity + similarity_boost + topic_boost, 1.0)
        return confidence
    
    def suggest_related_questions(self, question: str, n_suggestions: int = 5) -> List[str]:
        """Suggest related questions based on similarity and topic."""
        routing = self.route_question(question)
        
        suggestions = []
        
        # Get questions from same topic
        best_topic = routing['best_topic']
        if best_topic and best_topic in self.topics:
            topic_questions = self.topics[best_topic].get('sample_questions', [])
            suggestions.extend(topic_questions[:3])
        
        # Add similar questions
        similar_questions = routing['similar_questions']
        for similar in similar_questions:
            if similar['text'] not in suggestions:
                suggestions.append(similar['text'])
            
            if len(suggestions) >= n_suggestions:
                break
        
        return suggestions[:n_suggestions]
    
    def analyze_question_trends(self) -> Dict[str, Any]:
        """Analyze question patterns and trends."""
        question_nodes = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'question']
        
        # Topic distribution
        topic_counts = {}
        complexity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for question_id in question_nodes:
            question_data = self.graph.nodes[question_id]
            topic = question_data['topic']
            complexity = question_data['complexity']
            
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            complexity_counts[complexity] += 1
        
        # Graph metrics
        graph_metrics = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'total_questions': len(question_nodes),
            'total_topics': len([n for n, d in self.graph.nodes(data=True) if d['type'] == 'topic']),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected())
        }
        
        return {
            'topic_distribution': topic_counts,
            'complexity_distribution': complexity_counts,
            'graph_metrics': graph_metrics,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def save_graph(self):
        """Save question graph and indices to disk."""
        logger.info("Saving question graph...")
        
        try:
            # Save NetworkX graph
            with open(GRAPH_FILE, 'wb') as f:
                pickle.dump(self.graph, f)
            
            # Save FAISS index
            if self.faiss_index:
                faiss.write_index(self.faiss_index, str(FAISS_INDEX_FILE))
            
            # Save embeddings
            if self.question_embeddings is not None:
                np.save(EMBEDDINGS_FILE, self.question_embeddings)
            
            # Save metadata
            with open(METADATA_FILE, 'w') as f:
                json.dump(self.question_metadata, f, indent=2)
            
            # Save topics
            with open(TOPICS_FILE, 'w') as f:
                json.dump(self.topics, f, indent=2)
            
            logger.info("Question graph saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save question graph: {e}")
    
    def load_graph(self) -> bool:
        """Load question graph and indices from disk."""
        try:
            # Load NetworkX graph
            if GRAPH_FILE.exists():
                with open(GRAPH_FILE, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info("Question graph loaded")
            else:
                logger.warning("No saved graph found")
                return False
            
            # Load FAISS index
            if FAISS_INDEX_FILE.exists():
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
                logger.info("FAISS index loaded")
            
            # Load embeddings
            if EMBEDDINGS_FILE.exists():
                self.question_embeddings = np.load(EMBEDDINGS_FILE)
                logger.info("Question embeddings loaded")
            
            # Load metadata
            if METADATA_FILE.exists():
                with open(METADATA_FILE, 'r') as f:
                    metadata = json.load(f)
                    # Convert string keys back to integers
                    self.question_metadata = {int(k): v for k, v in metadata.items()}
                logger.info("Question metadata loaded")
            
            # Load topics
            if TOPICS_FILE.exists():
                with open(TOPICS_FILE, 'r') as f:
                    self.topics = json.load(f)
                logger.info("Topics loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load question graph: {e}")
            return False
    
    def build_complete_system(self):
        """Build the complete question graph system."""
        logger.info("=== Building Complete Question Graph System ===")
        
        # Build graph
        self.build_question_graph()
        
        # Store topics
        self.topics = self.define_geothermal_topics()
        
        # Build FAISS index
        self.build_faiss_index()
        
        # Save everything
        self.save_graph()
        
        # Generate analysis
        analysis = self.analyze_question_trends()
        
        logger.info("=== Question Graph System Complete ===")
        logger.info(f"Total questions: {analysis['graph_metrics']['total_questions']}")
        logger.info(f"Total topics: {analysis['graph_metrics']['total_topics']}")
        logger.info(f"FAISS index size: {self.faiss_index.ntotal if self.faiss_index else 0}")
        
        return analysis


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Question Graph System")
    parser.add_argument("--build", action="store_true", help="Build complete question graph")
    parser.add_argument("--update", action="store_true", help="Update existing graph")
    parser.add_argument("--query", type=str, help="Query the graph with a question")
    parser.add_argument("--analyze", action="store_true", help="Analyze question trends")
    parser.add_argument("--similar", type=str, help="Find similar questions")
    
    args = parser.parse_args()
    
    # Initialize system
    qg = GeothermalQuestionGraph()
    
    try:
        if args.build:
            # Build complete system
            analysis = qg.build_complete_system()
            
            print(f"\n=== Question Graph Built ===")
            print(f"Questions: {analysis['graph_metrics']['total_questions']}")
            print(f"Topics: {analysis['graph_metrics']['total_topics']}")
            print(f"Graph density: {analysis['graph_metrics']['density']:.3f}")
            
            print(f"\nTopic Distribution:")
            for topic, count in analysis['topic_distribution'].items():
                print(f"  {topic}: {count} questions")
        
        elif args.update:
            # Load and update
            if qg.load_graph():
                qg.build_faiss_index()  # Rebuild index
                qg.save_graph()
                print("Question graph updated successfully")
            else:
                print("No existing graph found - use --build first")
        
        elif args.query:
            # Query routing
            if not qg.load_graph():
                print("No graph found - use --build first")
                return
            
            routing = qg.route_question(args.query)
            
            print(f"\n=== Question Routing Analysis ===")
            print(f"Question: {args.query}")
            print(f"Primary Intent: {routing['intent_analysis']['primary_intent']}")
            print(f"Question Type: {routing['intent_analysis']['question_type']}")
            print(f"Complexity: {routing['intent_analysis']['complexity']}")
            print(f"Best Topic: {routing['best_topic']}")
            print(f"Routing Confidence: {routing['confidence']:.1%}")
            print(f"Estimated Processing Time: {routing['estimated_processing_time']:.1f}s")
            
            print(f"\nProcessing Recommendations:")
            recs = routing['processing_recommendations']
            for key, value in recs.items():
                print(f"  {key}: {value}")
            
            print(f"\nSimilar Questions:")
            for i, similar in enumerate(routing['similar_questions'][:3], 1):
                print(f"  {i}. {similar['text']} (similarity: {similar['similarity']:.2f})")
        
        elif args.similar:
            # Find similar questions
            if not qg.load_graph():
                print("No graph found - use --build first")
                return
            
            similar = qg.find_similar_questions(args.similar, k=10)
            
            print(f"\n=== Similar Questions ===")
            print(f"Query: {args.similar}")
            print(f"Found {len(similar)} similar questions:")
            
            for i, q in enumerate(similar, 1):
                print(f"{i:2d}. {q['text']}")
                print(f"     Topic: {q['topic']} | Similarity: {q['similarity']:.3f}")
        
        elif args.analyze:
            # Analyze trends
            if not qg.load_graph():
                print("No graph found - use --build first")
                return
            
            analysis = qg.analyze_question_trends()
            
            print(f"\n=== Question Graph Analysis ===")
            print(f"Total Questions: {analysis['graph_metrics']['total_questions']}")
            print(f"Total Topics: {analysis['graph_metrics']['total_topics']}")
            print(f"Graph Density: {analysis['graph_metrics']['density']:.3f}")
            
            print(f"\nTopic Distribution:")
            for topic, count in sorted(analysis['topic_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"  {topic}: {count}")
            
            print(f"\nComplexity Distribution:")
            for complexity, count in analysis['complexity_distribution'].items():
                print(f"  {complexity}: {count}")
        
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()