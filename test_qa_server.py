#!/usr/bin/env python3
"""
Test script for QA Server

Tests the complete question-answering system including:
1. System initialization and health checks
2. Context retrieval from all sources
3. Question processing and response generation
4. API endpoints and functionality
5. Integration with all knowledge sources

"""

import sys
import asyncio
import json
from pathlib import Path
import logging
import requests
import time
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qa_server import GeothermalQASystem, QuestionRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QAServerTester:
    """Comprehensive QA server testing."""
    
    def __init__(self):
        """Initialize the tester."""
        self.qa_system = None
        self.server_url = "http://127.0.0.1:8000"
        self.test_questions = [
            "What is the estimated geothermal capacity of the Semurup field?",
            "Where are the best reservoir zones located?",
            "What geological zones have been identified in the 3D models?",
            "What are the resistivity characteristics of the field?",
            "Where should we drill the first production well?"
        ]
    
    def test_system_initialization(self) -> bool:
        """Test QA system initialization."""
        print("=== Testing QA System Initialization ===\n")
        
        try:
            print("🔧 Initializing QA system...")
            self.qa_system = GeothermalQASystem()
            print("✅ QA system initialized successfully")
            
            # Check components
            components = {
                'Text Collection': self.qa_system.text_collection is not None,
                'Image Collection': self.qa_system.image_collection is not None,
                'Engineering Summaries': bool(self.qa_system.summaries),
                'Grid Metadata': bool(self.qa_system.grids_metadata)
            }
            
            print("\n📊 System Components:")
            for component, available in components.items():
                status = "✅ Available" if available else "⚠️  Not available"
                print(f"  {component}: {status}")
            
            # Show data counts
            if self.qa_system.text_collection:
                text_count = self.qa_system.text_collection.count()
                print(f"\n📄 Text Documents: {text_count:,}")
            
            if self.qa_system.image_collection:
                image_count = self.qa_system.image_collection.count()
                print(f"🖼️  Images: {image_count:,}")
            
            if self.qa_system.grids_metadata:
                print(f"🗺️  3D Models: {len(self.qa_system.grids_metadata)}")
                for model_name in self.qa_system.grids_metadata.keys():
                    print(f"    • {model_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    async def test_context_retrieval(self) -> bool:
        """Test context retrieval from all sources."""
        print("\n=== Testing Context Retrieval ===\n")
        
        if not self.qa_system:
            print("❌ QA system not initialized")
            return False
        
        try:
            test_question = "What is the geothermal capacity of Semurup field?"
            print(f"🧪 Testing with question: {test_question}")
            
            # Test text retrieval
            print("\n📄 Testing text context retrieval...")
            text_contexts = await self.qa_system.retrieve_text_context(test_question)
            print(f"  Retrieved {len(text_contexts)} text chunks")
            
            if text_contexts:
                for i, ctx in enumerate(text_contexts[:2], 1):
                    similarity = ctx['similarity']
                    source = ctx['source']
                    preview = ctx['content'][:100] + "..." if len(ctx['content']) > 100 else ctx['content']
                    print(f"    {i}. {source} (similarity: {similarity:.2f})")
                    print(f"       {preview}")
            
            # Test image retrieval
            print("\n🖼️  Testing image context retrieval...")
            image_contexts = await self.qa_system.retrieve_image_context(test_question)
            print(f"  Retrieved {len(image_contexts)} images")
            
            if image_contexts:
                for i, ctx in enumerate(image_contexts[:2], 1):
                    similarity = ctx['similarity']
                    source = ctx['source_pdf']
                    caption = ctx['caption']
                    print(f"    {i}. {source} p.{ctx['page']} (similarity: {similarity:.2f})")
                    print(f"       Caption: {caption}")
            
            # Test engineering context
            print("\n⚡ Testing engineering context retrieval...")
            engineering_context = self.qa_system.get_engineering_context(test_question)
            print(f"  Engineering context: {'Available' if engineering_context else 'Not available'}")
            
            if engineering_context:
                field_summary = engineering_context.get('field_summary', {})
                if field_summary:
                    capacity = field_summary.get('total_estimated_capacity_mw', 0)
                    status = field_summary.get('development_status', 'Unknown')
                    print(f"    Capacity: {capacity:.1f} MW")
                    print(f"    Status: {status}")
            
            # Test model data context
            print("\n🗺️  Testing model data context retrieval...")
            model_context = self.qa_system.get_model_data_context(test_question)
            print(f"  Model context: {len(model_context)} models matched")
            
            if model_context:
                for model_name, model_data in model_context.items():
                    property_name = model_data.get('property', 'Unknown')
                    print(f"    {model_name}: {property_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Context retrieval test failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    async def test_question_processing(self) -> bool:
        """Test complete question processing pipeline."""
        print("\n=== Testing Question Processing Pipeline ===\n")
        
        if not self.qa_system:
            print("❌ QA system not initialized")
            return False
        
        try:
            test_question = "What is the estimated geothermal capacity of the Semurup field and where are the best drilling locations?"
            print(f"🧪 Processing question: {test_question}")
            
            # Create request
            request = QuestionRequest(
                question=test_question,
                include_images=True,
                include_web=False,
                temperature=0.3
            )
            
            # Process question
            start_time = time.time()
            response = await self.qa_system.process_question(request)
            processing_time = time.time() - start_time
            
            print(f"\n✅ Question processed in {processing_time:.2f} seconds")
            print(f"🎯 Response confidence: {response.confidence_score:.1%}")
            
            # Display response
            print(f"\n📝 Answer ({len(response.answer)} characters):")
            print(response.answer[:500] + "..." if len(response.answer) > 500 else response.answer)
            
            # Display context used
            print(f"\n📊 Context Used:")
            context = response.context_used
            print(f"  Text chunks: {context.get('text_chunks', 0)}")
            print(f"  Images: {context.get('images', 0)}")
            print(f"  Engineering insights: {context.get('engineering_insights', False)}")
            print(f"  Model data: {context.get('model_data', False)}")
            print(f"  Coverage score: {context.get('coverage_score', 0):.1%}")
            
            # Display citations
            print(f"\n📚 Citations ({len(response.citations)}):")
            for i, citation in enumerate(response.citations[:5], 1):  # Show first 5
                print(f"  {i}. {citation.source_type}: {citation.source_name}")
                print(f"     Confidence: {citation.confidence:.1%}")
                if citation.excerpt:
                    excerpt = citation.excerpt[:100] + "..." if len(citation.excerpt) > 100 else citation.excerpt
                    print(f"     Excerpt: {excerpt}")
            
            return True
            
        except Exception as e:
            print(f"❌ Question processing test failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    async def test_multiple_questions(self) -> bool:
        """Test processing multiple different types of questions."""
        print("\n=== Testing Multiple Question Types ===\n")
        
        if not self.qa_system:
            print("❌ QA system not initialized")
            return False
        
        try:
            results = []
            
            for i, question in enumerate(self.test_questions, 1):
                print(f"🧪 Question {i}/{len(self.test_questions)}: {question}")
                
                request = QuestionRequest(
                    question=question,
                    include_images=False,  # Faster processing
                    include_web=False,
                    temperature=0.2
                )
                
                start_time = time.time()
                response = await self.qa_system.process_question(request)
                processing_time = time.time() - start_time
                
                results.append({
                    'question': question,
                    'processing_time': processing_time,
                    'confidence': response.confidence_score,
                    'answer_length': len(response.answer),
                    'citations': len(response.citations),
                    'context_used': response.context_used
                })
                
                print(f"  ⏱️  {processing_time:.2f}s | 🎯 {response.confidence_score:.1%} confidence | 📝 {len(response.answer)} chars")
                print(f"  📚 {len(response.citations)} citations | 📊 {response.context_used.get('coverage_score', 0):.1%} coverage")
                print()
            
            # Summary statistics
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            total_citations = sum(r['citations'] for r in results)
            
            print(f"📊 Summary Statistics:")
            print(f"  Average processing time: {avg_time:.2f} seconds")
            print(f"  Average confidence: {avg_confidence:.1%}")
            print(f"  Total citations: {total_citations}")
            print(f"  Questions processed: {len(results)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Multiple questions test failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def start_test_server(self) -> bool:
        """Start the QA server for API testing."""
        print("\n=== Starting Test Server ===\n")
        
        try:
            import subprocess
            import threading
            import time
            
            # Start server in background
            def run_server():
                subprocess.run([
                    sys.executable, "src/qa_server.py", 
                    "--host", "127.0.0.1", 
                    "--port", "8000"
                ], capture_output=True)
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait for server to start
            print("🔧 Starting server...")
            time.sleep(5)
            
            # Test if server is running
            try:
                response = requests.get(f"{self.server_url}/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Server started successfully")
                    return True
                else:
                    print(f"❌ Server health check failed: {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"❌ Could not connect to server: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Server startup failed: {e}")
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test FastAPI endpoints."""
        print("\n=== Testing API Endpoints ===\n")
        
        try:
            # Test root endpoint
            print("🧪 Testing root endpoint...")
            response = requests.get(f"{self.server_url}/")
            if response.status_code == 200:
                print("  ✅ Root endpoint working")
            else:
                print(f"  ❌ Root endpoint failed: {response.status_code}")
                return False
            
            # Test health endpoint
            print("🧪 Testing health endpoint...")
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("  ✅ Health endpoint working")
                print(f"    Status: {health_data.get('status', 'Unknown')}")
                
                components = health_data.get('components', {})
                for component, status in components.items():
                    print(f"    {component}: {status}")
            else:
                print(f"  ❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test stats endpoint
            print("🧪 Testing stats endpoint...")
            response = requests.get(f"{self.server_url}/system/stats")
            if response.status_code == 200:
                stats_data = response.json()
                print("  ✅ Stats endpoint working")
                
                kb = stats_data.get('knowledge_base', {})
                print(f"    Text documents: {kb.get('text_documents', 0):,}")
                print(f"    Images: {kb.get('images', 0):,}")
                print(f"    3D models: {kb.get('3d_models', 0)}")
            else:
                print(f"  ❌ Stats endpoint failed: {response.status_code}")
                return False
            
            # Test examples endpoint
            print("🧪 Testing examples endpoint...")
            response = requests.get(f"{self.server_url}/examples")
            if response.status_code == 200:
                examples_data = response.json()
                print("  ✅ Examples endpoint working")
                examples = examples_data.get('examples', [])
                print(f"    Question categories: {len(examples)}")
            else:
                print(f"  ❌ Examples endpoint failed: {response.status_code}")
                return False
            
            # Test ask endpoint
            print("🧪 Testing ask endpoint...")
            test_request = {
                "question": "What is the geothermal potential of Semurup?",
                "include_images": False,
                "include_web": False,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.server_url}/ask",
                json=test_request,
                timeout=30
            )
            
            if response.status_code == 200:
                ask_data = response.json()
                print("  ✅ Ask endpoint working")
                print(f"    Answer length: {len(ask_data.get('answer', ''))} characters")
                print(f"    Processing time: {ask_data.get('processing_time_ms', 0)}ms")
                print(f"    Confidence: {ask_data.get('confidence_score', 0):.1%}")
                print(f"    Citations: {len(ask_data.get('citations', []))}")
            else:
                print(f"  ❌ Ask endpoint failed: {response.status_code}")
                if response.text:
                    print(f"    Error: {response.text}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ API endpoint test failed: {e}")
            return False

async def main():
    """Run all tests."""
    print("🧪 Geothermal Digital Twin - QA Server Tests\n")
    
    # Check if we're in the right directory
    if not Path("src/qa_server.py").exists():
        print("❌ Run this script from the geo_twin_ai root directory")
        sys.exit(1)
    
    # Check environment
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - some tests may fail")
    
    tester = QAServerTester()
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: System initialization
    if tester.test_system_initialization():
        tests_passed += 1
    
    # Test 2: Context retrieval
    if await tester.test_context_retrieval():
        tests_passed += 1
    
    # Test 3: Question processing
    if await tester.test_question_processing():
        tests_passed += 1
    
    # Test 4: Multiple questions
    if await tester.test_multiple_questions():
        tests_passed += 1
    
    # Optional: API tests (requires manual server start)
    print("\n=== Optional API Tests ===")
    print("To test API endpoints, run in separate terminal:")
    print("  python src/qa_server.py --host 127.0.0.1 --port 8000")
    print("Then visit: http://127.0.0.1:8000/docs")
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All core tests passed!")
        print("\nReady for production:")
        print("  python src/qa_server.py  # Start the QA server")
        print("  Visit http://127.0.0.1:8000/docs for interactive API documentation")
        print("  Use /ask endpoint to submit questions")
    else:
        print("⚠️  Some tests failed - check the logs above")
        
        if tests_passed == 0:
            print("\n💡 Common issues:")
            print("  • Missing OPENAI_API_KEY environment variable")
            print("  • No processed data - run ingestion first:")
            print("    python src/ingest_literature.py")
            print("    python src/ingest_raw.py --all")
            print("    python src/twin_summariser.py --all")

if __name__ == "__main__":
    import os
    asyncio.run(main())