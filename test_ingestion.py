#!/usr/bin/env python3
"""
Test script for Literature Ingestion Module

Quick test to verify the ingestion pipeline works correctly.
Processes a small subset of PDFs and checks the results.
"""

import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest_literature import LiteratureIngester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic ingestion functionality."""
    print("=== Testing Literature Ingestion ===\n")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - text embeddings will fail")
        print("   Set the key to test full functionality")
        return False
    
    try:
        # Initialize ingester
        print("🔧 Initializing ingester...")
        ingester = LiteratureIngester()
        
        # Check corpus directory
        corpus_files = list(Path("knowledge/corpus").rglob("*.pdf"))
        if not corpus_files:
            print("⚠️  No PDF files found in knowledge/corpus/")
            print("   Add some PDFs to test ingestion")
            return False
        
        print(f"📚 Found {len(corpus_files)} PDF files")
        
        # Test processing first PDF
        test_pdf = corpus_files[0]
        print(f"🧪 Testing with: {test_pdf.name}")
        
        # Extract text and images
        text_content, images_data = ingester.extract_text_and_images(test_pdf)
        
        print(f"📄 Extracted {len(text_content)} characters of text")
        print(f"🖼️  Extracted {len(images_data)} images")
        
        if text_content:
            # Test chunking
            source_info = {
                'filename': test_pdf.name,
                'file_path': str(test_pdf),
                'file_id': test_pdf.stem
            }
            chunks = ingester.chunk_text(text_content, source_info)
            print(f"📝 Created {len(chunks)} text chunks")
        
        # Test CLIP model
        if images_data:
            print("🎨 Testing CLIP embeddings...")
            for img_data in images_data[:2]:  # Test first 2 images
                embedding = ingester.get_image_embedding(img_data['path'], img_data['caption'])
                if embedding:
                    print(f"   ✅ Generated {len(embedding)}-dim embedding for {img_data['path'].name}")
                else:
                    print(f"   ❌ Failed to generate embedding for {img_data['path'].name}")
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chromadb():
    """Test ChromaDB setup and basic operations."""
    print("\n=== Testing ChromaDB ===\n")
    
    try:
        ingester = LiteratureIngester()
        
        # Test text collection
        text_count = ingester.text_collection.count()
        print(f"📄 Text collection has {text_count} documents")
        
        # Test image collection
        image_count = ingester.image_collection.count()
        print(f"🖼️  Image collection has {image_count} documents")
        
        print("✅ ChromaDB test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Geothermal Digital Twin - Ingestion Tests\n")
    
    # Check if we're in the right directory
    if not Path("knowledge/corpus").exists():
        print("❌ Run this script from the geo_twin_ai root directory")
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_chromadb():
        tests_passed += 1
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        print("\nReady to run full ingestion:")
        print("   python src/ingest_literature.py")
    else:
        print("⚠️  Some tests failed - check the setup")
        print("   Run: python setup.py")

if __name__ == "__main__":
    main()