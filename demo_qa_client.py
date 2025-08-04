#!/usr/bin/env python3
"""
Demo QA Client for Geothermal Digital Twin

Interactive demonstration of the QA system capabilities.
Shows how to use the API and provides example interactions.

Usage:
    # Start the server first:
    python src/qa_server.py
    
    # Then run this demo:
    python demo_qa_client.py
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, Any

class GeothermalQAClient:
    """Simple client for interacting with the QA server."""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        """Initialize the client."""
        self.server_url = server_url
        self.session = requests.Session()
    
    def check_health(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            response = self.session.get(f"{self.server_url}/system/stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_examples(self) -> Dict[str, Any]:
        """Get example questions."""
        try:
            response = self.session.get(f"{self.server_url}/examples", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def ask_question(self, question: str, include_images: bool = True, 
                    include_web: bool = False, temperature: float = 0.3) -> Dict[str, Any]:
        """Ask a question to the QA system."""
        try:
            request_data = {
                "question": question,
                "include_images": include_images,
                "include_web": include_web,
                "temperature": temperature
            }
            
            response = self.session.post(
                f"{self.server_url}/ask",
                json=request_data,
                timeout=60  # Allow up to 60 seconds for response
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"error": "Request timed out - try a simpler question"}
        except Exception as e:
            return {"error": str(e)}

def format_response(response_data: Dict[str, Any]) -> str:
    """Format the QA response for display."""
    if "error" in response_data:
        return f"❌ Error: {response_data['error']}"
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {response_data.get('question', 'Unknown')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Answer
    answer = response_data.get('answer', 'No answer provided')
    lines.append("📝 Answer:")
    lines.append(answer)
    lines.append("")
    
    # Metadata
    confidence = response_data.get('confidence_score', 0) * 100
    processing_time = response_data.get('processing_time_ms', 0)
    lines.append(f"🎯 Confidence: {confidence:.1f}%")
    lines.append(f"⏱️  Processing Time: {processing_time}ms")
    lines.append("")
    
    # Context used
    context = response_data.get('context_used', {})
    lines.append("📊 Knowledge Sources Used:")
    lines.append(f"  • Text documents: {context.get('text_chunks', 0)}")
    lines.append(f"  • Images: {context.get('images', 0)}")
    lines.append(f"  • Engineering analysis: {'Yes' if context.get('engineering_insights') else 'No'}")
    lines.append(f"  • 3D model data: {'Yes' if context.get('model_data') else 'No'}")
    lines.append(f"  • Web results: {context.get('web_results', 0)}")
    lines.append(f"  • Coverage score: {context.get('coverage_score', 0)*100:.1f}%")
    lines.append("")
    
    # Citations
    citations = response_data.get('citations', [])
    if citations:
        lines.append(f"📚 Sources ({len(citations)}):")
        for i, citation in enumerate(citations[:10], 1):  # Show first 10
            source_type = citation.get('source_type', 'unknown')
            source_name = citation.get('source_name', 'Unknown')
            confidence = citation.get('confidence', 0) * 100
            
            lines.append(f"  {i}. [{source_type.title()}] {source_name}")
            lines.append(f"     Relevance: {confidence:.1f}%")
            
            if citation.get('page_number'):
                lines.append(f"     Page: {citation['page_number']}")
            
            if citation.get('excerpt'):
                excerpt = citation['excerpt'][:100] + "..." if len(citation['excerpt']) > 100 else citation['excerpt']
                lines.append(f"     Excerpt: {excerpt}")
            lines.append("")
    
    return "\n".join(lines)

def interactive_demo():
    """Run interactive demonstration."""
    print("🌋 Geothermal Digital Twin - Interactive QA Demo")
    print("=" * 60)
    
    client = GeothermalQAClient()
    
    # Check server health
    print("🔍 Checking server status...")
    health = client.check_health()
    if "error" in health:
        print(f"❌ Cannot connect to server: {health['error']}")
        print("\n💡 Make sure the server is running:")
        print("   python src/qa_server.py")
        return
    
    print(f"✅ Server is {health.get('status', 'unknown')}")
    
    # Get system stats
    print("\n📊 System Information:")
    stats = client.get_stats()
    if "error" not in stats:
        kb = stats.get('knowledge_base', {})
        print(f"  📄 Text documents: {kb.get('text_documents', 0):,}")
        print(f"  🖼️  Images: {kb.get('images', 0):,}")
        print(f"  🗺️  3D models: {kb.get('3d_models', 0)}")
        print(f"  ⚡ Engineering summaries: {'Available' if kb.get('engineering_summaries') else 'Not available'}")
        
        models = stats.get('available_models', [])
        if models:
            print(f"  🔬 Available models: {', '.join(models)}")
    
    # Show examples
    print("\n💡 Example Questions:")
    examples = client.get_examples()
    if "error" not in examples:
        for category in examples.get('examples', [])[:3]:  # Show first 3 categories
            print(f"\n  {category['category']}:")
            for question in category['questions'][:2]:  # Show 2 questions per category
                print(f"    • {question}")
    
    # Interactive loop
    print("\n" + "=" * 60)
    print("🎯 Ask questions about the Semurup geothermal field!")
    print("Type 'quit' to exit, 'examples' to see more examples")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'examples':
                examples = client.get_examples()
                if "error" not in examples:
                    print("\n💡 All Example Questions:")
                    for category in examples.get('examples', []):
                        print(f"\n  {category['category']}:")
                        for q in category['questions']:
                            print(f"    • {q}")
                continue
            
            if len(question) < 5:
                print("⚠️  Please ask a more detailed question (at least 5 characters)")
                continue
            
            print(f"\n🤔 Processing your question...")
            start_time = time.time()
            
            response = client.ask_question(
                question=question,
                include_images=True,
                include_web=False,  # Disable web search for demo
                temperature=0.3
            )
            
            response_time = time.time() - start_time
            print(f"⚡ Got response in {response_time:.1f} seconds")
            
            # Display formatted response
            formatted = format_response(response)
            print(formatted)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_demo():
    """Run batch demonstration with predefined questions."""
    print("🌋 Geothermal Digital Twin - Batch QA Demo")
    print("=" * 60)
    
    client = GeothermalQAClient()
    
    # Check server
    health = client.check_health()
    if "error" in health:
        print(f"❌ Cannot connect to server: {health['error']}")
        return
    
    # Demo questions
    demo_questions = [
        "What is the estimated geothermal capacity of the Semurup field?",
        "Where are the best reservoir zones located for drilling?",
        "What are the main geological zones identified in the 3D models?",
        "What drilling depths are recommended for this field?",
        "What do the geochemical surveys tell us about fluid temperatures?"
    ]
    
    print(f"🧪 Running batch demo with {len(demo_questions)} questions...\n")
    
    results = []
    
    for i, question in enumerate(demo_questions, 1):
        print(f"Question {i}/{len(demo_questions)}: {question}")
        print("-" * 50)
        
        start_time = time.time()
        response = client.ask_question(
            question=question,
            include_images=False,  # Faster processing
            temperature=0.2
        )
        processing_time = time.time() - start_time
        
        if "error" in response:
            print(f"❌ Error: {response['error']}\n")
            continue
        
        # Show summary
        answer_length = len(response.get('answer', ''))
        confidence = response.get('confidence_score', 0) * 100
        citations = len(response.get('citations', []))
        
        print(f"✅ Answer: {answer_length} chars | {confidence:.1f}% confidence | {citations} citations")
        print(f"⏱️  Time: {processing_time:.1f}s | Processing: {response.get('processing_time_ms', 0)}ms")
        
        # Show brief answer
        answer = response.get('answer', '')
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"📝 Preview: {preview}")
        print()
        
        results.append({
            'question': question,
            'processing_time': processing_time,
            'confidence': confidence,
            'answer_length': answer_length,
            'citations': citations
        })
        
        # Brief pause between questions
        time.sleep(1)
    
    # Summary statistics
    if results:
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        total_citations = sum(r['citations'] for r in results)
        
        print("=" * 60)
        print("📊 Batch Demo Summary:")
        print(f"  Questions processed: {len(results)}")
        print(f"  Average processing time: {avg_time:.1f} seconds")
        print(f"  Average confidence: {avg_confidence:.1f}%")
        print(f"  Total citations: {total_citations}")
        print("=" * 60)

def main():
    """Main demo entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_demo()
    else:
        interactive_demo()

if __name__ == "__main__":
    main()