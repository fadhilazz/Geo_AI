#!/usr/bin/env python3
"""
Evaluation Runner for Geothermal Digital Twin AI

Comprehensive nightly evaluation system for quality assurance and performance monitoring.
Tests all system components, tracks metrics, and generates detailed reports.

Features:
1. End-to-end QA system testing
2. Performance benchmarking and regression detection
3. Knowledge base validation and coverage analysis
4. Multi-dimensional quality scoring
5. Historical trend analysis and alerting
6. Automated report generation with insights
7. Integration testing across all modules
8. Error analysis and debugging assistance
9. Automated remediation suggestions
10. Configurable test suites and thresholds

Dependencies:
- All system modules (qa_server, question_graph, etc.)
- requests: API testing
- pandas: Data analysis and reporting
- matplotlib, seaborn: Visualization
- smtplib: Email notifications
- schedule: Task scheduling

Usage:
    python src/eval_runner.py [--run-tests] [--generate-report] [--schedule]
"""

import os
import sys
import json
import logging
import time
import asyncio
import smtplib
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import traceback
import statistics
import subprocess
import psutil

# Data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# HTTP requests for API testing
import requests

# Progress tracking
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from qa_server import GeothermalQASystem, QuestionRequest
    from question_graph import GeothermalQuestionGraph
    from twin_summariser import GeothermalTwinSummariser
    from ingest_raw import RawDataIngester
    MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some modules not available for testing: {e}")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EVAL_DIR = Path("digital_twin/evaluations")
REPORTS_DIR = EVAL_DIR / "reports"
RESULTS_DIR = EVAL_DIR / "results"
CONFIG_FILE = EVAL_DIR / "eval_config.json"
BENCHMARK_FILE = EVAL_DIR / "benchmarks.json"

# Default configuration
DEFAULT_CONFIG = {
    "evaluation_settings": {
        "max_test_duration_minutes": 30,
        "qa_server_url": "http://127.0.0.1:8000",
        "test_timeout_seconds": 60,
        "performance_baseline_percentile": 90,
        "quality_threshold": 0.75,
        "regression_threshold": 0.15
    },
    "test_suites": {
        "smoke_tests": {
            "enabled": True,
            "description": "Basic functionality verification",
            "max_duration_minutes": 5
        },
        "integration_tests": {
            "enabled": True,
            "description": "Cross-module integration testing",
            "max_duration_minutes": 10
        },
        "performance_tests": {
            "enabled": True,
            "description": "Performance and benchmarking",
            "max_duration_minutes": 10
        },
        "quality_tests": {
            "enabled": True,
            "description": "Answer quality and accuracy",
            "max_duration_minutes": 15
        }
    },
    "notifications": {
        "email_enabled": False,
        "email_recipients": [],
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "alert_on_failure": True,
        "alert_on_regression": True
    },
    "reporting": {
        "generate_pdf": True,
        "generate_excel": True,
        "include_visualizations": True,
        "retain_days": 30
    }
}


class GeothermalEvaluationRunner:
    """Comprehensive evaluation system for the geothermal digital twin."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the evaluation runner."""
        self.setup_directories()
        self.config = self.load_config(config_path)
        self.results = {}
        self.benchmarks = self.load_benchmarks()
        self.start_time = None
        self.end_time = None
        
        # Test question sets
        self.test_questions = self.define_test_questions()
        
        logger.info("Evaluation runner initialized")
    
    def setup_directories(self):
        """Create required directories."""
        for dir_path in [EVAL_DIR, REPORTS_DIR, RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: Optional[Path] = None) -> Dict:
        """Load evaluation configuration."""
        config_file = config_path or CONFIG_FILE
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info("Configuration loaded")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        # Save default config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        
        logger.info("Using default configuration")
        return DEFAULT_CONFIG
    
    def load_benchmarks(self) -> Dict:
        """Load historical benchmarks."""
        if BENCHMARK_FILE.exists():
            try:
                with open(BENCHMARK_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load benchmarks: {e}")
        
        return {
            "performance_baselines": {},
            "quality_baselines": {},
            "last_update": None
        }
    
    def save_benchmarks(self):
        """Save updated benchmarks."""
        try:
            with open(BENCHMARK_FILE, 'w') as f:
                json.dump(self.benchmarks, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save benchmarks: {e}")
    
    def define_test_questions(self) -> Dict[str, List[Dict]]:
        """Define comprehensive test question sets."""
        return {
            "smoke_tests": [
                {
                    "question": "What is geothermal energy?",
                    "expected_keywords": ["geothermal", "energy", "heat", "earth"],
                    "max_response_time": 5.0,
                    "min_confidence": 0.6
                },
                {
                    "question": "What is the Semurup field?",
                    "expected_keywords": ["semurup", "geothermal", "field", "sumatra"],
                    "max_response_time": 5.0,
                    "min_confidence": 0.7
                }
            ],
            
            "capacity_assessment": [
                {
                    "question": "What is the estimated geothermal capacity of the Semurup field?",
                    "expected_keywords": ["capacity", "mw", "megawatt", "semurup", "estimated"],
                    "max_response_time": 10.0,
                    "min_confidence": 0.8,
                    "requires_quantitative": True
                },
                {
                    "question": "How much power can the Semurup field generate?",
                    "expected_keywords": ["power", "generate", "mw", "capacity"],
                    "max_response_time": 10.0,
                    "min_confidence": 0.8,
                    "requires_quantitative": True
                },
                {
                    "question": "What is the sustainable production rate for the field?",
                    "expected_keywords": ["sustainable", "production", "rate"],
                    "max_response_time": 12.0,
                    "min_confidence": 0.7
                }
            ],
            
            "geological_analysis": [
                {
                    "question": "What geological zones have been identified in the 3D models?",
                    "expected_keywords": ["geological", "zones", "3d", "models", "caprock", "reservoir"],
                    "max_response_time": 15.0,
                    "min_confidence": 0.8,
                    "requires_model_data": True
                },
                {
                    "question": "Where is the caprock layer located?",
                    "expected_keywords": ["caprock", "layer", "location", "depth"],
                    "max_response_time": 12.0,
                    "min_confidence": 0.8,
                    "requires_spatial": True
                },
                {
                    "question": "What are the resistivity characteristics of the reservoir?",
                    "expected_keywords": ["resistivity", "characteristics", "reservoir", "ohm"],
                    "max_response_time": 15.0,
                    "min_confidence": 0.8,
                    "requires_model_data": True
                }
            ],
            
            "drilling_recommendations": [
                {
                    "question": "Where should we drill the first production well?",
                    "expected_keywords": ["drill", "production", "well", "location", "target"],
                    "max_response_time": 15.0,
                    "min_confidence": 0.8,
                    "requires_spatial": True
                },
                {
                    "question": "What drilling depths are recommended for this field?",
                    "expected_keywords": ["drilling", "depths", "recommended", "meters"],
                    "max_response_time": 10.0,
                    "min_confidence": 0.7,
                    "requires_quantitative": True
                },
                {
                    "question": "What are the injection well locations?",
                    "expected_keywords": ["injection", "well", "locations"],
                    "max_response_time": 12.0,
                    "min_confidence": 0.7,
                    "requires_spatial": True
                }
            ],
            
            "geochemistry": [
                {
                    "question": "What do the geochemical surveys tell us about fluid temperatures?",
                    "expected_keywords": ["geochemical", "surveys", "fluid", "temperatures"],
                    "max_response_time": 10.0,
                    "min_confidence": 0.8
                },
                {
                    "question": "What geothermometry results are available?",
                    "expected_keywords": ["geothermometry", "results", "temperature"],
                    "max_response_time": 10.0,
                    "min_confidence": 0.7
                }
            ],
            
            "complex_integration": [
                {
                    "question": "Based on the 3D models and geochemical data, what are the optimal development strategies for the Semurup field?",
                    "expected_keywords": ["3d", "models", "geochemical", "development", "strategies", "optimal"],
                    "max_response_time": 20.0,
                    "min_confidence": 0.8,
                    "requires_model_data": True,
                    "requires_integration": True
                },
                {
                    "question": "How do the resistivity models correlate with the geochemical survey results?",
                    "expected_keywords": ["resistivity", "models", "correlate", "geochemical", "survey"],
                    "max_response_time": 18.0,
                    "min_confidence": 0.8,
                    "requires_model_data": True,
                    "requires_integration": True
                }
            ]
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health before testing."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "system_resources": {},
            "service_status": {},
            "data_availability": {},
            "overall_health": True
        }
        
        try:
            # System resources
            health_status["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('.').percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }
            
            # Check data availability
            data_paths = {
                "literature_texts": Path("knowledge/text_emb"),
                "literature_images": Path("knowledge/image_emb"),
                "3d_grids": Path("digital_twin/grids"),
                "summaries": Path("digital_twin/cache/twin_summaries.yaml"),
                "question_graph": Path("digital_twin/cache/question_graph.pkl")
            }
            
            for name, path in data_paths.items():
                health_status["data_availability"][name] = path.exists()
                if not path.exists():
                    health_status["overall_health"] = False
            
            # Check QA server
            if self.config["evaluation_settings"]["qa_server_url"]:
                try:
                    response = requests.get(
                        f"{self.config['evaluation_settings']['qa_server_url']}/health",
                        timeout=10
                    )
                    health_status["service_status"]["qa_server"] = {
                        "available": response.status_code == 200,
                        "response_time": response.elapsed.total_seconds()
                    }
                except:
                    health_status["service_status"]["qa_server"] = {
                        "available": False,
                        "response_time": None
                    }
                    health_status["overall_health"] = False
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["overall_health"] = False
            health_status["error"] = str(e)
            return health_status
    
    async def run_smoke_tests(self) -> Dict[str, Any]:
        """Run basic smoke tests for system functionality."""
        logger.info("Running smoke tests...")
        
        results = {
            "test_type": "smoke_tests",
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        
        smoke_questions = self.test_questions["smoke_tests"]
        
        with tqdm(smoke_questions, desc="Smoke tests") as pbar:
            for test_case in pbar:
                pbar.set_description(f"Testing: {test_case['question'][:30]}...")
                
                test_result = await self.run_single_test(test_case)
                results["tests"].append(test_result)
        
        # Calculate summary
        passed_tests = sum(1 for test in results["tests"] if test["passed"])
        total_tests = len(results["tests"])
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "average_response_time": statistics.mean([test["response_time"] for test in results["tests"]]),
            "average_confidence": statistics.mean([test["confidence"] for test in results["tests"]])
        }
        
        results["end_time"] = datetime.now().isoformat()
        logger.info(f"Smoke tests completed: {passed_tests}/{total_tests} passed")
        
        return results
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests across all modules."""
        logger.info("Running integration tests...")
        
        results = {
            "test_type": "integration_tests",
            "start_time": datetime.now().isoformat(),
            "module_tests": {},
            "cross_module_tests": [],
            "summary": {}
        }
        
        # Test individual modules
        if MODULES_AVAILABLE:
            module_results = {}
            
            # Test QA System
            try:
                qa_system = GeothermalQASystem()
                module_results["qa_system"] = {
                    "initialized": True,
                    "text_collection_available": qa_system.text_collection is not None,
                    "image_collection_available": qa_system.image_collection is not None,
                    "summaries_available": bool(qa_system.summaries),
                    "grids_metadata_count": len(qa_system.grids_metadata)
                }
            except Exception as e:
                module_results["qa_system"] = {"initialized": False, "error": str(e)}
            
            # Test Question Graph
            try:
                qg = GeothermalQuestionGraph()
                loaded = qg.load_graph()
                module_results["question_graph"] = {
                    "initialized": True,
                    "graph_loaded": loaded,
                    "faiss_index_available": qg.faiss_index is not None
                }
            except Exception as e:
                module_results["question_graph"] = {"initialized": False, "error": str(e)}
            
            # Test Twin Summariser
            try:
                ts = GeothermalTwinSummariser()
                module_results["twin_summariser"] = {
                    "initialized": True,
                    "summaries_available": bool(ts.summaries)
                }
            except Exception as e:
                module_results["twin_summariser"] = {"initialized": False, "error": str(e)}
            
            results["module_tests"] = module_results
        
        # Run cross-module integration tests
        integration_questions = []
        for category in ["capacity_assessment", "geological_analysis", "complex_integration"]:
            integration_questions.extend(self.test_questions[category][:2])  # Take first 2 from each
        
        with tqdm(integration_questions, desc="Integration tests") as pbar:
            for test_case in pbar:
                pbar.set_description(f"Testing: {test_case['question'][:30]}...")
                
                test_result = await self.run_single_test(test_case)
                results["cross_module_tests"].append(test_result)
        
        # Calculate summary
        passed_tests = sum(1 for test in results["cross_module_tests"] if test["passed"])
        total_tests = len(results["cross_module_tests"])
        
        results["summary"] = {
            "total_integration_tests": total_tests,
            "passed_integration_tests": passed_tests,
            "module_tests_summary": {
                module: "passed" if data.get("initialized", False) else "failed"
                for module, data in results.get("module_tests", {}).items()
            },
            "integration_pass_rate": passed_tests / total_tests if total_tests > 0 else 0
        }
        
        results["end_time"] = datetime.now().isoformat()
        logger.info(f"Integration tests completed: {passed_tests}/{total_tests} passed")
        
        return results
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        logger.info("Running performance tests...")
        
        results = {
            "test_type": "performance_tests",
            "start_time": datetime.now().isoformat(),
            "benchmarks": {},
            "regressions": [],
            "summary": {}
        }
        
        # Performance test categories
        perf_categories = ["capacity_assessment", "geological_analysis", "drilling_recommendations"]
        
        for category in perf_categories:
            category_results = []
            questions = self.test_questions[category][:3]  # Test first 3 questions
            
            for test_case in questions:
                test_result = await self.run_single_test(test_case)
                category_results.append(test_result)
            
            # Calculate category benchmarks
            if category_results:
                response_times = [r["response_time"] for r in category_results]
                confidences = [r["confidence"] for r in category_results]
                
                results["benchmarks"][category] = {
                    "average_response_time": statistics.mean(response_times),
                    "p90_response_time": np.percentile(response_times, 90),
                    "p95_response_time": np.percentile(response_times, 95),
                    "average_confidence": statistics.mean(confidences),
                    "min_confidence": min(confidences),
                    "test_count": len(category_results)
                }
        
        # Check for regressions
        for category, benchmarks in results["benchmarks"].items():
            if category in self.benchmarks.get("performance_baselines", {}):
                baseline = self.benchmarks["performance_baselines"][category]
                
                # Check response time regression
                current_p90 = benchmarks["p90_response_time"]
                baseline_p90 = baseline.get("p90_response_time", current_p90)
                
                if current_p90 > baseline_p90 * (1 + self.config["evaluation_settings"]["regression_threshold"]):
                    results["regressions"].append({
                        "category": category,
                        "metric": "response_time",
                        "current": current_p90,
                        "baseline": baseline_p90,
                        "regression_percent": ((current_p90 - baseline_p90) / baseline_p90) * 100
                    })
                
                # Check confidence regression
                current_conf = benchmarks["average_confidence"]
                baseline_conf = baseline.get("average_confidence", current_conf)
                
                if current_conf < baseline_conf * (1 - self.config["evaluation_settings"]["regression_threshold"]):
                    results["regressions"].append({
                        "category": category,
                        "metric": "confidence",
                        "current": current_conf,
                        "baseline": baseline_conf,
                        "regression_percent": ((baseline_conf - current_conf) / baseline_conf) * 100
                    })
        
        # Update benchmarks
        self.benchmarks["performance_baselines"].update(results["benchmarks"])
        self.benchmarks["last_update"] = datetime.now().isoformat()
        self.save_benchmarks()
        
        results["summary"] = {
            "categories_tested": len(results["benchmarks"]),
            "regressions_detected": len(results["regressions"]),
            "overall_performance": "degraded" if results["regressions"] else "stable"
        }
        
        results["end_time"] = datetime.now().isoformat()
        logger.info(f"Performance tests completed: {len(results['regressions'])} regressions detected")
        
        return results
    
    async def run_quality_tests(self) -> Dict[str, Any]:
        """Run quality assessment tests."""
        logger.info("Running quality tests...")
        
        results = {
            "test_type": "quality_tests",
            "start_time": datetime.now().isoformat(),
            "quality_metrics": {},
            "failed_quality_checks": [],
            "summary": {}
        }
        
        # Test all categories for quality
        all_categories = list(self.test_questions.keys())
        quality_scores = {}
        
        for category in all_categories:
            if category == "smoke_tests":
                continue  # Skip smoke tests for quality assessment
            
            questions = self.test_questions[category]
            category_scores = []
            
            for test_case in questions:
                test_result = await self.run_single_test(test_case)
                
                # Calculate quality score
                quality_score = self.calculate_quality_score(test_result, test_case)
                category_scores.append(quality_score)
                
                # Check for quality failures
                if quality_score < self.config["evaluation_settings"]["quality_threshold"]:
                    results["failed_quality_checks"].append({
                        "category": category,
                        "question": test_case["question"],
                        "quality_score": quality_score,
                        "issues": self.identify_quality_issues(test_result, test_case)
                    })
            
            if category_scores:
                quality_scores[category] = {
                    "average_quality": statistics.mean(category_scores),
                    "min_quality": min(category_scores),
                    "max_quality": max(category_scores),
                    "quality_std": statistics.stdev(category_scores) if len(category_scores) > 1 else 0
                }
        
        results["quality_metrics"] = quality_scores
        
        # Overall quality assessment
        all_scores = [score for scores in quality_scores.values() for score in [scores["average_quality"]]]
        overall_quality = statistics.mean(all_scores) if all_scores else 0
        
        results["summary"] = {
            "overall_quality_score": overall_quality,
            "categories_below_threshold": len([
                cat for cat, scores in quality_scores.items()
                if scores["average_quality"] < self.config["evaluation_settings"]["quality_threshold"]
            ]),
            "total_quality_failures": len(results["failed_quality_checks"]),
            "quality_grade": self.get_quality_grade(overall_quality)
        }
        
        results["end_time"] = datetime.now().isoformat()
        logger.info(f"Quality tests completed: {overall_quality:.2f} overall score")
        
        return results
    
    async def run_single_test(self, test_case: Dict) -> Dict[str, Any]:
        """Run a single test case."""
        start_time = time.time()
        
        test_result = {
            "question": test_case["question"],
            "start_time": datetime.now().isoformat(),
            "response_time": 0,
            "passed": False,
            "confidence": 0,
            "answer_length": 0,
            "citations_count": 0,
            "errors": [],
            "context_used": {},
            "keyword_matches": []
        }
        
        try:
            # Make request to QA server
            qa_url = self.config["evaluation_settings"]["qa_server_url"]
            timeout = self.config["evaluation_settings"]["test_timeout_seconds"]
            
            request_data = {
                "question": test_case["question"],
                "include_images": True,
                "include_web": False,
                "temperature": 0.2  # Lower temperature for consistent testing
            }
            
            response = requests.post(
                f"{qa_url}/ask",
                json=request_data,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            test_result["response_time"] = response_time
            
            if response.status_code == 200:
                data = response.json()
                
                test_result["confidence"] = data.get("confidence_score", 0)
                test_result["answer_length"] = len(data.get("answer", ""))
                test_result["citations_count"] = len(data.get("citations", []))
                test_result["context_used"] = data.get("context_used", {})
                
                # Check expected keywords
                answer_lower = data.get("answer", "").lower()
                expected_keywords = test_case.get("expected_keywords", [])
                matched_keywords = [kw for kw in expected_keywords if kw in answer_lower]
                test_result["keyword_matches"] = matched_keywords
                
                # Determine if test passed
                test_result["passed"] = self.evaluate_test_result(test_result, test_case)
                
            else:
                test_result["errors"].append(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            test_result["errors"].append("Request timeout")
            test_result["response_time"] = timeout
        except Exception as e:
            test_result["errors"].append(str(e))
            test_result["response_time"] = time.time() - start_time
        
        test_result["end_time"] = datetime.now().isoformat()
        return test_result
    
    def evaluate_test_result(self, test_result: Dict, test_case: Dict) -> bool:
        """Evaluate if a test result passes the criteria."""
        # Check response time
        max_response_time = test_case.get("max_response_time", 30.0)
        if test_result["response_time"] > max_response_time:
            return False
        
        # Check minimum confidence
        min_confidence = test_case.get("min_confidence", 0.5)
        if test_result["confidence"] < min_confidence:
            return False
        
        # Check keyword matches
        expected_keywords = test_case.get("expected_keywords", [])
        if expected_keywords:
            match_ratio = len(test_result["keyword_matches"]) / len(expected_keywords)
            if match_ratio < 0.5:  # At least 50% of keywords should match
                return False
        
        # Check for errors
        if test_result["errors"]:
            return False
        
        # Check answer length (should have substantial content)
        if test_result["answer_length"] < 50:
            return False
        
        return True
    
    def calculate_quality_score(self, test_result: Dict, test_case: Dict) -> float:
        """Calculate a quality score for a test result."""
        score = 0.0
        
        # Confidence contribution (30%)
        confidence_score = test_result["confidence"] * 0.3
        score += confidence_score
        
        # Keyword matching contribution (25%)
        expected_keywords = test_case.get("expected_keywords", [])
        if expected_keywords:
            match_ratio = len(test_result["keyword_matches"]) / len(expected_keywords)
            keyword_score = match_ratio * 0.25
            score += keyword_score
        else:
            score += 0.25  # Full points if no keywords specified
        
        # Response completeness (20%)
        answer_length = test_result["answer_length"]
        completeness_score = min(answer_length / 200, 1.0) * 0.2  # Normalize to 200 chars
        score += completeness_score
        
        # Citations quality (15%)
        citations_count = test_result["citations_count"]
        citations_score = min(citations_count / 3, 1.0) * 0.15  # Normalize to 3 citations
        score += citations_score
        
        # Response time penalty (10%)
        max_response_time = test_case.get("max_response_time", 30.0)
        response_time = test_result["response_time"]
        time_score = max(0, (max_response_time - response_time) / max_response_time) * 0.1
        score += time_score
        
        return min(score, 1.0)
    
    def identify_quality_issues(self, test_result: Dict, test_case: Dict) -> List[str]:
        """Identify specific quality issues with a test result."""
        issues = []
        
        if test_result["confidence"] < 0.6:
            issues.append("Low confidence score")
        
        if test_result["answer_length"] < 100:
            issues.append("Answer too short")
        
        if test_result["citations_count"] == 0:
            issues.append("No citations provided")
        
        expected_keywords = test_case.get("expected_keywords", [])
        if expected_keywords:
            match_ratio = len(test_result["keyword_matches"]) / len(expected_keywords)
            if match_ratio < 0.3:
                issues.append("Poor keyword coverage")
        
        max_response_time = test_case.get("max_response_time", 30.0)
        if test_result["response_time"] > max_response_time * 0.8:
            issues.append("Slow response time")
        
        if test_result["errors"]:
            issues.append("Execution errors")
        
        return issues
    
    def get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        logger.info("=== Starting Complete Evaluation Suite ===")
        self.start_time = datetime.now()
        
        # Check system health first
        health_status = self.check_system_health()
        if not health_status["overall_health"]:
            logger.error("System health check failed - aborting evaluation")
            return {
                "status": "aborted",
                "reason": "System health check failed",
                "health_status": health_status
            }
        
        evaluation_results = {
            "evaluation_id": f"eval_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            "start_time": self.start_time.isoformat(),
            "config": self.config,
            "health_status": health_status,
            "test_results": {},
            "overall_summary": {}
        }
        
        # Run test suites
        test_suites = self.config["test_suites"]
        
        try:
            if test_suites.get("smoke_tests", {}).get("enabled", True):
                evaluation_results["test_results"]["smoke_tests"] = await self.run_smoke_tests()
            
            if test_suites.get("integration_tests", {}).get("enabled", True):
                evaluation_results["test_results"]["integration_tests"] = await self.run_integration_tests()
            
            if test_suites.get("performance_tests", {}).get("enabled", True):
                evaluation_results["test_results"]["performance_tests"] = await self.run_performance_tests()
            
            if test_suites.get("quality_tests", {}).get("enabled", True):
                evaluation_results["test_results"]["quality_tests"] = await self.run_quality_tests()
            
            # Calculate overall summary
            evaluation_results["overall_summary"] = self.calculate_overall_summary(evaluation_results["test_results"])
            
            self.end_time = datetime.now()
            evaluation_results["end_time"] = self.end_time.isoformat()
            evaluation_results["total_duration_minutes"] = (self.end_time - self.start_time).total_seconds() / 60
            
            # Save results
            self.save_evaluation_results(evaluation_results)
            
            logger.info("=== Evaluation Suite Completed ===")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            evaluation_results["status"] = "failed"
            evaluation_results["error"] = str(e)
            evaluation_results["traceback"] = traceback.format_exc()
            return evaluation_results
    
    def calculate_overall_summary(self, test_results: Dict) -> Dict[str, Any]:
        """Calculate overall evaluation summary."""
        summary = {
            "overall_status": "passed",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "overall_pass_rate": 0,
            "performance_grade": "unknown",
            "quality_grade": "unknown",
            "critical_issues": [],
            "recommendations": []
        }
        
        # Aggregate test counts
        for suite_name, suite_results in test_results.items():
            if "summary" in suite_results:
                suite_summary = suite_results["summary"]
                
                if suite_name == "smoke_tests":
                    total = suite_summary.get("total_tests", 0)
                    passed = suite_summary.get("passed_tests", 0)
                    summary["total_tests"] += total
                    summary["passed_tests"] += passed
                    summary["failed_tests"] += (total - passed)
                
                elif suite_name == "integration_tests":
                    total = suite_summary.get("total_integration_tests", 0)
                    passed = suite_summary.get("passed_integration_tests", 0)
                    summary["total_tests"] += total
                    summary["passed_tests"] += passed
                    summary["failed_tests"] += (total - passed)
                
                elif suite_name == "quality_tests":
                    quality_score = suite_summary.get("overall_quality_score", 0)
                    summary["quality_grade"] = self.get_quality_grade(quality_score)
                    
                    if suite_summary.get("total_quality_failures", 0) > 0:
                        summary["critical_issues"].append("Quality tests failed")
        
        # Calculate overall pass rate
        if summary["total_tests"] > 0:
            summary["overall_pass_rate"] = summary["passed_tests"] / summary["total_tests"]
        
        # Determine overall status
        if summary["overall_pass_rate"] < 0.8:
            summary["overall_status"] = "failed"
            summary["critical_issues"].append("Low overall pass rate")
        
        # Check for performance regressions
        perf_results = test_results.get("performance_tests", {})
        if perf_results.get("regressions"):
            summary["critical_issues"].append("Performance regressions detected")
            summary["overall_status"] = "degraded"
        
        # Generate recommendations
        if summary["overall_pass_rate"] < 0.9:
            summary["recommendations"].append("Review failed test cases and improve system reliability")
        
        if summary["quality_grade"] in ["D", "F"]:
            summary["recommendations"].append("Improve answer quality and knowledge base coverage")
        
        if len(summary["critical_issues"]) == 0:
            summary["recommendations"].append("System performing well - continue monitoring")
        
        return summary
    
    def save_evaluation_results(self, results: Dict):
        """Save evaluation results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = RESULTS_DIR / f"evaluation_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 80,
            "GEOTHERMAL DIGITAL TWIN - EVALUATION REPORT",
            "=" * 80,
            f"Evaluation ID: {results.get('evaluation_id', 'Unknown')}",
            f"Date: {results.get('start_time', 'Unknown')}",
            f"Duration: {results.get('total_duration_minutes', 0):.1f} minutes",
            ""
        ])
        
        # Overall Summary
        overall = results.get("overall_summary", {})
        report_lines.extend([
            "OVERALL SUMMARY",
            "-" * 40,
            f"Status: {overall.get('overall_status', 'Unknown').upper()}",
            f"Pass Rate: {overall.get('overall_pass_rate', 0):.1%}",
            f"Quality Grade: {overall.get('quality_grade', 'Unknown')}",
            f"Total Tests: {overall.get('total_tests', 0)}",
            f"Passed: {overall.get('passed_tests', 0)}",
            f"Failed: {overall.get('failed_tests', 0)}",
            ""
        ])
        
        # Critical Issues
        critical_issues = overall.get("critical_issues", [])
        if critical_issues:
            report_lines.extend([
                "CRITICAL ISSUES",
                "-" * 40
            ])
            for issue in critical_issues:
                report_lines.append(f"• {issue}")
            report_lines.append("")
        
        # Test Suite Details
        test_results = results.get("test_results", {})
        
        for suite_name, suite_data in test_results.items():
            suite_title = suite_name.replace("_", " ").title()
            report_lines.extend([
                f"{suite_title}",
                "-" * len(suite_title)
            ])
            
            summary = suite_data.get("summary", {})
            
            if suite_name == "smoke_tests":
                report_lines.extend([
                    f"Tests Run: {summary.get('total_tests', 0)}",
                    f"Passed: {summary.get('passed_tests', 0)}",
                    f"Pass Rate: {summary.get('pass_rate', 0):.1%}",
                    f"Average Response Time: {summary.get('average_response_time', 0):.2f}s",
                    f"Average Confidence: {summary.get('average_confidence', 0):.1%}"
                ])
            
            elif suite_name == "performance_tests":
                report_lines.extend([
                    f"Categories Tested: {summary.get('categories_tested', 0)}",
                    f"Regressions Detected: {summary.get('regressions_detected', 0)}",
                    f"Performance Status: {summary.get('overall_performance', 'Unknown')}"
                ])
                
                # Add regression details
                regressions = suite_data.get("regressions", [])
                if regressions:
                    report_lines.append("\nPerformance Regressions:")
                    for regression in regressions:
                        report_lines.append(
                            f"  • {regression['category']} {regression['metric']}: "
                            f"{regression['regression_percent']:.1f}% slower"
                        )
            
            elif suite_name == "quality_tests":
                report_lines.extend([
                    f"Overall Quality Score: {summary.get('overall_quality_score', 0):.2f}",
                    f"Quality Grade: {summary.get('quality_grade', 'Unknown')}",
                    f"Quality Failures: {summary.get('total_quality_failures', 0)}"
                ])
            
            report_lines.append("")
        
        # Recommendations
        recommendations = overall.get("recommendations", [])
        if recommendations:
            report_lines.extend([
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for rec in recommendations:
                report_lines.append(f"• {rec}")
            report_lines.append("")
        
        # System Health
        health = results.get("health_status", {})
        if health:
            report_lines.extend([
                "SYSTEM HEALTH",
                "-" * 40,
                f"Overall Health: {'HEALTHY' if health.get('overall_health') else 'UNHEALTHY'}",
            ])
            
            resources = health.get("system_resources", {})
            if resources:
                report_lines.extend([
                    f"CPU Usage: {resources.get('cpu_percent', 0):.1f}%",
                    f"Memory Usage: {resources.get('memory_percent', 0):.1f}%",
                    f"Disk Usage: {resources.get('disk_percent', 0):.1f}%",
                ])
        
        report_lines.extend([
            "",
            "=" * 80,
            f"Report generated at {datetime.now().isoformat()}",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def send_notification(self, results: Dict):
        """Send email notification with results."""
        if not self.config["notifications"]["email_enabled"]:
            return
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.config["notifications"].get("from_email", "noreply@geothermal.ai")
            msg['Subject'] = f"Geothermal Digital Twin Evaluation - {results.get('overall_summary', {}).get('overall_status', 'Unknown').upper()}"
            
            # Generate report
            report_text = self.generate_evaluation_report(results)
            msg.attach(MimeText(report_text, 'plain'))
            
            # Send to recipients
            recipients = self.config["notifications"]["email_recipients"]
            if recipients:
                server = smtplib.SMTP(
                    self.config["notifications"]["smtp_server"],
                    self.config["notifications"]["smtp_port"]
                )
                server.starttls()
                # Note: Add authentication as needed
                
                for recipient in recipients:
                    msg['To'] = recipient
                    server.send_message(msg)
                
                server.quit()
                logger.info(f"Notification sent to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Geothermal Digital Twin Evaluation Runner")
    parser.add_argument("--run-tests", action="store_true", help="Run complete evaluation suite")
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke tests only")
    parser.add_argument("--performance-only", action="store_true", help="Run performance tests only")
    parser.add_argument("--quality-only", action="store_true", help="Run quality tests only")
    parser.add_argument("--generate-report", type=str, help="Generate report from results file")
    parser.add_argument("--schedule", action="store_true", help="Schedule nightly runs")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluation runner
        config_path = Path(args.config) if args.config else None
        runner = GeothermalEvaluationRunner(config_path)
        
        if args.run_tests:
            # Run complete evaluation
            results = asyncio.run(runner.run_complete_evaluation())
            
            # Generate and display report
            report = runner.generate_evaluation_report(results)
            print(report)
            
            # Send notifications if enabled
            runner.send_notification(results)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = REPORTS_DIR / f"evaluation_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\nDetailed report saved to: {report_file}")
        
        elif args.smoke_only:
            results = asyncio.run(runner.run_smoke_tests())
            print(f"Smoke tests: {results['summary']['passed_tests']}/{results['summary']['total_tests']} passed")
        
        elif args.performance_only:
            results = asyncio.run(runner.run_performance_tests())
            print(f"Performance tests: {len(results['regressions'])} regressions detected")
        
        elif args.quality_only:
            results = asyncio.run(runner.run_quality_tests())
            print(f"Quality score: {results['summary']['overall_quality_score']:.2f}")
        
        elif args.generate_report:
            # Load and generate report from existing results
            results_file = Path(args.generate_report)
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                report = runner.generate_evaluation_report(results)
                print(report)
            else:
                print(f"Results file not found: {results_file}")
        
        elif args.schedule:
            # Schedule nightly runs
            schedule.every().day.at("02:00").do(
                lambda: asyncio.run(runner.run_complete_evaluation())
            )
            
            print("Evaluation scheduled for 2:00 AM daily. Press Ctrl+C to stop.")
            
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()