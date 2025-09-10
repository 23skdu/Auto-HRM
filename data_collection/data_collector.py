"""
Autonomous Data Collection System for HRM Training
Collects diverse data sources for reasoning, tool use, and instruction following
"""
import os
import json
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
import time

import requests
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import T5Tokenizer

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    name: str
    url: str
    data_type: str  # 'reasoning', 'instruction', 'tool_use', 'conversation'
    collection_method: str  # 'api', 'scrape', 'dataset'
    priority: int = 1
    last_collected: Optional[datetime] = None
    collection_interval: timedelta = timedelta(hours=24)

class DataCollector:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.data_sources = self._initialize_data_sources()
        self.collected_data = []
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False)
        
        # Create data storage directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/reasoning", exist_ok=True)
        os.makedirs("data/instruction_following", exist_ok=True)
        os.makedirs("data/tool_use", exist_ok=True)
        os.makedirs("data/error_correction", exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load data collection configuration"""
        default_config = {
            "max_daily_samples": 10000,
            "quality_threshold": 0.7,
            "diversity_weight": 0.3,
            "reasoning_weight": 0.4,
            "instruction_weight": 0.2,
            "tool_use_weight": 0.1,
            "collection_batch_size": 100,
            "max_sequence_length": 512
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_data_sources(self) -> List[DataSource]:
        """Initialize diverse data sources for training"""
        sources = [
            # Reasoning datasets
            DataSource("gsm8k", "gsm8k", "reasoning", "dataset", priority=3),
            DataSource("math", "competition_math", "reasoning", "dataset", priority=3),
            DataSource("arc", "ai2_arc", "reasoning", "dataset", priority=2),
            DataSource("hellaswag", "hellaswag", "reasoning", "dataset", priority=2),
            
            # Instruction following
            DataSource("alpaca", "tatsu-lab/alpaca", "instruction", "dataset", priority=3),
            DataSource("dolly", "databricks/databricks-dolly-15k", "instruction", "dataset", priority=2),
            DataSource("oasst1", "OpenAssistant/oasst1", "instruction", "dataset", priority=2),
            
            # Tool use and function calling
            DataSource("toolbench", "qiantong-xu/toolbench", "tool_use", "dataset", priority=3),
            DataSource("api_bank", "ArtifactAI/APIBank", "tool_use", "dataset", priority=2),
            
            # Code and programming
            DataSource("code_alpaca", "sahil2801/CodeAlpaca-20k", "instruction", "dataset", priority=2),
            DataSource("python_code", "codeparrot/github-code", "tool_use", "dataset", priority=1),
            
            # Conversational data
            DataSource("sharegpt", "anon8231489123/ShareGPT_Vicuna_unfiltered", "conversation", "dataset", priority=1),
            
            # Web scraping sources (for real-time data)
            DataSource("arxiv_recent", "http://export.arxiv.org/api/query", "reasoning", "api", priority=1),
            DataSource("github_issues", "https://api.github.com/search/issues", "tool_use", "api", priority=1),
            DataSource("stackoverflow", "https://api.stackexchange.com/2.3/questions", "instruction", "api", priority=1),
        ]
        
        return sources
    
    async def collect_all_data(self) -> Dict[str, List[Dict]]:
        """Collect data from all sources asynchronously"""
        logger.info("Starting comprehensive data collection...")
        
        collected_data = {
            "reasoning": [],
            "instruction": [],
            "tool_use": [],
            "conversation": [],
            "error_correction": []
        }
        
        # Collect from each data source
        for source in self.data_sources:
            try:
                if self._should_collect(source):
                    logger.info(f"Collecting from {source.name}...")
                    data = await self._collect_from_source(source)
                    
                    if data:
                        processed_data = self._process_data(data, source.data_type)
                        collected_data[source.data_type].extend(processed_data)
                        source.last_collected = datetime.now()
                        
                        logger.info(f"Collected {len(processed_data)} samples from {source.name}")
                    
            except Exception as e:
                logger.error(f"Error collecting from {source.name}: {e}")
                continue
        
        # Generate synthetic error correction data
        error_data = self._generate_error_correction_data()
        collected_data["error_correction"].extend(error_data)
        
        # Save collected data
        self._save_collected_data(collected_data)
        
        return collected_data
    
    def _should_collect(self, source: DataSource) -> bool:
        """Determine if data should be collected from this source"""
        if source.last_collected is None:
            return True
        
        time_since_last = datetime.now() - source.last_collected
        return time_since_last >= source.collection_interval
    
    async def _collect_from_source(self, source: DataSource) -> List[Dict]:
        """Collect data from a specific source"""
        if source.collection_method == "dataset":
            return await self._collect_from_dataset(source)
        elif source.collection_method == "api":
            return await self._collect_from_api(source)
        elif source.collection_method == "scrape":
            return await self._collect_from_scraping(source)
        else:
            logger.warning(f"Unknown collection method: {source.collection_method}")
            return []
    
    async def _collect_from_dataset(self, source: DataSource) -> List[Dict]:
        """Collect data from HuggingFace datasets"""
        try:
            if source.name == "gsm8k":
                dataset = load_dataset("gsm8k", "main", split="train")
                return [{"question": item["question"], "answer": item["answer"], "type": "math_reasoning"} 
                       for item in dataset.select(range(min(1000, len(dataset))))]
            
            elif source.name == "math":
                dataset = load_dataset("competition_math", split="train")
                return [{"problem": item["problem"], "solution": item["solution"], "type": "competition_math"} 
                       for item in dataset.select(range(min(500, len(dataset))))]
            
            elif source.name == "alpaca":
                dataset = load_dataset("tatsu-lab/alpaca", split="train")
                return [{"instruction": item["instruction"], "input": item.get("input", ""), 
                        "output": item["output"], "type": "instruction_following"} 
                       for item in dataset.select(range(min(2000, len(dataset))))]
            
            elif source.name == "toolbench":
                try:
                    dataset = load_dataset("qiantong-xu/toolbench", split="train")
                    return [{"query": item.get("query", ""), "tools": item.get("tools", []), 
                            "solution": item.get("solution", ""), "type": "tool_use"} 
                           for item in dataset.select(range(min(1000, len(dataset))))]
                except:
                    # Fallback to synthetic tool use data
                    return self._generate_synthetic_tool_data()
            
            elif source.name == "code_alpaca":
                dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
                return [{"instruction": item["instruction"], "input": item.get("input", ""), 
                        "output": item["output"], "type": "code_instruction"} 
                       for item in dataset.select(range(min(1500, len(dataset))))]
            
            else:
                # Generic dataset handling
                try:
                    dataset = load_dataset(source.url, split="train")
                    return [dict(item) for item in dataset.select(range(min(1000, len(dataset))))]
                except:
                    logger.warning(f"Could not load dataset {source.url}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error loading dataset {source.name}: {e}")
            return []
    
    async def _collect_from_api(self, source: DataSource) -> List[Dict]:
        """Collect data from APIs"""
        try:
            if source.name == "arxiv_recent":
                return await self._collect_arxiv_papers()
            elif source.name == "github_issues":
                return await self._collect_github_issues()
            elif source.name == "stackoverflow":
                return await self._collect_stackoverflow_questions()
            else:
                return []
        except Exception as e:
            logger.error(f"Error collecting from API {source.name}: {e}")
            return []
    
    async def _collect_arxiv_papers(self) -> List[Dict]:
        """Collect recent AI/ML papers from arXiv"""
        try:
            query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
            url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Parse XML and extract paper information
                        # This is a simplified version - you'd want proper XML parsing
                        papers = []
                        # Add paper parsing logic here
                        return papers
            return []
        except Exception as e:
            logger.error(f"Error collecting arXiv papers: {e}")
            return []
    
    async def _collect_github_issues(self) -> List[Dict]:
        """Collect GitHub issues for tool use examples"""
        try:
            # This would require GitHub API token
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error collecting GitHub issues: {e}")
            return []
    
    async def _collect_stackoverflow_questions(self) -> List[Dict]:
        """Collect StackOverflow questions for instruction following"""
        try:
            # This would require StackExchange API
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error collecting StackOverflow questions: {e}")
            return []
    
    async def _collect_from_scraping(self, source: DataSource) -> List[Dict]:
        """Collect data through web scraping"""
        # Implement web scraping logic here
        return []
    
    def _generate_synthetic_tool_data(self) -> List[Dict]:
        """Generate synthetic tool use training data"""
        tools = [
            {"name": "calculator", "description": "Perform mathematical calculations"},
            {"name": "web_search", "description": "Search the web for information"},
            {"name": "file_reader", "description": "Read and analyze files"},
            {"name": "code_executor", "description": "Execute code snippets"},
            {"name": "data_analyzer", "description": "Analyze datasets and generate insights"},
        ]
        
        synthetic_data = []
        for _ in range(200):
            tool = random.choice(tools)
            query = f"I need to use {tool['name']} to {tool['description'].lower()}"
            solution = f"I'll use the {tool['name']} tool. {tool['description']}."
            
            synthetic_data.append({
                "query": query,
                "tools": [tool],
                "solution": solution,
                "type": "synthetic_tool_use"
            })
        
        return synthetic_data
    
    def _generate_error_correction_data(self) -> List[Dict]:
        """Generate data for error correction training"""
        error_patterns = [
            {
                "incorrect": "The answer is 42 because I calculated 20 + 20 = 42",
                "correct": "The answer is 40 because I calculated 20 + 20 = 40",
                "error_type": "arithmetic_error"
            },
            {
                "incorrect": "To solve this, I'll use the wrong_tool function",
                "correct": "To solve this, I'll use the appropriate tool based on the task requirements",
                "error_type": "tool_selection_error"
            },
            {
                "incorrect": "The instruction asks for X but I'll do Y instead",
                "correct": "The instruction asks for X, so I'll focus on providing exactly what was requested",
                "error_type": "instruction_following_error"
            }
        ]
        
        error_data = []
        for _ in range(300):
            pattern = random.choice(error_patterns)
            error_data.append({
                "incorrect_response": pattern["incorrect"],
                "correct_response": pattern["correct"],
                "error_type": pattern["error_type"],
                "type": "error_correction"
            })
        
        return error_data
    
    def _process_data(self, data: List[Dict], data_type: str) -> List[Dict]:
        """Process and clean collected data"""
        processed = []
        
        for item in data:
            try:
                # Basic cleaning and formatting
                processed_item = self._clean_item(item, data_type)
                
                # Quality filtering
                if self._passes_quality_check(processed_item):
                    processed.append(processed_item)
                    
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
        
        # Ensure diversity
        processed = self._ensure_diversity(processed, data_type)
        
        return processed
    
    def _clean_item(self, item: Dict, data_type: str) -> Dict:
        """Clean and format individual data items"""
        cleaned = {}
        
        # Remove None values and empty strings
        for key, value in item.items():
            if value is not None and value != "":
                if isinstance(value, str):
                    cleaned[key] = value.strip()
                else:
                    cleaned[key] = value
        
        # Add metadata
        cleaned["data_type"] = data_type
        cleaned["collected_at"] = datetime.now().isoformat()
        cleaned["source"] = "autonomous_collector"
        
        return cleaned
    
    def _passes_quality_check(self, item: Dict) -> bool:
        """Check if data item meets quality standards"""
        # Basic quality checks
        if not item:
            return False
        
        # Check for minimum content length
        content_fields = ["question", "instruction", "query", "problem", "text"]
        has_content = False
        
        for field in content_fields:
            if field in item and len(str(item[field])) > 10:
                has_content = True
                break
        
        if not has_content:
            return False
        
        # Check for toxic content (basic filtering)
        toxic_keywords = ["hate", "violence", "explicit", "harmful"]
        text_content = " ".join([str(v) for v in item.values()]).lower()
        
        for keyword in toxic_keywords:
            if keyword in text_content:
                return False
        
        return True
    
    def _ensure_diversity(self, data: List[Dict], data_type: str) -> List[Dict]:
        """Ensure diversity in collected data"""
        if len(data) <= 100:
            return data
        
        # Simple diversity sampling based on content similarity
        # In a more sophisticated implementation, you'd use embeddings
        diverse_data = []
        seen_patterns = set()
        
        for item in data:
            # Create a simple pattern hash
            content = str(item.get("question", "") + item.get("instruction", "") + item.get("query", ""))
            pattern = hash(content[:100]) % 1000
            
            if pattern not in seen_patterns or len(diverse_data) < 50:
                diverse_data.append(item)
                seen_patterns.add(pattern)
            
            if len(diverse_data) >= min(500, len(data)):
                break
        
        return diverse_data
    
    def _save_collected_data(self, collected_data: Dict[str, List[Dict]]):
        """Save collected data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, data_list in collected_data.items():
            if data_list:
                filename = f"data/{data_type}/collected_{timestamp}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                with open(filename, 'w') as f:
                    json.dump(data_list, f, indent=2)
                
                logger.info(f"Saved {len(data_list)} {data_type} samples to {filename}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about data collection"""
        stats = {
            "total_sources": len(self.data_sources),
            "active_sources": len([s for s in self.data_sources if s.last_collected]),
            "last_collection": max([s.last_collected for s in self.data_sources if s.last_collected], default=None),
            "data_types": {}
        }
        
        # Count data by type
        for data_type in ["reasoning", "instruction", "tool_use", "conversation", "error_correction"]:
            data_dir = f"data/{data_type}"
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
                total_samples = 0
                for file in files:
                    try:
                        with open(os.path.join(data_dir, file), 'r') as f:
                            data = json.load(f)
                            total_samples += len(data)
                    except:
                        continue
                stats["data_types"][data_type] = {
                    "files": len(files),
                    "samples": total_samples
                }
        
        return stats

# Example usage
async def main():
    collector = DataCollector()
    collected_data = await collector.collect_all_data()
    stats = collector.get_collection_stats()
    print(f"Collection complete. Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())