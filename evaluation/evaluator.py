"""
Comprehensive Model Evaluation System
Evaluates reasoning capabilities, instruction following, tool use, and error correction
"""
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import random
import re

import torch
import torch.nn.functional as F
import numpy as np
from transformers import T5Tokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.evaluation_history = []
        self.benchmark_cache = {}
        
        # Load evaluation datasets
        self._load_evaluation_datasets()
        
        # Define evaluation metrics
        self.metrics = {
            "reasoning_accuracy": 0.0,
            "instruction_following": 0.0,
            "tool_use_success": 0.0,
            "error_correction": 0.0,
            "overall_confidence": 0.0,
            "reasoning_efficiency": 0.0,
            "response_quality": 0.0
        }
    
    def _load_evaluation_datasets(self):
        """Load datasets for evaluation"""
        try:
            # Math reasoning
            self.gsm8k_test = load_dataset("gsm8k", "main", split="test")
            logger.info("Loaded GSM8K test set")
        except:
            logger.warning("Could not load GSM8K test set")
            self.gsm8k_test = None
        
        try:
            # Instruction following
            self.alpaca_eval = load_dataset("tatsu-lab/alpaca_eval", split="eval")
            logger.info("Loaded Alpaca eval set")
        except:
            logger.warning("Could not load Alpaca eval set")
            self.alpaca_eval = None
        
        # Create synthetic evaluation sets
        self._create_synthetic_evaluations()
    
    def _create_synthetic_evaluations(self):
        """Create synthetic evaluation datasets"""
        # Tool use evaluation
        self.tool_use_eval = [
            {
                "query": "Calculate the square root of 144 and then multiply by 3",
                "expected_tools": ["calculator"],
                "expected_answer": "36",
                "difficulty": "easy"
            },
            {
                "query": "Search for information about machine learning and summarize the key concepts",
                "expected_tools": ["web_search", "text_summarizer"],
                "expected_answer": "summary of ML concepts",
                "difficulty": "medium"
            },
            {
                "query": "Read the file 'data.csv', analyze the trends, and create a visualization",
                "expected_tools": ["file_reader", "data_analyzer", "visualization_tool"],
                "expected_answer": "analysis with visualization",
                "difficulty": "hard"
            }
        ]
        
        # Error correction evaluation
        self.error_correction_eval = [
            {
                "incorrect": "The capital of France is London",
                "correct": "The capital of France is Paris",
                "error_type": "factual_error"
            },
            {
                "incorrect": "To calculate 15 + 27, I get 52",
                "correct": "To calculate 15 + 27, I get 42",
                "error_type": "arithmetic_error"
            },
            {
                "incorrect": "I'll use the wrong_function() to solve this problem",
                "correct": "I'll use the appropriate function based on the problem requirements",
                "error_type": "tool_selection_error"
            }
        ]
        
        # Reasoning evaluation
        self.reasoning_eval = [
            {
                "problem": "If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours?",
                "expected_answer": "150 miles",
                "reasoning_steps": ["identify speed: 60 mph", "multiply by time: 60 * 2.5", "result: 150 miles"],
                "difficulty": "easy"
            },
            {
                "problem": "A company has 100 employees. 60% work in engineering, 25% in sales, and the rest in administration. If engineering gets a 20% budget increase and sales gets 10%, what's the weighted average increase?",
                "expected_answer": "16%",
                "reasoning_steps": ["engineering: 60% * 20% = 12%", "sales: 25% * 10% = 2.5%", "admin: 15% * 0% = 0%", "total: 12% + 2.5% + 0% = 14.5%"],
                "difficulty": "hard"
            }
        ]
    
    async def comprehensive_evaluation(self, model, tokenizer) -> Dict[str, float]:
        """Perform comprehensive model evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # 1. Reasoning evaluation
        reasoning_score = await self._evaluate_reasoning(model, tokenizer)
        results["reasoning_accuracy"] = reasoning_score
        
        # 2. Instruction following evaluation
        instruction_score = await self._evaluate_instruction_following(model, tokenizer)
        results["instruction_following"] = instruction_score
        
        # 3. Tool use evaluation
        tool_use_score = await self._evaluate_tool_use(model, tokenizer)
        results["tool_use_success"] = tool_use_score
        
        # 4. Error correction evaluation
        error_correction_score = await self._evaluate_error_correction(model, tokenizer)
        results["error_correction"] = error_correction_score
        
        # 5. Confidence evaluation
        confidence_score = await self._evaluate_confidence(model, tokenizer)
        results["overall_confidence"] = confidence_score
        
        # 6. Efficiency evaluation
        efficiency_score = await self._evaluate_efficiency(model, tokenizer)
        results["reasoning_efficiency"] = efficiency_score
        
        # 7. Response quality evaluation
        quality_score = await self._evaluate_response_quality(model, tokenizer)
        results["response_quality"] = quality_score
        
        # Calculate overall score
        weights = {
            "reasoning_accuracy": 0.25,
            "instruction_following": 0.20,
            "tool_use_success": 0.15,
            "error_correction": 0.15,
            "overall_confidence": 0.10,
            "reasoning_efficiency": 0.10,
            "response_quality": 0.05
        }
        
        overall_score = sum(results[metric] * weight for metric, weight in weights.items())
        results["overall"] = overall_score
        
        # Store evaluation history
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "model_state": self._get_model_state_summary(model)
        }
        self.evaluation_history.append(evaluation_record)
        
        logger.info(f"Evaluation complete. Overall score: {overall_score:.3f}")
        return results
    
    async def _evaluate_reasoning(self, model, tokenizer) -> float:
        """Evaluate mathematical and logical reasoning"""
        correct = 0
        total = 0
        
        # Use GSM8K if available
        if self.gsm8k_test:
            test_samples = list(self.gsm8k_test.select(range(min(50, len(self.gsm8k_test)))))
        else:
            test_samples = []
        
        # Add synthetic reasoning problems
        test_samples.extend(self.reasoning_eval)
        
        for sample in test_samples:
            try:
                if "question" in sample and "answer" in sample:
                    # GSM8K format
                    question = sample["question"]
                    expected = self._extract_number_from_answer(sample["answer"])
                elif "problem" in sample:
                    # Synthetic format
                    question = sample["problem"]
                    expected = sample["expected_answer"]
                else:
                    continue
                
                # Generate response
                response = await self._generate_response(model, tokenizer, f"Solve this problem: {question}")
                
                # Extract answer from response
                predicted = self._extract_answer_from_response(response)
                
                # Check correctness
                if self._answers_match(predicted, expected):
                    correct += 1
                
                total += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating reasoning sample: {e}")
                continue
        
        return correct / max(total, 1)
    
    async def _evaluate_instruction_following(self, model, tokenizer) -> float:
        """Evaluate instruction following capabilities"""
        correct = 0
        total = 0
        
        # Create instruction following test cases
        test_cases = [
            {
                "instruction": "Write a haiku about technology",
                "criteria": ["3 lines", "5-7-5 syllable pattern", "technology theme"]
            },
            {
                "instruction": "List 5 benefits of renewable energy",
                "criteria": ["exactly 5 items", "renewable energy topic", "benefits listed"]
            },
            {
                "instruction": "Explain photosynthesis in simple terms for a 10-year-old",
                "criteria": ["simple language", "photosynthesis explanation", "age-appropriate"]
            },
            {
                "instruction": "Convert 100 degrees Fahrenheit to Celsius and show your work",
                "criteria": ["conversion formula", "correct answer: 37.78°C", "work shown"]
            }
        ]
        
        for test_case in test_cases:
            try:
                instruction = test_case["instruction"]
                criteria = test_case["criteria"]
                
                # Generate response
                response = await self._generate_response(model, tokenizer, instruction)
                
                # Evaluate against criteria
                score = self._evaluate_against_criteria(response, criteria)
                correct += score
                total += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating instruction following: {e}")
                continue
        
        return correct / max(total, 1)
    
    async def _evaluate_tool_use(self, model, tokenizer) -> float:
        """Evaluate tool use and function calling"""
        correct = 0
        total = 0
        
        for test_case in self.tool_use_eval:
            try:
                query = test_case["query"]
                expected_tools = test_case["expected_tools"]
                
                # Generate response with tool use context
                prompt = f"Query: {query}\nAvailable tools: calculator, web_search, file_reader, data_analyzer, text_summarizer, visualization_tool\nSolution:"
                response = await self._generate_response(model, tokenizer, prompt, return_tool_outputs=True)
                
                # Check if appropriate tools were mentioned/used
                tools_mentioned = self._extract_tools_from_response(response)
                
                # Calculate score based on tool selection accuracy
                score = self._calculate_tool_use_score(tools_mentioned, expected_tools)
                correct += score
                total += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating tool use: {e}")
                continue
        
        return correct / max(total, 1)
    
    async def _evaluate_error_correction(self, model, tokenizer) -> float:
        """Evaluate error detection and correction capabilities"""
        correct = 0
        total = 0
        
        for test_case in self.error_correction_eval:
            try:
                incorrect = test_case["incorrect"]
                correct_answer = test_case["correct"]
                
                # Test error detection
                detection_prompt = f"Is there an error in this statement? '{incorrect}'"
                detection_response = await self._generate_response(model, tokenizer, detection_prompt)
                
                # Test error correction
                correction_prompt = f"Correct this statement: '{incorrect}'"
                correction_response = await self._generate_response(model, tokenizer, correction_prompt)
                
                # Evaluate responses
                detected_error = "error" in detection_response.lower() or "incorrect" in detection_response.lower()
                corrected_properly = self._check_correction_quality(correction_response, correct_answer)
                
                if detected_error and corrected_properly:
                    correct += 1
                elif detected_error or corrected_properly:
                    correct += 0.5
                
                total += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating error correction: {e}")
                continue
        
        return correct / max(total, 1)
    
    async def _evaluate_confidence(self, model, tokenizer) -> float:
        """Evaluate model confidence calibration"""
        confidence_scores = []
        
        # Test confidence on various tasks
        test_prompts = [
            "What is 2 + 2?",  # Easy - should be high confidence
            "What is the capital of France?",  # Easy - should be high confidence
            "Explain quantum entanglement in detail",  # Hard - should be lower confidence
            "What will the stock market do tomorrow?"  # Impossible - should be very low confidence
        ]
        
        for prompt in test_prompts:
            try:
                response = await self._generate_response(model, tokenizer, prompt, return_confidence=True)
                
                # Extract confidence if available
                if hasattr(model, 'confidence_head'):
                    # Get confidence from model
                    with torch.no_grad():
                        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
                        inputs = inputs.to(model.device)
                        outputs = model(inputs, return_tool_outputs=True)
                        
                        if "reasoning_confidence" in outputs:
                            confidence = torch.mean(torch.stack(outputs["reasoning_confidence"])).item()
                            confidence_scores.append(confidence)
                
            except Exception as e:
                logger.warning(f"Error evaluating confidence: {e}")
                continue
        
        # Return average confidence (higher is better for calibration)
        return np.mean(confidence_scores) if confidence_scores else 0.5
    
    async def _evaluate_efficiency(self, model, tokenizer) -> float:
        """Evaluate reasoning efficiency (fewer steps for same quality)"""
        efficiency_scores = []
        
        test_problems = [
            "Calculate 15 * 8",
            "What is the area of a circle with radius 5?",
            "If I have 10 apples and eat 3, how many are left?"
        ]
        
        for problem in test_problems:
            try:
                response = await self._generate_response(model, tokenizer, problem, return_tool_outputs=True)
                
                # Count reasoning steps (if available)
                if hasattr(model, 'performance_history'):
                    recent_performance = model.performance_history[-1:] if model.performance_history else []
                    if recent_performance:
                        steps = recent_performance[0].get("reasoning_steps", model.max_steps)
                        # Efficiency is inverse of steps used (normalized)
                        efficiency = 1.0 - (steps / model.max_steps)
                        efficiency_scores.append(max(0.0, efficiency))
                
            except Exception as e:
                logger.warning(f"Error evaluating efficiency: {e}")
                continue
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.5
    
    async def _evaluate_response_quality(self, model, tokenizer) -> float:
        """Evaluate overall response quality"""
        quality_scores = []
        
        test_prompts = [
            "Explain the concept of machine learning",
            "How do you make a paper airplane?",
            "What are the benefits of exercise?"
        ]
        
        for prompt in test_prompts:
            try:
                response = await self._generate_response(model, tokenizer, prompt)
                
                # Basic quality metrics
                quality_score = 0.0
                
                # Length check (not too short, not too long)
                if 50 <= len(response) <= 500:
                    quality_score += 0.3
                
                # Coherence check (basic)
                if self._is_coherent_response(response):
                    quality_score += 0.4
                
                # Relevance check
                if self._is_relevant_response(response, prompt):
                    quality_score += 0.3
                
                quality_scores.append(quality_score)
                
            except Exception as e:
                logger.warning(f"Error evaluating response quality: {e}")
                continue
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    async def _generate_response(self, model, tokenizer, prompt: str, 
                                return_tool_outputs: bool = False, 
                                return_confidence: bool = False) -> str:
        """Generate response from model"""
        try:
            model.eval()
            with torch.no_grad():
                # Tokenize input
                inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
                inputs = inputs.to(model.device)
                
                # Generate response
                outputs = model(inputs, return_tool_outputs=return_tool_outputs)
                logits = outputs["logits"]
                
                # Get predicted tokens (simple greedy decoding)
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode response
                response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                
                # Remove input from response
                input_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
                if response.startswith(input_text):
                    response = response[len(input_text):].strip()
                
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def _extract_number_from_answer(self, answer: str) -> str:
        """Extract numerical answer from text"""
        # Look for numbers in the answer
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        return numbers[-1] if numbers else ""
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from model response"""
        # Look for patterns like "The answer is X" or just numbers
        answer_patterns = [
            r"(?:answer is|equals?|result is)\s*([+-]?\d+\.?\d*)",
            r"([+-]?\d+\.?\d*)\s*(?:is the answer|is correct)",
            r"([+-]?\d+\.?\d*)"
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                return matches[-1]
        
        return ""
    
    def _answers_match(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected"""
        try:
            pred_num = float(predicted) if predicted else None
            exp_num = float(expected) if expected else None
            
            if pred_num is not None and exp_num is not None:
                return abs(pred_num - exp_num) < 0.01
            
            return predicted.strip().lower() == expected.strip().lower()
        except:
            return predicted.strip().lower() == expected.strip().lower()
    
    def _evaluate_against_criteria(self, response: str, criteria: List[str]) -> float:
        """Evaluate response against specific criteria"""
        score = 0.0
        
        for criterion in criteria:
            if self._meets_criterion(response, criterion):
                score += 1.0 / len(criteria)
        
        return score
    
    def _meets_criterion(self, response: str, criterion: str) -> bool:
        """Check if response meets a specific criterion"""
        criterion_lower = criterion.lower()
        response_lower = response.lower()
        
        if "3 lines" in criterion_lower:
            return len(response.split('\n')) >= 3
        elif "5 items" in criterion_lower or "exactly 5" in criterion_lower:
            # Count numbered items or bullet points
            items = len(re.findall(r'^\d+\.|\*|\-', response, re.MULTILINE))
            return items == 5
        elif "simple language" in criterion_lower:
            # Basic check for simple language (average word length)
            words = response.split()
            avg_length = sum(len(word) for word in words) / len(words) if words else 0
            return avg_length < 6
        elif "work shown" in criterion_lower:
            return "=" in response or "formula" in response_lower
        else:
            # Generic keyword matching
            keywords = criterion_lower.split()
            return any(keyword in response_lower for keyword in keywords)
    
    def _extract_tools_from_response(self, response: str) -> List[str]:
        """Extract mentioned tools from response"""
        tools = ["calculator", "web_search", "file_reader", "data_analyzer", "text_summarizer", "visualization_tool"]
        mentioned_tools = []
        
        response_lower = response.lower()
        for tool in tools:
            if tool.replace("_", " ") in response_lower or tool in response_lower:
                mentioned_tools.append(tool)
        
        return mentioned_tools
    
    def _calculate_tool_use_score(self, mentioned_tools: List[str], expected_tools: List[str]) -> float:
        """Calculate tool use accuracy score"""
        if not expected_tools:
            return 1.0 if not mentioned_tools else 0.5
        
        # Calculate precision and recall
        correct_tools = set(mentioned_tools) & set(expected_tools)
        precision = len(correct_tools) / len(mentioned_tools) if mentioned_tools else 0
        recall = len(correct_tools) / len(expected_tools)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _check_correction_quality(self, correction: str, expected: str) -> bool:
        """Check if correction is of good quality"""
        # Simple similarity check
        correction_words = set(correction.lower().split())
        expected_words = set(expected.lower().split())
        
        # Calculate word overlap
        overlap = len(correction_words & expected_words)
        total_words = len(expected_words)
        
        return overlap / total_words > 0.5 if total_words > 0 else False
    
    def _is_coherent_response(self, response: str) -> bool:
        """Basic coherence check"""
        # Check for basic sentence structure
        sentences = response.split('.')
        if len(sentences) < 2:
            return False
        
        # Check for reasonable sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        return 3 <= avg_sentence_length <= 30
    
    def _is_relevant_response(self, response: str, prompt: str) -> bool:
        """Check if response is relevant to prompt"""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        overlap = len(prompt_words & response_words)
        return overlap >= min(3, len(prompt_words) // 2)
    
    def _get_model_state_summary(self, model) -> Dict[str, Any]:
        """Get summary of model state for evaluation record"""
        return {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "max_halt_steps": getattr(model, 'max_steps', None),
            "performance_history_length": len(getattr(model, 'performance_history', []))
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation history"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        latest = self.evaluation_history[-1]
        
        # Calculate trends if we have multiple evaluations
        trends = {}
        if len(self.evaluation_history) >= 2:
            prev = self.evaluation_history[-2]
            for metric in latest["results"]:
                if metric in prev["results"]:
                    change = latest["results"][metric] - prev["results"][metric]
                    trends[metric] = "improving" if change > 0.01 else "declining" if change < -0.01 else "stable"
        
        return {
            "latest_results": latest["results"],
            "evaluation_count": len(self.evaluation_history),
            "trends": trends,
            "last_evaluation": latest["timestamp"]
        }

# Example usage
async def main():
    try:
        from ..core.hrm_model import EnhancedHierarchicalReasoningModel
    except ImportError:
        from core.hrm_model import EnhancedHierarchicalReasoningModel
    from transformers import T5Tokenizer
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Mock model and tokenizer for testing
    tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False)
    model_config = {
        "vocab_size": len(tokenizer),
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "halt_max_steps": 8,
        "ponder_loss_weight": 1e-2,
        "num_tools": 100
    }
    model = EnhancedHierarchicalReasoningModel(model_config)
    
    # Run evaluation
    results = await evaluator.comprehensive_evaluation(model, tokenizer)
    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    asyncio.run(main())