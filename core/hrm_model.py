"""
Enhanced Hierarchical Reasoning Model with Tool Use and Function Calling Capabilities
"""
import os, math, json, random, io, datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from datasets import load_dataset
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from huggingface_hub import HfApi, HfFolder, hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        activated = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(activated))

class HRMBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMuchPelu(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class ToolUseHead(nn.Module):
    """Head for tool use and function calling decisions"""
    def __init__(self, d_model, num_tools=100):
        super().__init__()
        self.tool_classifier = nn.Linear(d_model, num_tools)
        self.use_tool_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        tool_logits = self.tool_classifier(x)
        use_tool_prob = self.use_tool_gate(x)
        confidence = self.confidence_head(x)
        return {
            'tool_logits': tool_logits,
            'use_tool_prob': use_tool_prob,
            'confidence': confidence
        }

class ScratchpadModule(nn.Module):
    """Module for maintaining scratchpad memory during reasoning"""
    def __init__(self, d_model, max_scratchpad_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_scratchpad_length
        self.scratchpad_embeddings = nn.Embedding(max_scratchpad_length, d_model)
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.memory_bank = None
        
    def forward(self, current_state, step):
        if self.memory_bank is None:
            batch_size = current_state.size(0)
            self.memory_bank = torch.zeros(batch_size, self.max_length, self.d_model, 
                                         device=current_state.device)
        
        # Update scratchpad with current reasoning state
        pos_emb = self.scratchpad_embeddings(torch.tensor(step % self.max_length, 
                                                         device=current_state.device))
        
        # Gate mechanism to decide what to store
        combined = torch.cat([current_state.mean(dim=1), pos_emb.unsqueeze(0).expand(current_state.size(0), -1)], dim=-1)
        update_gate = self.update_gate(combined)
        
        # Update memory bank
        self.memory_bank[:, step % self.max_length] = update_gate * current_state.mean(dim=1)
        
        return self.memory_bank
    
    def reset(self):
        self.memory_bank = None

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])
        self.H_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
        self.L_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
        
        # Enhanced components
        self.tool_head = ToolUseHead(config["d_model"], config.get("num_tools", 100))
        self.scratchpad = ScratchpadModule(config["d_model"])
        
    def forward(self, z_H, z_L, step=0, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H
        z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z_H_input = z_H + z_L_new
        z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # Tool use decision
        tool_outputs = self.tool_head(z_H_new)
        
        # Update scratchpad
        scratchpad_memory = self.scratchpad(z_H_new, step)
        
        return z_H_new, z_L_new, tool_outputs, scratchpad_memory

class EnhancedHierarchicalReasoningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner_model = HRMInner(config)
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.halt_head = nn.Sequential(nn.Linear(config["d_model"], 1), nn.Sigmoid())
        self.max_steps = config["halt_max_steps"]
        self.ponder_loss_weight = config["ponder_loss_weight"]
        
        # Self-improvement components
        self.error_detection_head = nn.Sequential(
            nn.Linear(config["d_model"], 1),
            nn.Sigmoid()
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(config["d_model"], 1),
            nn.Sigmoid()
        )
        
        # Performance tracking
        self.performance_history = []
        self.error_patterns = {}

    def forward(self, input_ids, labels=None, attention_mask=None, return_tool_outputs=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        z_L = self.inner_model.token_embeddings(input_ids)
        z_H = torch.zeros_like(z_L)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        mask_val = -1e4
        causal = torch.zeros((seq_len, seq_len), device=device)
        causal = causal.masked_fill(torch.triu(torch.ones_like(causal), diagonal=1).bool(), mask_val)

        halting_probs = torch.zeros((batch_size, seq_len, self.max_steps), device=device)
        remainders = torch.ones((batch_size, seq_len), device=device)
        total_z_H = 0.1 * z_L.clone()
        
        # Track tool usage and reasoning steps
        tool_usage_history = []
        reasoning_confidence = []
        error_detection_scores = []

        eps = 1e-6
        for step in range(self.max_steps):
            p_halt = self.halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
            is_last = (step == self.max_steps - 1)
            halt_now_prob = torch.ones_like(p_halt) if is_last else p_halt
            contrib = (remainders * halt_now_prob).clamp(min=0.0, max=1.0)
            halting_probs[:, :, step] = contrib
            total_z_H = total_z_H + contrib.unsqueeze(-1) * z_H
            remainders = (remainders * (1 - p_halt)).clamp(min=eps, max=1.0)
            
            # Enhanced reasoning step with tool use and error detection
            z_H, z_L, tool_outputs, scratchpad_memory = self.inner_model(
                z_H, z_L, step, attn_mask=causal, key_padding_mask=key_padding_mask
            )
            
            # Track reasoning quality
            confidence = self.confidence_head(z_H).mean()
            error_score = self.error_detection_head(z_H).mean()
            
            tool_usage_history.append(tool_outputs)
            reasoning_confidence.append(confidence)
            error_detection_scores.append(error_score)
            
            if torch.all(remainders < 1e-4):
                break

        logits = self.lm_head(total_z_H)
        loss = ponder_loss = tool_loss = confidence_loss = None
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config["vocab_size"]), shift_labels.view(-1))
            ponder_loss = torch.mean(torch.sum(halting_probs, dim=-1))
            
            # Additional losses for enhanced capabilities
            confidence_loss = -torch.mean(torch.stack(reasoning_confidence))  # Encourage confidence
            
            loss = lm_loss + self.ponder_loss_weight * ponder_loss + 0.1 * confidence_loss
        
        outputs = {
            "loss": loss, 
            "logits": logits, 
            "ponder_loss": ponder_loss,
            "confidence_loss": confidence_loss,
            "reasoning_confidence": reasoning_confidence,
            "error_detection_scores": error_detection_scores
        }
        
        if return_tool_outputs:
            outputs["tool_usage_history"] = tool_usage_history
            
        return outputs
    
    def self_evaluate(self, outputs, targets=None):
        """Self-evaluation mechanism for continuous improvement"""
        confidence_scores = outputs.get("reasoning_confidence", [])
        error_scores = outputs.get("error_detection_scores", [])
        
        # Calculate performance metrics
        avg_confidence = torch.mean(torch.stack(confidence_scores)) if confidence_scores else 0.0
        avg_error_score = torch.mean(torch.stack(error_scores)) if error_scores else 0.0
        
        performance_metrics = {
            "confidence": avg_confidence.item() if torch.is_tensor(avg_confidence) else avg_confidence,
            "error_likelihood": avg_error_score.item() if torch.is_tensor(avg_error_score) else avg_error_score,
            "reasoning_steps": len(confidence_scores),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.performance_history.append(performance_metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return performance_metrics
    
    def get_improvement_suggestions(self):
        """Analyze performance history and suggest improvements"""
        if len(self.performance_history) < 10:
            return []
        
        recent_performance = self.performance_history[-10:]
        suggestions = []
        
        # Analyze confidence trends
        confidences = [p["confidence"] for p in recent_performance]
        if sum(confidences) / len(confidences) < 0.5:
            suggestions.append("Low confidence detected - consider additional training on reasoning tasks")
        
        # Analyze error patterns
        error_scores = [p["error_likelihood"] for p in recent_performance]
        if sum(error_scores) / len(error_scores) > 0.7:
            suggestions.append("High error likelihood - focus on error correction training")
        
        # Analyze reasoning efficiency
        reasoning_steps = [p["reasoning_steps"] for p in recent_performance]
        avg_steps = sum(reasoning_steps) / len(reasoning_steps)
        if avg_steps > self.max_steps * 0.8:
            suggestions.append("Inefficient reasoning - consider optimizing halt mechanism")
        
        return suggestions

    def reset_scratchpad(self):
        """Reset scratchpad memory for new reasoning session"""
        self.inner_model.scratchpad.reset()

class LLMDataset(Dataset):
    def __init__(self, hf_dataset, block_size):
        all_token_ids = [tid for doc in hf_dataset["input_ids"] for tid in doc]
        self.examples = [all_token_ids[i:i+block_size]
                         for i in range(0, len(all_token_ids) - block_size + 1, block_size)]
    
    def __len__(self): 
        return len(self.examples)
    
    def __getitem__(self, i): 
        return torch.tensor(self.examples[i], dtype=torch.long)

def create_enhanced_hrm_model(config):
    """Factory function to create enhanced HRM model"""
    return EnhancedHierarchicalReasoningModel(config)