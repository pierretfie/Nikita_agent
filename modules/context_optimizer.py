#!/usr/bin/env python3
"""
Context Optimizer Module for Nikita Agent

Optimizes conversation context for LLM interactions by selecting relevant
messages, handling token limits, and improving prompt quality.
"""

import re
import psutil
from datetime import datetime
import json
from .engagement_manager import engagement_memory

# Default token limits
DEFAULT_MAX_TOKENS = 2048
DEFAULT_RESERVE_TOKENS = 512

class ContextOptimizer:
    def __init__(self, max_tokens=DEFAULT_MAX_TOKENS, reserve_tokens=DEFAULT_RESERVE_TOKENS, 
                engagement_memory=None, memory_limit=20):
        """
        Initialize the context optimizer.
        
        Args:
            max_tokens (int): Maximum token limit for context window
            reserve_tokens (int): Tokens to reserve for model response
            engagement_memory (dict, optional): Dictionary of engagement memory
            memory_limit (int): Maximum number of messages to keep in memory
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.cache = {}  # Cache for frequently used contexts
        self.prompt_cache = {}  # Cache for generated prompts
        self.engagement_memory = engagement_memory or {}
        self.memory_limit = memory_limit
        self.tool_context_cache = {}  # Cache for tool contexts
        self.cache_limit = 50 # Limit cache size
        
    def format_tool_context(self, tool_context):
        """Format tool context into a readable string for the model"""
        if not tool_context:
            return ""
            
        formatted = []
        
        # Format man page information
        if tool_context.get("man_page"):
            man_page = tool_context["man_page"]
            formatted.append("Tool Documentation:")
            if man_page.get("name"):
                formatted.append(f"Name: {man_page['name']}")
            if man_page.get("synopsis"):
                formatted.append(f"Usage: {man_page['synopsis']}")
            if man_page.get("description"):
                formatted.append(f"Description: {man_page['description']}")
            if man_page.get("options"):
                formatted.append(f"Options: {man_page['options']}")
            if man_page.get("examples"):
                formatted.append(f"Examples: {man_page['examples']}")
        
        # Format fine-tuning data
        if tool_context.get("fine_tuning"):
            formatted.append("\nCommon Use Cases:")
            for entry in tool_context["fine_tuning"]:
                formatted.append(f"- {entry.get('instruction', '')}")
                if entry.get("command"):
                    formatted.append(f"  Command: {entry['command']}")
        
        # Format common usage patterns
        if tool_context.get("common_usage"):
            formatted.append("\nCommon Usage Patterns:")
            for pattern_name, pattern in tool_context["common_usage"].items():
                formatted.append(f"- {pattern_name}: {pattern}")
        
        return "\n".join(formatted)

    def optimize_context(self, chat_memory, current_task, targets=None):
        """
        Optimize context window by selecting relevant messages.
        
        Args:
            chat_memory (list): List of chat messages (dicts with 'role', 'content')
            current_task (str): Current user task/query
            targets (list, optional): List of targets (IPs, etc.) to prioritize
            
        Returns:
            list: List of relevant context messages
        """
        # Check cache first for performance
        cache_key = f"{current_task}_{len(chat_memory)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Handle empty history
        if not chat_memory:
            return []
            
        # Process only recent messages to save processing time
        recent_messages = chat_memory[-min(self.memory_limit, len(chat_memory)):]
        
        # Faster relevance scoring - avoid complex calculations
        scored_messages = []
        
        # Focus on just the last 3 messages for faster processing
        if len(recent_messages) <= 3:
            # Just return all messages if 3 or fewer
            relevant_msgs = [msg['content'] for msg in recent_messages if isinstance(msg, dict) and msg.get('content')]
            self.cache[cache_key] = relevant_msgs
            return relevant_msgs
            
        # Get last 3 messages directly - fast path optimization
        relevant_msgs = [msg['content'] for msg in recent_messages[-3:] 
                         if isinstance(msg, dict) and msg.get('content')]
        
        # Cache the result
        self.cache[cache_key] = relevant_msgs
        
        # Limit cache size to prevent memory growth
        if len(self.cache) > 50:
            # Remove oldest entries (simple approach)
            keys_to_remove = list(self.cache.keys())[:-25]  # Keep 25 newest items
            for key in keys_to_remove:
                self.cache.pop(key, None)
                
        return relevant_msgs

    def get_optimized_prompt(self, chat_memory, current_task, base_prompt, reasoning_context=None, 
                           follow_up_questions=None, tool_context=None, intent_analysis_context=None):
        """
        Generate an optimized prompt for the LLM, incorporating various context elements
        and using caching.
        """
        # --- Cache Key Generation (Ultra-Simplified) ---
        # Use lengths and basic string representations to guarantee hashability
        key_part_base_prompt = str(base_prompt)
        key_part_current_task = str(current_task)
        key_part_chat_len = len(chat_memory)
        key_part_reasoning_len = len(json.dumps(reasoning_context)) if reasoning_context else 0
        key_part_followup_len = len(follow_up_questions) if follow_up_questions else 0
        # Keep a hash of follow-up content for some sensitivity, ensuring strings
        key_part_followup_content_hash = hash(tuple(str(q) for q in follow_up_questions)) if follow_up_questions else 0
        key_part_tool_context_str = str(tool_context) if tool_context else ""
        key_part_intent_len = len(json.dumps(intent_analysis_context)) if intent_analysis_context else 0

        cache_key = (
            key_part_base_prompt,
            key_part_current_task,
            key_part_chat_len,
            key_part_reasoning_len,
            key_part_followup_len,
            key_part_followup_content_hash, # Mix of length and content hash
            key_part_tool_context_str,
            key_part_intent_len
        )

        # Check cache
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        # --- Context Assembly ---
        optimized_context = []
        # Add recent conversation history (e.g., last 5 messages)
        context_messages = chat_memory[-5:]
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in context_messages])

        # --- Incorporate Reasoning & Follow-ups --- 
        reasoning_str = ""
        if reasoning_context:
             # Add active targets if available in engagement memory (part of reasoning context now)
             active_targets = engagement_memory.get("targets", [])
             target_str = f"Active Targets: {', '.join(active_targets)}\n" if active_targets else ""
             # Format the reasoning dictionary for inclusion
             try:
                 # Pretty print JSON for readability in the prompt
                 reasoning_detail = json.dumps(reasoning_context, indent=2)
                 reasoning_str = f"\nReasoning Context:\n{target_str}{reasoning_detail}\n"
             except TypeError:
                 reasoning_str = f"\nReasoning Context:\n{target_str}(Could not serialize reasoning detail)\n"

        follow_up_str = ""
        if follow_up_questions:
             follow_up_str = "\nFollow-up Questions:\n" + "\n".join([f"- {q}" for q in follow_up_questions]) + "\n"

        # --- Tool Context ---
        tool_context_str_formatted = f"\nTool Context:\n{tool_context}\n" if tool_context else ""

        # --- Intent Analysis Context (Optional inclusion) ---
        # Decide if/how to include intent analysis details. Might be redundant with reasoning context.
        # intent_str = f"\nIntent Analysis:\n{json.dumps(intent_analysis_context, indent=2)}\n" if intent_analysis_context else ""

        # --- Construct Final Prompt --- 
        # Use the base_prompt template and fill placeholders
        # Assuming base_prompt has placeholders like {chat_history}, {reasoning_context}, etc.
        # If not, adjust the final string construction accordingly.
        final_prompt = base_prompt # Start with the template

        # Replace placeholders - use .replace() for safety if template structure varies
        final_prompt = final_prompt.replace("{chat_history}", history_str)
        final_prompt = final_prompt.replace("{reasoning_context}", reasoning_str) # Placeholder for combined reasoning
        final_prompt = final_prompt.replace("{follow_up_questions}", follow_up_str) # Placeholder if needed
        final_prompt = final_prompt.replace("{tool_context}", tool_context_str_formatted)
        final_prompt = final_prompt.replace("{current_task}", current_task)
        # Add other replacements if the template uses different placeholders

        # Fallback if template doesn't have placeholders (simple concatenation)
        # This part might need adjustment based on how `base_prompt` is structured.
        # If the base_prompt IS the full structure already, this might be simpler:
        # final_prompt = f"{base_prompt}\n\n{history_str}{reasoning_str}{follow_up_str}{tool_context_str_formatted}\nTask: {current_task}\nResponse:"


        # --- Token Limit Check (Simplified) ---
        # A proper tokenizer would be needed for accurate count. This is an estimate.
        estimated_tokens = len(final_prompt.split())
        if estimated_tokens > (self.max_tokens - self.reserve_tokens):
            # Basic truncation strategy: trim history first
            print(f"[ContextOptimizer] Warning: Estimated prompt tokens ({estimated_tokens}) exceed limit. Truncating history.")
            # More sophisticated truncation could be implemented here
            context_messages = chat_memory[-3:] # Reduce history further
            history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in context_messages])
            # Rebuild prompt with truncated history
            final_prompt = base_prompt
            final_prompt = final_prompt.replace("{chat_history}", history_str)
            final_prompt = final_prompt.replace("{reasoning_context}", reasoning_str)
            final_prompt = final_prompt.replace("{follow_up_questions}", follow_up_str)
            final_prompt = final_prompt.replace("{tool_context}", tool_context_str_formatted)
            final_prompt = final_prompt.replace("{current_task}", current_task)


        # --- Cache Management ---
        # Remove oldest entry if cache exceeds limit
        if len(self.prompt_cache) >= self.cache_limit:
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]
        
        # Cache the final prompt
        self.prompt_cache[cache_key] = final_prompt

        return final_prompt
        
    def clear_cache(self):
        """Clear the internal cache to free memory"""
        self.cache.clear()
        self.prompt_cache.clear()
        
    def update_memory_limit(self, new_limit):
        """Update the memory limit for messages"""
        if new_limit > 0:
            self.memory_limit = new_limit
            
    def estimate_tokens(self, text):
        """
        Roughly estimate the number of tokens in text - simplified for speed
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Very rough approximation, about 4 chars per token on average
        # Fast path for empty or small text
        if not text or len(text) < 100:
            return len(text) // 4 + 1
            
        return len(text) // 4

if __name__ == "__main__":
    # Simple self-test
    optimizer = ContextOptimizer()
    
    # Test with simple chat memory
    chat_memory = [
        {"role": "user", "content": "How do I scan a network?"},
        {"role": "assistant", "content": "You can use nmap for network scanning."},
        {"role": "user", "content": "Show me an example for 192.168.1.0/24"}
    ]
    
    current_task = "How do I scan for specific services?"
    
    optimized_context = optimizer.optimize_context(chat_memory, current_task)
    print("Optimized Context:")
    for ctx in optimized_context:
        print(f"- {ctx}")
        
    prompt = optimizer.get_optimized_prompt(
        chat_memory, 
        current_task, 
        "You are Nikita, a security assistant."
    )
    
    print("\nFull Prompt:")
    print(prompt) 