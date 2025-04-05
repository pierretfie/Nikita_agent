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

# Default token limits
DEFAULT_MAX_TOKENS = 2048
DEFAULT_RESERVE_TOKENS = 512

class ContextOptimizer:
    def __init__(self, max_tokens=DEFAULT_MAX_TOKENS, reserve_tokens=DEFAULT_RESERVE_TOKENS, 
                engagement_memory=None, memory_limit=15):
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
        
        # Focus on just the last 15 messages for faster processing
        if len(recent_messages) <= 15:
            # Just return all messages if 15 or fewer
            relevant_msgs = [msg['content'] for msg in recent_messages if isinstance(msg, dict) and msg.get('content')]
            self.cache[cache_key] = relevant_msgs
            return relevant_msgs
            
        # Get last 15 messages directly - fast path optimization
        relevant_msgs = [msg['content'] for msg in recent_messages[-15:] 
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
                           follow_up_questions=None, tool_context=None):
        """
        Get an optimized prompt with context for the LLM.
        
        Args:
            chat_memory (list): List of chat messages
            current_task (str): Current user task/query
            base_prompt (str): Base system prompt/instruction
            reasoning_context (dict, optional): Context from reasoning engine
            follow_up_questions (list, optional): List of follow-up questions
            tool_context (dict, optional): Context about the tool being used
            
        Returns:
            str: Optimized prompt with context
        """
        # Check prompt cache first
        cache_key = f"{base_prompt}_{current_task}_{len(chat_memory)}"
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
            
        # Extract targets from memory if available
        targets = self.engagement_memory.get("targets", []) if self.engagement_memory else []
            
        # Get optimized context - keep last 5 messages for better continuity
        context_messages = []
        for msg in chat_memory[-5:]:
            if isinstance(msg, dict) and msg.get('content'):
                # Add role prefix for clarity
                role = msg.get('role', 'user')
                content = msg['content']
                context_messages.append(f"{role.upper()}: {content}")
        
        context_str = "\n".join(context_messages)
        
        # Format tool context if available
        tool_context_str = ""
        if tool_context:
            tool_context_str = self.format_tool_context(tool_context)
        
        # Format reasoning context if available
        reasoning_str = ""
        if reasoning_context:
            # Add target information to reasoning context if available
            if targets:
                reasoning_context["active_targets"] = targets
            reasoning_str = f"\nReasoning Context:\n{json.dumps(reasoning_context, indent=2)}"
        
        # Format follow-up questions if available
        follow_up_str = ""
        if follow_up_questions:
            follow_up_str = f"\nFollow-up Questions:\n" + "\n".join(f"- {q}" for q in follow_up_questions)
        elif targets:  # Generate comprehensive follow-up questions for targets
            follow_up_str = "\nFollow-up Questions:\n"
            for target in targets:
                if re.match(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', target):  # IP or CIDR
                    # Network scanning options
                    follow_up_str += f"- Would you like me to scan {target} for open ports?\n"
                    follow_up_str += f"- Should I check if {target} is responding to ping?\n"
                    follow_up_str += f"- Would you like to see what services are running on {target}?\n"
                    follow_up_str += f"- Should I perform a vulnerability scan on {target}?\n"
                    follow_up_str += f"- Would you like me to check for common web vulnerabilities on {target}?\n"
                    follow_up_str += f"- I can scan {target} for specific ports or services you're interested in.\n"
                    follow_up_str += f"- Would you like to see what operating system {target} is running?\n"
                    follow_up_str += f"- Should I check if {target} has any exposed databases?\n"
                    follow_up_str += f"- Would you like me to scan {target} for common misconfigurations?\n"
                    follow_up_str += f"- I can check if {target} has any exposed admin interfaces.\n"
                    
                    # Network mapping options
                    follow_up_str += f"- Would you like me to map the network topology around {target}?\n"
                    follow_up_str += f"- I can check what other hosts are in the same network as {target}.\n"
                    follow_up_str += f"- Should I identify the network services and their versions on {target}?\n"
                    follow_up_str += f"- Would you like to see the network path to {target}?\n"
                    follow_up_str += f"- I can check for any network security devices protecting {target}.\n"
                    
                    # Security assessment options
                    follow_up_str += f"- Would you like me to check {target} for common security misconfigurations?\n"
                    follow_up_str += f"- I can scan {target} for known vulnerabilities in running services.\n"
                    follow_up_str += f"- Should I check if {target} is running any outdated or vulnerable software?\n"
                    follow_up_str += f"- Would you like me to analyze {target}'s security posture?\n"
                    follow_up_str += f"- I can check if {target} has any exposed sensitive information.\n"
                    
                elif re.match(r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}', target):  # Hostname
                    # DNS and resolution options
                    follow_up_str += f"- Would you like me to resolve the DNS for {target}?\n"
                    follow_up_str += f"- I can check all DNS records associated with {target}.\n"
                    follow_up_str += f"- Should I verify the SSL/TLS configuration of {target}?\n"
                    follow_up_str += f"- Would you like to see the IP addresses associated with {target}?\n"
                    follow_up_str += f"- I can check if {target} has any subdomains.\n"
                    
                    # Web-specific options
                    follow_up_str += f"- Would you like me to scan {target} for web vulnerabilities?\n"
                    follow_up_str += f"- I can check if {target} has any exposed admin panels.\n"
                    follow_up_str += f"- Should I analyze the security headers of {target}?\n"
                    follow_up_str += f"- Would you like me to check for common web misconfigurations on {target}?\n"
                    follow_up_str += f"- I can scan {target} for exposed sensitive files.\n"
                    
                    # General security options
                    follow_up_str += f"- Would you like me to check if {target} is responding to ping?\n"
                    follow_up_str += f"- I can scan {target} for open ports and services.\n"
                    follow_up_str += f"- Should I check if {target} has any known vulnerabilities?\n"
                    follow_up_str += f"- Would you like to see what services are running on {target}?\n"
                    follow_up_str += f"- I can analyze the security posture of {target}.\n"
        
        # Add active targets if any
        targets_str = ""
        if targets:
            targets_str = f"\nActive Targets:\n" + "\n".join(f"- {t}" for t in targets)
        
        # Create enhanced prompt with all context
        prompt = f"{base_prompt}\n\n"
        if context_str:
            prompt += f"Recent Conversation:\n{context_str}\n"
        if targets_str:
            prompt += f"{targets_str}\n"
        if tool_context_str:
            prompt += f"{tool_context_str}\n"
        if reasoning_str:
            prompt += f"{reasoning_str}\n"
        if follow_up_str:
            prompt += f"{follow_up_str}\n"
        
        prompt += f"\nTask: {current_task}\nResponse:"
        
        # Cache the result
        self.prompt_cache[cache_key] = prompt
        
        # Limit cache size
        if len(self.prompt_cache) > 50:
            keys_to_remove = list(self.prompt_cache.keys())[:-25]
            for key in keys_to_remove:
                self.prompt_cache.pop(key, None)
                
        return prompt
        
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