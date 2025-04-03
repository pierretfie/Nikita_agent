#!/usr/bin/env python3

import os
import subprocess
import shlex
import psutil
from llama_cpp import Llama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
from datetime import datetime
from pathlib import Path
import re
import time
import random
import socket
import getpass
import sys
import readline # Add readline for command history and editing
import warnings
import contextlib

console = Console()

# ===============================
# === CONFIG ====================
# ===============================

# Base directory for Nikita
NIKITA_BASE_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model")

# Config paths
MODEL_PATH = os.path.join(NIKITA_BASE_DIR, "mistral.gguf")
OUTPUT_DIR = os.path.join(NIKITA_BASE_DIR, "outputs")
HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "history.json")
CHAT_HISTORY_FILE = Path(os.path.join(NIKITA_BASE_DIR, "nikita_history.json"))
COMMAND_HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "command_history")

# Create necessary directories
os.makedirs(NIKITA_BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
MAX_TOKENS = 256  # Reduced from 512
TEMPERATURE = 0.5  # Reduced from 0.7 for more focused responses

# ===============================
# === COMMAND HISTORY ===========
# ===============================

def setup_keyboard_shortcuts():
    """Configure advanced keyboard shortcuts for command editing"""
    if sys.platform != 'win32':
        try:
            # Command history navigation
            readline.parse_and_bind(r'"\e[A": previous-history')    # Up arrow
            readline.parse_and_bind(r'"\e[B": next-history')        # Down arrow
            
            # Cursor movement 
            readline.parse_and_bind(r'"\e[C": forward-char')        # Right arrow
            readline.parse_and_bind(r'"\e[D": backward-char')       # Left arrow
            
            # Word navigation
            readline.parse_and_bind(r'"\e[1;5C": forward-word')     # Ctrl+Right
            readline.parse_and_bind(r'"\e[1;5D": backward-word')    # Ctrl+Left
            
            # Line navigation
            readline.parse_and_bind(r'"\C-a": beginning-of-line')   # Ctrl+A
            readline.parse_and_bind(r'"\C-e": end-of-line')         # Ctrl+E
            
            # History search (type part of command then use these)
            readline.parse_and_bind(r'"\C-r": reverse-search-history') # Ctrl+R
            
            # Advanced editing
            readline.parse_and_bind(r'"\C-k": kill-line')           # Ctrl+K (delete to end)
            readline.parse_and_bind(r'"\C-u": unix-line-discard')   # Ctrl+U (delete to start)
            
            console.print("🔤 [green]Keyboard shortcuts enabled[/green]")
            return True
        except Exception as e:
            console.print(f"[yellow]Keyboard shortcuts setup failed: {e}[/yellow]")
    
    return False

def setup_command_history():
    """Configure readline for command history and editing"""
    try:
        # Set up readline history file
        if not os.path.exists(COMMAND_HISTORY_FILE):
            with open(COMMAND_HISTORY_FILE, 'w') as f:
                pass
        
        # Configure readline
        try:
            readline.read_history_file(COMMAND_HISTORY_FILE)
        except Exception as e:
            console.print(f"[yellow]Could not read command history: {e}[/yellow]")
        
        # Set history length to 1000 entries
        readline.set_history_length(1000)
        
        # Enable auto-complete with tab
        readline.parse_and_bind("tab: complete")
        
        # Load available commands for auto-completion
        commands, _ = discover_system_commands()
        completer = CommandCompleter(list(commands.keys()))
        readline.set_completer(completer.complete)
        
        # Setup advanced keyboard shortcuts
        setup_keyboard_shortcuts()
        
        console.print("🔄 [cyan]Command history and editing enabled[/cyan]")
        return True
    except Exception as e:
        console.print(f"[yellow]Command history setup failed: {e}[/yellow]")
        console.print("[yellow]Continuing without command history support[/yellow]")
        return False

def save_command_history():
    """Save command history to file"""
    try:
        readline.write_history_file(COMMAND_HISTORY_FILE)
    except Exception as e:
        console.print(f"[yellow]Could not save command history: {e}[/yellow]")

def get_input_with_history():
    """Get user input with readline history support and better error handling"""
    try:
        user_input = input().strip()
        
        # Save non-empty commands to history
        if user_input and not user_input.isspace():
            # Add to history only if it's different from the last command
            hist_len = readline.get_current_history_length()
            if hist_len == 0 or user_input != readline.get_history_item(hist_len):
                save_command_history()
        
        return user_input
    except EOFError:
        # Handle Ctrl+D gracefully
        console.print("\n[yellow]EOF detected. Use 'exit' to quit.[/yellow]")
        return ""
    except KeyboardInterrupt:
        # Should be caught at a higher level, but just in case
        console.print("\n[yellow]Command interrupted[/yellow]")
        return ""
    except Exception as e:
        console.print(f"\n[yellow]Error reading input: {e}[/yellow]")
        return ""

# ===============================
# === FINE TUNING IMPLEMENTATION
# ===============================

class FinetuningKnowledge:
    def __init__(self):
        self.knowledge_base = {}
        self.categories = set()
        self.tools = set()
        self.load_knowledge()

    def load_knowledge(self):
        try:
            with open(os.path.join(NIKITA_BASE_DIR, "fine_tuning.json"), "r") as f:
                data = json.load(f)
                for entry in data:
                    # Create category index
                    category = entry["category"]
                    self.categories.add(category)
                    if category not in self.knowledge_base:
                        self.knowledge_base[category] = []
                    self.knowledge_base[category].append(entry)
                    
                    # Track tools
                    self.tools.add(entry["tool_used"])
                    
                console.print(f"🧠 [green]Loaded {len(data)} fine-tuning entries across {len(self.categories)} categories[/green]")
        except Exception as e:
            console.print(f"[red]Error loading fine-tuning data: {e}[/red]")
            self.knowledge_base = {}

    def get_command_for_task(self, task):
        """Get the most relevant command for a given task"""
        task_lower = task.lower()
        best_match = None
        best_score = 0
        
        # Search through all entries
        for category, entries in self.knowledge_base.items():
            for entry in entries:
                score = 0
                
                # Check instruction match
                instruction_words = set(entry["instruction"].lower().split())
                task_words = set(task_lower.split())
                common_words = instruction_words & task_words
                score += len(common_words) * 2
                
                # Check if key phrases match
                if "how to" in task_lower and "how to" in entry["instruction"].lower():
                    score += 2
                if entry["tool_used"].lower() in task_lower:
                    score += 3
                
                # Check category relevance
                if any(word in task_lower for word in category.lower().split()):
                    score += 2
                
                # Update best match if this score is higher
                if score > best_score:
                    best_score = score
                    best_match = entry
        
        # Return the best match if it has a minimum score
        return best_match if best_score >= 2 else None

    def suggest_next_steps(self, current_task, output):
        """Suggest next steps based on current task and output"""
        suggestions = []
        current_category = None
        
        # Find the current category
        for cat, entries in self.knowledge_base.items():
            for entry in entries:
                if entry["instruction"].lower() in current_task.lower():
                    current_category = cat
                    break
            if current_category:
                break
                    
        return suggestions

# Initialize fine-tuning knowledge
fine_tuning = FinetuningKnowledge()

# ===============================
# === REASONING ENGINE =========
# ===============================

class ReasoningEngine:
    def __init__(self):
        self.reasoning_template = """
Thought Process:
1. UNDERSTAND: {task}
   - Goal: {goal}
   - Context: {context}
   - Constraints: {constraints}

2. PLAN:
   - Required steps: {steps}
   - Dependencies: {dependencies}
   - Order: {order}

3. TOOLS:
   - Primary tool: {primary_tool}
   - Alternative tools: {alternative_tools}
   - Parameters needed: {parameters}

4. SAFETY:
   - Risks: {risks}
   - Precautions: {precautions}
   - Fallback plan: {fallback}

5. EXECUTION:
   Command: {command}
   Explanation: {explanation}

6. ANALYSIS:
   - Expected output: {expected_output}
   - Success indicators: {success_indicators}
   - Next steps: {next_steps}
"""

    def analyze_task(self, task, fine_tuned_match=None):
        """Generate structured reasoning about a task"""
        
        if fine_tuned_match:
            # Use fine-tuned knowledge for better reasoning
            category = fine_tuned_match["category"]
            difficulty = fine_tuned_match["difficulty"]
            attack_technique = fine_tuned_match["attack_technique"]
            command = fine_tuned_match["output"].split("\n")[0].replace("Command: ", "")
            analysis = fine_tuned_match["output"].split("\n")[1].replace("Analysis: ", "")
            
            # Extract steps based on category
            steps = self._get_category_steps(category)
            
            # Determine tools needed
            primary_tool = fine_tuned_match["tool_used"]
            alternative_tools = self._get_additional_tools(category)
            
            # Get parameters from command
            parameters = self._extract_parameters(command)
            
            # Determine goal from category
            goals = {
                "Recon": "Gather information about target systems or networks",
                "Web": "Identify vulnerabilities or gather information about web applications",
                "Wireless": "Analyze or test wireless network security",
                "Password": "Test or bypass authentication mechanisms"
            }
            goal = goals.get(category, "Accomplish the requested security task effectively")
            
            # Safety considerations
            risks = self._get_risks(category)
            precautions = self._get_precautions(category)
            
        else:
            # Enhanced reasoning for unknown tasks
            task_lower = task.lower()
            
            # Categorize the task
            category = self._guess_category(task)
            
            # Determine goal based on task keywords
            goal = self._determine_goal(task)
            
            # Analyze context based on task
            context = self._analyze_context(task)
            
            # Identify constraints
            constraints = self._identify_constraints(task)
            
            # Get steps based on category
            steps = self._get_category_steps(category)
            
            # Determine dependencies
            dependencies = self._determine_dependencies(category)
            
            # Determine execution order
            order = "Sequential execution of the steps outlined above"
            
            # Determine primary tool based on task
            primary_tool = self._guess_primary_tool(task)
            
            # Get alternative tools
            alternative_tools = self._get_additional_tools(category)
            
            # Determine parameters
            parameters = ["To be determined based on specific command"]
            
            # Determine risks and precautions
            risks = self._get_risks(category)
            precautions = self._get_precautions(category)
            fallback = "Use alternative command or tool if primary approach fails"
            
            # Determine command
            command = "To be determined based on further analysis"
            explanation = "Will select appropriate command based on task requirements"
            
            # Determine expected outputs
            expected_output = self._determine_expected_output(category, primary_tool)
            success_indicators = self._determine_success_indicators(category)
            next_steps = self._determine_next_steps(category)
        
        # Format the reasoning
        reasoning = self.reasoning_template.format(
            task=task,
            goal=goal if 'goal' in locals() else "Accomplish the requested task effectively",
            context=context if 'context' in locals() else "Current system state and user request",
            constraints=constraints if 'constraints' in locals() else "Operating within system limitations",
            steps="\n   - ".join(steps),
            dependencies=dependencies if 'dependencies' in locals() else "Required system tools and access",
            order=order if 'order' in locals() else "Sequential execution of steps",
            primary_tool=primary_tool,
            alternative_tools=", ".join(alternative_tools) if alternative_tools else "None required",
            parameters=", ".join(parameters) if parameters else "To be determined",
            risks=risks if 'risks' in locals() else "Minimal risks for basic operations",
            precautions=precautions if 'precautions' in locals() else "Use appropriate validation",
            fallback=fallback if 'fallback' in locals() else "Use alternative approaches if needed",
            command=command,
            explanation=explanation if 'explanation' in locals() else "Requires more context or clarification",
            expected_output=expected_output if 'expected_output' in locals() else "Command-specific output",
            success_indicators=success_indicators if 'success_indicators' in locals() else "Successful command execution",
            next_steps=next_steps if 'next_steps' in locals() else "Analyze output and determine follow-up actions"
        )
        
        return reasoning

    def _determine_goal(self, task):
        """Determine the goal based on task description"""
        task_lower = task.lower()
        
        if re.search(r'(scan|enumerate|discover|find|list)', task_lower):
            return "Discover and enumerate network resources or information"
        elif re.search(r'(check|verify|confirm|test)', task_lower):
            return "Verify system status or confirm operational state"
        elif re.search(r'(analyze|examine|study|investigate)', task_lower):
            return "Analyze system data or investigate specific conditions"
        elif re.search(r'(show|display|get|what)', task_lower):
            return "Retrieve and display specific system information"
        elif re.search(r'(ip|address|network)', task_lower):
            return "Identify network configuration or addressing information"
        else:
            return "Execute requested operation and provide relevant information"

    def _analyze_context(self, task):
        """Analyze context of the task"""
        # Get active targets from engagement memory
        targets = engagement_memory.get("targets", [])
        return f"User request in context of system state and {'active targets: ' + ', '.join(targets) if targets else 'no active targets'}"

    def _identify_constraints(self, task):
        """Identify constraints for the task"""
        constraints = ["Current user permissions", "System resource availability"]
        
        # Add task-specific constraints
        if "remote" in task.lower():
            constraints.append("Network connectivity to remote systems")
        if "scan" in task.lower():
            constraints.append("Scan performance and target responsiveness")
            
        return ", ".join(constraints)

    def _determine_dependencies(self, category):
        """Determine dependencies based on category"""
        dependencies = {
            "Recon": "Network access, required scan permissions",
            "Web": "HTTP client libraries, web server access",
            "Wireless": "Wireless interface in monitor mode",
            "Password": "Dictionary files, processing capability",
            "General": "Basic system tools and utilities"
        }
        return dependencies.get(category, dependencies["General"])

    def _get_risks(self, category):
        """Get risks associated with the category"""
        risks = {
            "Recon": "Target detection of scanning activity, false positives",
            "Web": "Unexpected service disruption, detected intrusion attempts",
            "Wireless": "Regulatory compliance issues, detection by monitors",
            "Password": "Account lockouts, audit log generation",
            "General": "Command timeout, unexpected output format"
        }
        return risks.get(category, risks["General"])

    def _get_precautions(self, category):
        """Get precautions for the category"""
        precautions = {
            "Recon": "Use proper scan timing, verify scope authorization",
            "Web": "Validate input, check for WAF/security controls",
            "Wireless": "Ensure regulatory compliance, verify isolated testing",
            "Password": "Monitor for lockout policies, use incremental approach",
            "General": "Validate commands before execution, review output carefully"
        }
        return precautions.get(category, precautions["General"])

    def _determine_expected_output(self, category, tool):
        """Determine expected output based on category and tool"""
        if category == "Recon" and tool == "nmap":
            return "Port status, service versions, host information"
        elif category == "Web":
            return "HTTP responses, discovered endpoints, potential vulnerabilities"
        elif "ping" in tool:
            return "Response time, packet statistics, host availability"
        elif "ip" in tool:
            return "Network interface information, addressing details"
        else:
            return "Command-specific output relevant to the task"

    def _determine_success_indicators(self, category):
        """Determine success indicators based on category"""
        indicators = {
            "Recon": "Discovered hosts, identified services, mapped network",
            "Web": "Accessible endpoints, identified technologies, vulnerability confirmation",
            "Wireless": "Captured packets, identified networks, successful authentication",
            "Password": "Successful authentication, cracked hashes, identified weaknesses",
            "General": "Clean command execution, relevant output, actionable information"
        }
        return indicators.get(category, indicators["General"])

    def _determine_next_steps(self, category):
        """Determine next steps based on category"""
        next_steps = {
            "Recon": "Target specific services, perform deeper analysis on open ports",
            "Web": "Follow up on identified endpoints, test vulnerabilities, gather more information",
            "Wireless": "Analyze captured data, attempt authentication if appropriate",
            "Password": "Use identified credentials, attempt privilege escalation",
            "General": "Analyze output, refine approach based on results"
        }
        return next_steps.get(category, next_steps["General"])

    def _get_category_steps(self, category):
        """Get standard steps for a given category"""
        steps_by_category = {
            "Recon": [
                "Verify target scope and permissions",
                "Perform initial host discovery",
                "Identify open ports and services",
                "Gather service versions and details"
            ],
            "Web": [
                "Check target website availability",
                "Identify web technologies used",
                "Scan for common vulnerabilities",
                "Test discovered endpoints"
            ],
            "Wireless": [
                "Put interface in monitor mode",
                "Scan for target networks",
                "Capture required handshakes/data",
                "Analyze captured data"
            ],
            "Password": [
                "Identify authentication mechanism",
                "Prepare wordlist/attack method",
                "Execute brute force attempt",
                "Monitor for successful attempts"
            ]
        }
        return steps_by_category.get(category, ["Analyze requirements", "Plan approach", "Execute safely", "Verify results"])

    def _guess_category(self, task):
        """Guess the category based on task keywords"""
        task_lower = task.lower()
        categories = {
            "Recon": ["scan", "enumerate", "discover", "find", "list"],
            "Web": ["http", "website", "url", "web", "port 80", "port 443"],
            "Wireless": ["wifi", "wireless", "wpa", "handshake", "deauth"],
            "Password": ["crack", "brute", "password", "hash", "login"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in task_lower for keyword in keywords):
                return category
        return "General"

    def _guess_primary_tool(self, task):
        """Guess the primary tool needed based on task description"""
        task_lower = task.lower()
        tool_keywords = {
            "nmap": ["scan", "port", "service", "version"],
            "hashcat": ["crack", "hash", "password"],
            "aircrack-ng": ["wifi", "wpa", "wireless", "handshake"],
            "gobuster": ["directory", "web", "brute", "website"],
            "hydra": ["brute", "login", "password", "ssh", "ftp"]
        }
        
        for tool, keywords in tool_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return tool
        return "To be determined"

    def _get_additional_tools(self, category):
        """Get commonly paired tools for a category"""
        tool_pairs = {
            "Recon": ["dig", "whois", "traceroute"],
            "Web": ["curl", "burpsuite", "nikto"],
            "Wireless": ["airmon-ng", "airodump-ng", "wireshark"],
            "Password": ["john", "hashcat", "crunch"]
        }
        return tool_pairs.get(category, [])

    def _extract_parameters(self, command):
        """Extract parameters from a command string"""
        if not command or command == "To be determined based on further analysis":
            return []
            
        params = []
        parts = shlex.split(command)
        for part in parts[1:]:  # Skip the command name
            if part.startswith("-"):
                params.append(part)
        return params

# Initialize reasoning engine
reasoning_engine = ReasoningEngine()

# ===============================
# === CONTEXT OPTIMIZER ========
# ===============================

class ContextOptimizer:
    def __init__(self, max_tokens=2048, reserve_tokens=512):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.cache = {}  # Add cache for frequently used contexts
        
    def optimize_context(self, chat_memory, current_task, targets=None):
        """Optimize context window with minimal processing"""
        # Check cache first
        cache_key = f"{current_task}_{len(chat_memory)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Only process last message
        if not chat_memory:
            return []
        
        last_msg = chat_memory[-1]
        if not isinstance(last_msg, dict) or not last_msg.get("content"):
            return []
        
        content = last_msg["content"]
        
        # Simple relevance check
        if any(target in content for target in (targets or [])) or \
           any(word in content.lower() for word in current_task.lower().split()):
            relevant_context = [content]
        else:
            relevant_context = []
        
        # Cache the result
        self.cache[cache_key] = relevant_context
        return relevant_context
    
    def get_optimized_prompt(self, chat_memory, current_task, base_prompt):
        """Get optimized prompt with minimal context for faster processing"""
        # Only use last message for context
        optimized_context = self.optimize_context(chat_memory[-1:], current_task)
        context_str = "\n".join(optimized_context[-1:])  # Only use last message
        
        # Minimal system state information
        ram, swap, cpu_count, ram_gb = get_system_info()
        system_state = f"RAM: {ram_gb:.1f}GB ({ram.percent}%)"
        
        # Ultra-simplified reasoning guide
        reasoning_guide = """
Quick Response:
1. Direct answers for simple queries
2. Tool suggestions for commands
3. Key findings for analysis
4. Concise, actionable responses
"""
        
        # Combine everything into a minimal prompt
        enhanced_prompt = f"{base_prompt}\n{system_state}\n{reasoning_guide}\n\nContext: {context_str}\nTask: {current_task}\nResponse:"
        
        return enhanced_prompt

# ===============================
# === MEMORY ENGINE =============
# ===============================

memory = {
    "targets": [],
    "credentials": [],
    "loot": [],
    "network_maps": [],
    "system_info": {}
}

chat_memory = []

# === Resource Management ===
def get_system_info():
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_count = os.cpu_count() or 4
    ram_gb = ram.total / (1024 * 1024 * 1024)
    return ram, swap, cpu_count, ram_gb

def get_dynamic_params():
    ram, swap, cpu_count, ram_gb = get_system_info()
    
    # More aggressive RAM allocation for speed
    available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    used_ram = ram.used / (1024 * 1024 * 1024)
    
    # RAM allocation tiers with ultra-optimized settings
    if ram_gb >= 32:  # High-end systems
        ram_target_utilization = 0.9  # Increased from 0.8
        context_limit = 32768  # Fixed context limit
        n_batch = 258  # Fixed batch size
    elif ram_gb >= 16:  # Mid-range systems
        ram_target_utilization = 0.8  # Increased from 0.7
        context_limit = 4096  # Fixed context limit
        n_batch = 258  # Fixed batch size
    elif ram_gb >= 8:  # Common systems
        ram_target_utilization = 0.7  # Increased from 0.6
        context_limit = 2048  # Fixed context limit
        n_batch = 258  # Fixed batch size
    else:  # Low-memory systems
        ram_target_utilization = 0.7  # Increased from 0.6
        context_limit = 2048  # Fixed context limit
        n_batch = 258  # Fixed batch size
    
    # Adjust based on actual usage
    if used_ram < ram_gb * 0.3:  # If usage is below 30%
        ram_target_utilization = min(ram_target_utilization + 0.1, 0.95)  # Increased max to 95%
    
    memory_limit = int(min(ram_gb * ram_target_utilization, available_ram * 0.95))
    
    # Ultra-optimized configuration for speed
    base_config = {
        'n_threads': max(1, int(cpu_count * 0.98)),  # Increased to 98% for maximum CPU utilization
        'n_batch': n_batch,
        'max_tokens': 1024,  # Increased from 256 for longer responses
        'context_limit': context_limit,
        'memory_limit': memory_limit,
        'temperature': 0.3,  # Reduced for more focused and accurate responses
        'top_k': 20,  # Reduced to focus on most relevant tokens
        'top_p': 0.85,  # Reduced for more deterministic sampling
        'repeat_penalty': 1.1,  # Slightly increased to reduce repetition while maintaining coherence
        'n_gpu_layers': 0,
        'use_mmap': True,
        'f16_kv': True,
        'rope_scaling': {"type": "linear", "factor": 0.1}  # Further reduced from 0.25
    }
    
    console.print(f"⚙️ [cyan]RAM Tier: {ram_gb:.1f}GB system | Using {memory_limit}GB | Target: {ram_target_utilization*100}%[/cyan]")
    console.print(f"📊 [cyan]Current usage: {used_ram:.1f}GB ({ram.percent}%) | Context: {context_limit} tokens | Batch: {n_batch}[/cyan]")
    
    return base_config

def optimize_memory_resources():
    """Optimize memory usage with aggressive settings"""
    try:
        # Run garbage collection
        import gc
        gc.collect()
        
        # Get available RAM
        available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        
        # Configure environment for maximum memory usage
        if available_ram > 1.0:  # Reduced threshold from 2.0
            # Enable memory mapping and mlock
            os.environ["GGML_MLOCK"] = "1"
            os.environ["GGML_USE_MMAP"] = "1"
            
            # Increase memory limits
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                new_limit = min(int(available_ram * 0.98 * 1024 * 1024 * 1024), hard)  # Increased to 98%
                resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
            except:
                pass
                
            console.print("💫 [green]Aggressive memory optimization activated[/green]")
        else:
            console.print("⚠️ [yellow]Limited RAM available - optimizing for balanced usage[/yellow]")
        
        return True
    except Exception as e:
        console.print(f"[yellow]Memory optimization failed: {e} - continuing with defaults[/yellow]")
        return False

def optimize_cpu_usage():
    """Optimize CPU usage with maximum core allocation"""
    process = psutil.Process()
    cpu_count = os.cpu_count()
    
    if cpu_count > 1:
        # Calculate target cores based on system load
        current_load = psutil.getloadavg()[0] / cpu_count
        if current_load < 0.9:  # Increased threshold from 0.8
            # Use 95% of available cores when load is low
            target_cores = max(2, int(cpu_count * 0.95))
        else:
            # Use 80% of available cores when load is high
            target_cores = max(1, int(cpu_count * 0.8))
        
        # Ensure we don't use all cores
        target_cores = min(target_cores, cpu_count - 1)
        
        # Create affinity list with correct number of cores
        affinity = list(range(target_cores))
        try:
            process.cpu_affinity(affinity)
            console.print(f"⚡ [cyan]CPU affinity set to use {target_cores} cores based on current load ({current_load:.2f})[/cyan]")
        except Exception as e:
            console.print(f"⚠️ [yellow]Could not set CPU affinity: {e}[/yellow]")
    else:
        console.print("⚠️ [yellow]Single core system detected - CPU affinity not applicable[/yellow]")

# === Model Setup ===
console.print("🔧 [cyan]Initializing Nikita...[/cyan]")

# Optimize system resources
optimize_memory_resources()

# Look for hardware acceleration
try:
    from llama_cpp import llama_cpp
    has_cuda = hasattr(llama_cpp, "LLAMA_BACKEND_CUDA") and llama_cpp.LLAMA_BACKEND_CUDA
    has_metal = hasattr(llama_cpp, "LLAMA_BACKEND_METAL") and llama_cpp.LLAMA_BACKEND_METAL
    
    if has_cuda:
        console.print("✅ [bold green]CUDA support detected[/bold green]")
    elif has_metal:
        console.print("✅ [bold green]Metal support detected (Apple GPU)[/bold green]")
    else:
        console.print("ℹ️ [yellow]Hardware acceleration not detected - using CPU only[/yellow]")
except:
    console.print("ℹ️ [yellow]Could not check for hardware acceleration - using CPU only[/yellow]")
    has_cuda = False
    has_metal = False

# Get system parameters
system_params = get_dynamic_params()

# Ensure mlock is enabled for RAM optimization
try:
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
except:
    pass

# Initialize the model with scalable parameters
import warnings

# Create a context manager to redirect stderr
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Initialize model with stderr suppression
with suppress_stderr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=system_params['context_limit'],
            n_threads=system_params['n_threads'],
            n_batch=system_params['n_batch'],
            max_tokens=system_params['max_tokens'],
            use_mlock=True,
            use_mmap=True,
            low_vram=False,
            verbose=False,
            f16_kv=True,
            seed=42,
            embedding=False,
            rope_scaling={"type": "linear", "factor": 0.5},
            n_gpu_layers=system_params['n_gpu_layers'],
            vocab_only=False,
            tensor_split=None
        )

# Apply more precise CPU optimizations
optimize_cpu_usage()

# Prewarm the model with RAM optimization
console.print("🔥 [cyan]Prewarming model with RAM optimization...[/cyan]")
llm("System check complete. Nikita is ready.", max_tokens=20)

# === Prompt & Memory ===
PROMPT_TEMPLATE = """You are Nikita 🐺, an Offline AI Security Assistant on Kali Linux.
Your primary goal is to understand and execute security tasks effectively while maintaining operational security.

SECURITY PRINCIPLES:
1. Reconnaissance First
   - Start passive before active
   - Verify target scope
   - Document findings
   - Assess potential risks

2. Command Selection
   - Choose least intrusive option first
   - Validate target parameters
   - Consider OPSEC implications
   - Add safety flags when needed

3. Output Analysis
   - Look for security indicators
   - Identify potential vulnerabilities
   - Note unusual patterns
   - Suggest follow-up actions

4. Tool Categories:
   A. Reconnaissance
      - Network mapping (nmap, ping)
      - Service enumeration
      - Version detection
      - Subdomain enumeration (amass, subfinder)
      
   B. Vulnerability Assessment
      - Port scanning
      - Service fingerprinting
      - Vulnerability scanning
      - Web application testing (burpsuite, nikto)
      
   C. Web Security
      - Directory enumeration
      - Parameter testing
      - CMS identification
      - Proxy interception (burpsuite)
      
   D. System Security
      - Process monitoring
      - File integrity
      - Log analysis
      - Password cracking (john, hashcat)

5. Security Tools Knowledge:
   - Burp Suite: Web application security testing tool for intercepting, analyzing, and modifying HTTP traffic
   - Amass: Subdomain enumeration and attack surface mapping tool for discovering assets
   - Nmap: Network exploration and security scanning tool
   - Nikto: Web server scanner for identifying vulnerabilities
   - Gobuster: Directory/File, DNS and VHost busting tool
   - Hydra: Password cracking tool supporting multiple protocols
   - John the Ripper: Password cracking tool
   - Hashcat: Advanced password recovery tool

RESPONSE GUIDELINES:
1. For Security Scans:
   - Start with stealth options
   - Gradually increase intensity
   - Save all outputs
   - Document findings

2. For System Analysis:
   - Check permissions first
   - Monitor resource usage
   - Look for anomalies
   - Maintain audit trail

3. For Network Operations:
   - Verify scope
   - Use appropriate timing
   - Monitor for alerts
   - Document responses

4. For Error Handling:
   - Provide clear error context
   - Suggest safe alternatives
   - Explain potential risks
   - Offer recovery 

5. For general knowledge and facts:
   - Provide accurate and up-to-date information
   - Use credible sources
   - Verify information
   - Offer context when relevant
   - Stick to the facts if available
   

Remember:
- Always verify target scope
- Start with passive techniques
- Document all findings
- Consider security implications
- Suggest follow-up actions
- Offer as dangerous offemsive exploit suggestions as possible 
"""

# Update system commands list
system_commands = [
    # System Info
    "hostname", "uname", "whoami", "id", "ps", "top", "htop",
    "df", "free", "uptime", "lsof", "netstat", "ss",
    
    # Network
    "ping", "ip", "ifconfig", "iwconfig", "arp",
    "route", "traceroute", "mtr", "curl", "wget",
    "dig", "nslookup", "whois",
    
    # Security
    "nmap", "gobuster", "amass", "smbclient",
    "sqlmap", "nikto", "dirb", "hydra", "john", "hashcat",
    
    # File and Process
    "ls", "cat", "grep", "find", "locate",
    "ps", "kill", "pkill", "service", "systemctl"
]

# === Command Discovery and Validation ===
def discover_system_commands():
    """Dynamically discover available system commands"""
    commands = {}
    common_paths = ['/usr/bin', '/bin', '/usr/sbin', '/sbin', '/usr/local/bin']
    
    # Common command categories
    categories = {
        'system': ['ps', 'top', 'htop', 'free', 'df', 'du', 'uname', 'uptime', 'who', 'whoami', 'w'],
        'network': ['ping', 'netstat', 'ss', 'ip', 'ifconfig', 'route', 'arp', 'traceroute', 'dig', 'nslookup', 'whois', 'curl', 'wget'],
        'security': ['nmap', 'sqlmap', 'gobuster', 'nikto', 'dirb', 'hydra', 'john', 'hashcat', 'aircrack-ng'],
        'file': ['ls', 'cat', 'grep', 'find', 'locate', 'which', 'file', 'dd', 'tar', 'zip', 'unzip'],
        'process': ['kill', 'pkill', 'pgrep', 'nice', 'renice', 'nohup', 'watch', 'screen', 'tmux'],
        'package': ['apt', 'apt-get', 'dpkg', 'snap', 'flatpak'],
        'service': ['service', 'systemctl', 'journalctl']
    }
    
    for path in common_paths:
        if os.path.exists(path):
            for cmd in os.listdir(path):
                cmd_path = os.path.join(path, cmd)
                if os.access(cmd_path, os.X_OK):
                    commands[cmd] = cmd_path
    
    # Categorize discovered commands
    categorized_commands = {cat: [] for cat in categories}
    for cmd in commands:
        for cat, cmd_list in categories.items():
            if cmd in cmd_list:
                categorized_commands[cat].append(cmd)
                break
    
    return commands, categorized_commands

# Command completer for readline
class CommandCompleter:
    def __init__(self, commands):
        self.commands = sorted(commands)
        self.matches = []
    
    def complete(self, text, state):
        if state == 0:
            # Create new matches list for this completion
            if text:
                self.matches = [cmd for cmd in self.commands if cmd.startswith(text)]
            else:
                self.matches = list(self.commands)
        
        # Return match or None if no more matches
        try:
            return self.matches[state]
        except IndexError:
            return None

def validate_command(cmd):
    """Validate and sanitize command input"""
    # Basic command validation
    if not cmd:
        return None, "Empty command"
    
    # Remove any quotes that might cause parsing issues
    cmd = cmd.strip('"\'')
    
    # Basic security checks
    if ';' in cmd or '&&' in cmd or '||' in cmd:
        return None, "Invalid command: contains unsafe characters"
    
    try:
        # Split command and arguments safely
        parts = shlex.split(cmd)
    except ValueError as e:
        # Handle malformed commands
        return None, f"Invalid command format: {str(e)}"
    
    if not parts:
        return None, "Empty command after parsing"
    
    base_cmd = parts[0]
    
    # Check if command exists
    if base_cmd not in system_commands:
        # Try to find the command in PATH
        try:
            cmd_path = subprocess.check_output(['which', base_cmd], text=True).strip()
            if cmd_path:
                return cmd, None
        except subprocess.CalledProcessError:
            return None, f"Command '{base_cmd}' not found"
    
    return cmd, None

# Initialize command discovery at startup
system_commands, categorized_commands = discover_system_commands()
console.print(f"🔍 [cyan]Discovered {len(system_commands)} system commands[/cyan]")

# Initialize context optimizer with dynamic parameters
context_optimizer = ContextOptimizer(
    max_tokens=system_params['context_limit'],
    reserve_tokens=system_params['max_tokens']
)
console.print("🧠 [cyan]Context optimizer initialized[/cyan]")

# === Memory System ===
engagement_memory = {
    "targets": [],
    "credentials": [],
    "loot": [],
    "network_maps": [],
    "attack_history": []
}

# === Chat Memory Configuration ===
system_params = get_dynamic_params()
MEMORY_LIMIT = system_params['memory_limit']

def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                # Keep only the last MEMORY_LIMIT messages
                return history[-MEMORY_LIMIT:]
        except Exception as e:
            console.print(f"[yellow]Could not load chat history: {e}[/yellow]")
    return []

def save_chat_history(messages):
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Could not save chat history: {e}[/yellow]")

def extract_targets(task):
    targets = re.findall(r'(?:\d{1,3}\.){3}\d{1,3}', task)
    for t in targets:
        if t not in engagement_memory["targets"]:
            engagement_memory["targets"].append(t)
            console.print(f"🎯 [bold blue]New target identified:[/bold blue] {t}")
    return targets

def suggest_attack_plan(task):
    task_lower = task.lower()
    if "recon" in task_lower:
        return "🔍 Recommended: nmap -sC -sV -oA recon_scan <target> | tee recon_scan.txt | grep open"
    if "priv esc" in task_lower:
        return "⚡ Recommended: wget https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh -O linpeas.sh && chmod +x linpeas.sh && ./linpeas.sh | tee linpeas.txt"
    if "pivot" in task_lower:
        return "🔄 Recommended: ssh -D 9050 user@target -f -C -q -N (SOCKS Proxy Pivot)"
    if "web" in task_lower:
        return "🌐 Recommended: ffuf -u http://<target>/FUZZ -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt"
    return None

def get_dynamic_prompt():
    """Get dynamic prompt with system info and current context"""
    ram, swap, cpu_count, ram_gb = get_system_info()
    
    return f"""You are Nikita 🐺, an Offline AI Security Assistant.
Current System State:
- RAM: {ram_gb:.1f}GB ({ram.percent}% used)
- CPU Cores: {cpu_count}
- Active Targets: {', '.join(engagement_memory['targets']) if engagement_memory['targets'] else 'None'}
- Mode: {'Low RAM' if swap.used > (512 * 1024 * 1024) else 'Normal'}

{PROMPT_TEMPLATE}
"""

class IntentAnalyzer:
    def __init__(self):
        # Define intent categories
        self.intent_categories = {
            "agent_identity": [
                r'who are you', r'what are you', r'tell me about yourself',
                r'your name', r'your purpose', r'what can you do',
                r'who are', r'what is your name', r'introduce yourself',
                r'who is nikita', r'what kind of (ai|assistant|agent)', r'tell me who you are',
                r'what\'s your role', r'describe yourself', r'your function',
                r'your capabilities', r'what are you capable of', r'who created you',
                r'your creator', r'your version', r'your abilities', r'identify yourself',
                r'what do you do', r'your job', r'are you an ai', r'are you (nikita|an assistant)'
            ],
            "agent_status": [
                r'how are you', r'are you ok', r'are you working',
                r'your status', r'are you functional', r'how are you doing',
                r'are you online', r'are you available', r'are you operational',
                r'(having|experiencing) (issues|problems|errors|trouble)', r'do you work',
                r'functioning (properly|correctly|ok|well)', r'system status',
                r'health status', r'diagnostics', r'self-check', r'you (running|operating) (well|properly)',
                r'status report', r'everything (ok|working|good)', r'ready',
                r'operational status', r'feeling'
            ],
            "system_info": [
                r'(ip|address)', r'network', r'hardware', r'system',
                r'memory', r'cpu', r'disk', r'storage', r'version',
                r'operating system', r'kernel', r'linux version', r'processor',
                r'ram usage', r'memory usage', r'disk space', r'connected (to|devices)',
                r'(show|list|display) (my|the|all|system) (resources|specs|hardware|configuration)',
                r'(what|which) (os|operating system)', r'(how much|available) (memory|disk|storage|space)',
                r'uptime', r'running processes', r'services', r'running (since|for)',
                r'loaded modules', r'network (config|configuration|interfaces|cards)',
                r'(ethernet|wifi) connection', r'mac address', r'hostname',
                r'(show|get|what is) my (public|external|private|local) ip'
            ],
            "security_how_to": [
                r'how to (hack|exploit|break|attack|crack)',
                r'how to (secure|protect|defend)',
                r'how to scan', r'how to test',
                r'steps to (hack|exploit|secure|protect|scan|test|analyze)',
                r'tutorial (for|on)', r'guide (for|on)',
                r'explain how (to|I can|one would)',
                r'(teach|show) me how to', r'process (of|for)',
                r'method (of|for)', r'technique (for|to)',
                r'approach (to|for)', r'strategy (for|to)',
                r'walkthrough (of|for)', r'instructions (for|on)',
                r'best (way|method|approach) to', r'what\'s the (way|process) to',
                r'can you (explain|describe) how to'
            ],
            "security_scan": [
                r'scan .+ for', r'find vulnerabilities', r'check security',
                r'discover (hosts|services)', r'enumerate',
                r'(perform|run|execute|do) (a|an) scan',
                r'(search|look) for (open ports|vulnerabilities|security issues|weaknesses)',
                r'(detect|identify) (hosts|devices|machines|computers|servers)',
                r'network discovery', r'port scan',
                r'vulnerability assessment', r'security assessment',
                r'reconnaissance', r'footprinting', r'information gathering',
                r'(map|mapping) (the|a) network',
                r'(locate|find) (hosts|servers|devices) on (the|my|a) network',
                r'(test|probe|check) (security|vulnerability|exposure)',
                r'(audit|examine) (security|network|system)',
                r'(detect|find|identify) running services',
                r'scan (my|the) network', r'discover network devices'
            ],
            "wifi_operations": [
                r'wifi', r'wireless', r'wlan', r'access points',
                r'(scan|find|list) .+ networks', r'(show|display|get) (wifi|wireless) networks',
                r'(connected to|available) (wifi|wireless) networks',
                r'(monitor|observe|watch) (wifi|wireless) traffic',
                r'(join|connect to) (a|the) (wifi|wireless) network',
                r'(setup|configure) (wifi|wireless) (connection|adapter)',
                r'(change|switch|modify) (wifi|wireless) (settings|configuration)',
                r'(wifi|wireless) signal (strength|quality)',
                r'(nearby|surrounding|available) (wifi|access points|networks)',
                r'(turn|enable|disable) (on|off) (wifi|wireless)',
                r'(status|state) of (wifi|wireless) (connection|interface)'
            ],
            "find_command": [
                r'find my', r'show my', r'what is my', r'tell me my',
                r'display my', r'get my', r'what\'s my',
                r'locate my', r'where is my', r'discover my',
                r'check my', r'determine my', r'list my',
                r'show the', r'what is the', r'tell me the',
                r'find out (my|the|about)', r'give me (my|the|information about)',
                r'show me (my|the)', r'do I have',
                r'(can you|could you|please) (find|tell me|show me|give me) (my|the)',
                r'(query|lookup|search for) my'
            ],
            "process_management": [
                r'(running|active) processes', r'process (list|status)',
                r'(kill|terminate|end|stop) process', r'(start|launch|run) (process|program|application)',
                r'(monitor|watch) (process|cpu|memory) (usage|utilization)',
                r'(top|running) (applications|programs)', r'background (tasks|processes|jobs)',
                r'(restart|relaunch) (process|service|application|program)',
                r'(process|service) (control|management)',
                r'(what\'s|what is) (running|using) (resources|memory|cpu)',
                r'(resource|cpu|memory) (hog|intensive) (process|application)',
                r'task (manager|list)'
            ],
            "file_operations": [
                r'(find|locate|search for) (file|directory|folder)',
                r'(list|show|display) (files|directories|folders)',
                r'(read|view|open|cat) (file|content|text)',
                r'(create|make|new) (file|directory|folder)',
                r'(delete|remove|rm) (file|directory|folder)',
                r'(move|copy|rename) (file|directory|folder)',
                r'(change|modify|edit) (file|permissions|ownership)',
                r'(file|disk) (size|usage|space)',
                r'(compress|extract|archive) (file|directory|folder)',
                r'(search|grep|find) (text|string|pattern) in (file|files)'
            ],
            "network_devices": [
                r'(how many|what|which) (devices|hosts|machines|computers) (are|on|connected to) (my|the) network',
                r'(find|detect|discover|enumerate|count) (devices|hosts) on (my|the) network',
                r'(who|what) is on (my|the) network',
                r'network devices',
                r'(all|active) (devices|hosts) on (my|the) network',
                r'(show|list|display) (devices|hosts|computers) on (my|the) network',
                r'(scan|probe|search) (my|the) network for (devices|hosts)',
                r'devices connected to (my|the) network',
                r'count (devices|hosts|machines) on (my|the) network'
            ],
            "hash_identification": [
                r'(identify|detect|determine|check|what|which).*(hash|hashing).*(algorithm|type)',
                r'(identify|detect|determine|check).*(algorithm|type) of .*(hash|hashed)',
                r'what (algorithm|type) .*(hash|hashed)',
                r'hashcat.*identify',
                r'identify.*hash',
                r'hash.*identify'
            ]
        }
        
        # Command mappings for specific intents
        self.command_mappings = {
            "wifi_operations": {
                "scan": "nmcli dev wifi list",
                "networks": "nmcli dev wifi list",
                "access points": "nmcli dev wifi list",
                "monitor": "airmon-ng start wlan0",
                "connect": "nmcli dev wifi connect {SSID} password {PASSWORD}",
                "signal": "iwconfig",
                "status": "nmcli radio wifi",
                "interface": "iwconfig",
                "available": "nmcli dev wifi list --rescan yes"
            },
            "network_devices": {
                "default": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')",
                "how many": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')",
                "devices": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')",
                "hosts": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')",
                "scan": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')",
                "discover": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')",
                "count": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')",
                "list": "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')"
            },
            "system_info": {
                "ip": "ip -4 addr show",
                "address": "ip -4 addr show",
                "ethernet": "ip -4 addr show eth0",
                "wifi address": "ip -4 addr show wlan0",
                "network": "ip addr",
                "interfaces": "ip addr",
                "memory": "free -h",
                "cpu": "lscpu",
                "disk": "df -h",
                "hardware": "lshw -short",
                "system": "uname -a && uptime && free -h && df -h",
                "os": "cat /etc/os-release",
                "kernel": "uname -r",
                "uptime": "uptime",
                "version": "uname -a",
                "hostname": "hostname",
                "mac": "ip link show",
                "public ip": "curl -s ifconfig.me || wget -qO- ifconfig.me",
                "processor": "cat /proc/cpuinfo | grep 'model name' | uniq"
            },
            "security_scan": {
                "port scan": "nmap -sV -sC {TARGET}",
                "vulnerability": "nmap -sV --script vuln {TARGET}",
                "web": "nikto -h {TARGET}",
                "hosts": "nmap -sn {NETWORK}/24",
                "services": "nmap -sV {TARGET}",
                "quick": "nmap -F {TARGET}",
                "full": "nmap -p- {TARGET}",
                "udp": "nmap -sU -F {TARGET}",
                "os": "nmap -O {TARGET}",
                "discovery": "nmap -sn {NETWORK}/24"
            },
            "process_management": {
                "list": "ps aux",
                "running": "ps aux",
                "top": "top -n 1 -b",
                "cpu usage": "ps aux --sort=-%cpu | head -10",
                "memory usage": "ps aux --sort=-%mem | head -10",
                "kill": "pkill {PROCESS}",
                "find process": "pgrep -a {PROCESS}",
                "process": "ps aux | grep {PROCESS}"
            },
            "file_operations": {
                "list": "ls -la",
                "find": "find . -name '{PATTERN}' -type f",
                "search": "grep -r '{PATTERN}' .",
                "space": "du -sh * | sort -hr",
                "disk": "df -h",
                "content": "cat {FILE}",
                "permissions": "ls -la {FILE}"
            },
            "hash_identification": {
                "default": "hashcat --identify {FILE}",
                "identify": "hashcat --identify {FILE}",
                "detect": "hashcat --identify {FILE}",
                "determine": "hashcat --identify {FILE}",
                "check": "hashcat --identify {FILE}"
            }
        }
        
        # Response templates for agent-related queries
        self.response_templates = {
            "agent_identity": [
                "I am Nikita, an Offline AI Security Assistant designed to help with security tasks and system operations.",
                "I'm Nikita, your Offline Security Assistant. I can help you with security tasks, reconnaissance, and system information.",
                "Nikita here - I'm an AI security assistant that works offline to help with network analysis and security operations."
            ],
            "agent_status": [
                "I'm operational and ready to assist with your security tasks.",
                "Systems nominal and ready to help with your requests.",
                "I'm functioning properly and ready to assist you."
            ]
        }
        
        # Command output quality assessment patterns
        self.quality_indicators = {
            "empty_output": r'^\s*$',
            "error_patterns": [
                r'command not found', r'invalid option', r'unknown option',
                r'unreachable', r'operation not permitted', r'permission denied',
                r'unable to resolve', r'failed', r'error:',
                r'not recognized', r'100% packet loss'
            ],
            "success_patterns": {
                "ip": [r'inet\s+\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', r'mtu', r'link/ether'],
                "ping": [r'bytes from', r'icmp_seq=', r'time='],
                "nmap": [r'open\s+\w+', r'Nmap scan report', r'Host is'],
                "wifi": [r'SSID', r'MODE', r'CHAN', r'SIGNAL'],
                "ps": [r'PID', r'CPU', r'MEM', r'CMD'],
                "netstat": [r'LISTEN', r'ESTABLISHED', r'Proto', r'Address'],
                "hashcat": [r'hash-mode', r'Name', r'Category', r'\d+ \|']
            }
        }

    def analyze(self, query):
        """Analyze user query to determine intent and appropriate action"""
        query_lower = query.lower()
        
        # Special case for hash identification
        if any(re.search(pattern, query_lower) for pattern in self.intent_categories["hash_identification"]):
            # Extract file path from query
            file_path = re.search(r'(/[^ ]+)', query)
            if file_path:
                cmd = f"hashcat --identify {file_path.group(0)}"
                return {
                    "intent": "hash_identification",
                    "confidence": 0.95,
                    "pattern_matched": "hash identification",
                    "command": cmd,
                    "should_execute": True
                }
            else:
                return {
                    "intent": "hash_identification",
                    "confidence": 0.95,
                    "pattern_matched": "hash identification",
                    "response": "Please provide the path to the hash file you want to identify.",
                    "should_execute": False
                }
        
        # Special case for "scan my network" or "scan network" commands
        if re.search(r'scan\s+(my\s+)?network', query_lower):
            # Get the default network scan command
            cmd = self.command_mappings["network_devices"]["default"]
            return {
                "intent": "network_devices",
                "confidence": 0.95,
                "pattern_matched": "scan network",
                "command": cmd,
                "should_execute": True
            }
        
        # Check for direct agent identity queries with higher priority
        for pattern in self.intent_categories["agent_identity"]:
            if re.search(pattern, query_lower):
                return {
                    "intent": "agent_identity",
                    "confidence": 0.95,
                    "pattern_matched": pattern,
                    "response": self._get_agent_response("agent_identity"),
                    "should_execute": False
                }
        
        # Check for direct agent status queries with higher priority
        for pattern in self.intent_categories["agent_status"]:
            if re.search(pattern, query_lower):
                return {
                    "intent": "agent_status",
                    "confidence": 0.95,
                    "pattern_matched": pattern,
                    "response": self._get_agent_response("agent_status"),
                    "should_execute": False
                }
        
        # Check for direct command requests first
        if query_lower.startswith(('run ', 'execute ', 'show ')):
            return {
                "intent": "direct_command",
                "confidence": 1.0,
                "command": query.split(' ', 1)[1].strip()
            }
        
        # Check for agent-related intents
        for intent, patterns in self.intent_categories.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    result = {
                        "intent": intent,
                        "confidence": 0.8,
                        "pattern_matched": pattern
                    }
                    
                    # For system or security intents, determine appropriate command
                    if intent in ["system_info", "security_scan", "wifi_operations"]:
                        result["command"] = self._determine_command(intent, query_lower)
                    
                    # For agent-related intents, determine response
                    if intent in ["agent_identity", "agent_status"]:
                        result["response"] = self._get_agent_response(intent)
                        result["should_execute"] = False
                    else:
                        result["should_execute"] = True
                        
                    return result
        
        # If no specific intent is matched, treat as potential command
        return {
            "intent": "unknown",
            "confidence": 0.3,
            "should_execute": False
        }

    def _determine_command(self, intent, query):
        """Determine appropriate command based on intent and query"""
        mapping = self.command_mappings.get(intent, {})
        
        # Check for exact matches
        for keyword, command in mapping.items():
            if keyword in query:
                # Extract potential targets
                target_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', query)
                target = target_match.group(0) if target_match else "127.0.0.1"
                
                # Extract potential network
                network_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', query)
                network = network_match.group(0) if network_match else "192.168.1.0/24"
                
                # Replace placeholders
                command = command.replace("{TARGET}", target)
                command = command.replace("{NETWORK}", network)
                
                return command
        
        # Handle special cases
        if intent == "wifi_operations" and "scan" in query:
            return "nmcli dev wifi list"
        
        if intent == "system_info" and "ip" in query:
            if "eth0" in query or "ethernet" in query:
                return "ip -4 addr show eth0"
            elif "wlan" in query or "wifi" in query:
                return "ip -4 addr show wlan0"
            else:
                return "ip -4 addr show"
                
        if intent == "security_scan":
            # Extract IP if present
            ip_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', query)
            target = ip_match.group(0) if ip_match else "127.0.0.1"
            
            if "port" in query or "service" in query:
                return f"nmap -sV -sC {target}"
            elif "vuln" in query:
                return f"nmap -sV --script vuln {target}"
            elif "web" in query:
                return f"nikto -h {target}"
            else:
                return f"nmap -sV {target}"
        
        return None
    
    def _get_agent_response(self, intent):
        """Get an appropriate response for agent-related queries"""
        templates = self.response_templates.get(intent, ["I'm Nikita, your security assistant."])
        return random.choice(templates)
    
    def format_command_response(self, command, output, error=None):
        """Format a helpful response for command execution results with enhanced output validation"""
        # Handle hashcat --identify output specifically
        if command.startswith("hashcat --identify"):
            if not output.strip():
                return "❌ No hash was found in the specified file. Please verify the file contains a valid hash."
            if "No hash-mode match found" in output:
                return "❌ Could not identify the hash type. The file might not contain a valid hash format."
            
            # Look for hash mode matches
            mode_matches = re.findall(r'(\d+) \| ([^|]+)\|([^\n]+)', output)
            if mode_matches:
                response = "✅ Identified possible hash types:\n"
                for mode_id, name, category in mode_matches:
                    response += f"• Mode {mode_id}: {name.strip()} ({category.strip()})\n"
                return response
            else:
                return "⚠️ Hash identification completed but no standard hash formats were detected."
        
        # Check for empty output or errors
        if not output.strip():
            if error and error.strip():
                return f"Command encountered an error: {error.strip()}"
            return "Command executed successfully, but didn't produce any output. This might indicate the command didn't find what it was looking for."
        
        # Check for error patterns in output
        for pattern in self.quality_indicators["error_patterns"]:
            if re.search(pattern, output, re.IGNORECASE) or (error and re.search(pattern, error, re.IGNORECASE)):
                return f"Command executed but encountered issues: {re.search(pattern, output or error, re.IGNORECASE).group(0)}"
        
        # Get command base for specialized formatting
        cmd_base = command.split()[0]
        cmd_args = " ".join(command.split()[1:]) if len(command.split()) > 1 else ""
        
        # Validate output quality based on command type
        success_patterns = self.quality_indicators["success_patterns"]
        
        # Check if command output contains expected patterns
        if cmd_base in success_patterns:
            valid_output = any(re.search(pattern, output) for pattern in success_patterns[cmd_base])
            if not valid_output:
                return f"Command executed, but the output doesn't contain expected information. This might indicate {cmd_base} couldn't retrieve the requested data."
        
        # Handle specific commands with specialized formatting
        if cmd_base == "ip" and "addr" in command:
            # Extract and format IP addresses
            interfaces = []
            current_iface = None
            ip_addresses = []
            
            for line in output.strip().split('\n'):
                # Detect interface lines
                if ': ' in line and '<' in line and '>' in line:
                    current_iface = line.split(':', 1)[0].strip().strip(':')
                    state = "UP" if "UP" in line else "DOWN"
                    interfaces.append(f"{current_iface} ({state})")
                # Detect IPv4 address lines
                elif current_iface and "inet " in line:
                    ip = line.strip().split()[1].split('/')[0]
                    ip_addresses.append(f"{ip} ({current_iface})")
            
            if ip_addresses:
                formatted_output = "Found these IP addresses:\n"
                for i, ip in enumerate(ip_addresses, 1):
                    formatted_output += f"{i}. {ip}\n"
                return formatted_output.strip()
            elif interfaces:
                return f"Found network interfaces:\n" + "\n".join(interfaces) + "\nBut no IP addresses were assigned."
            else:
                return "No active network interfaces or IP addresses were found."
                
        # Ping command with improved failure detection
        elif cmd_base == "ping":
            target = command.split()[-1]
            if "0 received" in output or "100% packet loss" in output:
                return f"❌ Target {target} is not responding to ping. This could indicate the host is offline, blocking ICMP packets, or there's a network connectivity issue."
            elif "icmp_seq=" in output:
                # Extract ping stats
                stats_match = re.search(r'(\d+) packets transmitted, (\d+) received', output)
                time_match = re.search(r'min/avg/max/(?:mdev|stddev) = ([\d.]+)/([\d.]+)/([\d.]+)', output)
                
                if stats_match:
                    sent, received = stats_match.groups()
                    loss_percent = 100 - (int(received) / int(sent) * 100)
                    
                    status = "✅ up and responding normally" if loss_percent == 0 else f"⚠️ responding with {loss_percent}% packet loss"
                    
                    if time_match and len(time_match.groups()) >= 2:
                        avg_time = time_match.groups()[1]
                        return f"Target {target} is {status} (avg response time: {avg_time}ms)"
                    
                    return f"Target {target} is {status} (received {received}/{sent} packets)"
            
            return f"Ping to {target} completed, but the results are inconclusive."
            
        # Nmap command with improved output analysis
        elif cmd_base == "nmap":
            target = [arg for arg in command.split() if not arg.startswith('-')][-1]
            
            if "0 hosts up" in output or "All 1000 scanned ports on" in output and "closed" in output:
                return f"❌ Scan of {target} found no open ports or services. Target may be offline, firewalled, or not running any services on scanned ports."
            
            # Handle host discovery scans (-sn) for network device queries
            if "-sn" in command:
                # Count the hosts found
                hosts_up = re.findall(r'Host is up', output)
                host_count = len(hosts_up)
                
                # Extract IP addresses
                ip_addresses = re.findall(r'Nmap scan report for (?:[a-zA-Z0-9-]+\s)?\(?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\)?', output)
                
                # Format the response
                if host_count > 0:
                    response = f"✅ Found {host_count} devices on your network:\n"
                    for i, ip in enumerate(ip_addresses, 1):
                        # Try to extract hostnames if available
                        hostname_match = re.search(rf'Nmap scan report for ([a-zA-Z0-9-_.]+)\s+\(?{ip}\)?', output)
                        if not hostname_match:
                            # Try alternative pattern without parentheses
                            hostname_match = re.search(rf'Nmap scan report for ([a-zA-Z0-9-_.]+)(?!\s+\({ip}\))', output)
                        hostname = hostname_match.group(1) if hostname_match else ""
                        
                        # Add hostname if available
                        if hostname and hostname != ip:
                            response += f"{i}. {ip} ({hostname})\n"
                        else:
                            response += f"{i}. {ip}\n"
                    
                    # Add note about localhost
                    if "127.0.0.1" in ip_addresses:
                        response += "\nNote: This includes your local machine (127.0.0.1)."
                    
                    return response
                else:
                    return "No devices were found on your network. This might indicate network connectivity issues or nmap couldn't detect any responding hosts."
            
            # Extract open ports with improved formatting
            open_ports = []
            for line in output.split('\n'):
                if 'open' in line and '/tcp' in line:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        port = parts[0].split('/')[0]
                        service = parts[2]
                        version = ' '.join(parts[3:]) if len(parts) > 3 else ""
                        open_ports.append(f"Port {port}: {service} {version}")
            
            if open_ports:
                return f"✅ Scan of {target} found {len(open_ports)} open ports:\n" + "\n".join(open_ports)
            else:
                # Look for other useful information
                host_status = "Host appears to be up" if "Host is up" in output else "Host status unclear"
                latency = re.search(r'latency: ([\d.]+)s', output)
                latency_info = f" (latency: {latency.group(1)}s)" if latency else ""
                
                return f"⚠️ Scan of {target} completed. {host_status}{latency_info}. No open ports were found in the scan range."
                
        # Handle specific commands with specialized formatting
        elif cmd_base == "hashcat" and "--identify" in command:
            # Extract the hash mode information
            if "No hash-mode match found" in output:
                return "❌ Could not identify the hash type. The file might not contain a valid hash format."
            
            # Look for hash mode matches
            mode_matches = re.findall(r'(\d+) \| ([^|]+)\|([^\n]+)', output)
            if mode_matches:
                response = "✅ Identified possible hash types:\n"
                for mode_id, name, category in mode_matches:
                    response += f"• Mode {mode_id}: {name.strip()} ({category.strip()})\n"
                return response
            else:
                return "⚠️ Hash identification completed but no standard hash formats were detected."
                
        # Default - return condensed output with validation
        output_lines = output.strip().split('\n')
        if len(output_lines) > 10:
            summary = "\n".join(output_lines[:5]) + f"\n... and {len(output_lines)-5} more lines"
            return f"Command returned {len(output_lines)} lines of output. Here's a sample:\n{summary}"
        
        # If we got here, just return the formatted output
        return output.strip()

# Initialize the intent analyzer
intent_analyzer = IntentAnalyzer()

# === REPLACE main() FUNCTION ===
def main():
    global chat_memory, system_commands, categorized_commands
    
    # Initialize command discovery
    system_commands, categorized_commands = discover_system_commands()
    
    # Setup command history with readline
    history_enabled = setup_command_history()
    
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        console.print(f"\n[bold red]Error:[/bold red] Model file not found at {MODEL_PATH}")
        console.print("[yellow]Please ensure the model file is placed in the correct location.[/yellow]")
        return
    
    chat_memory = load_chat_history()
    
    # Print version banner
    console.print("\n[bold red]╔══════════════════════════════════════╗[/bold red]")
    console.print("[bold red]║[/bold red]     [bold white]NIKITA AI AGENT v1.0[/bold white]      [bold red]║[/bold red]")
    console.print("[bold red]╚══════════════════════════════════════╝[/bold red]\n")
    
    console.print("✅ [bold green]Nikita (Offline Operator Mode) Loaded[/bold green]")
    console.print("💡 [cyan]Tip: Press Ctrl+C during thinking to interrupt and return to prompt[/cyan]")
    
    if history_enabled:
        console.print("🔼 [cyan]Up/Down arrows: Browse command history, Left/Right: Edit command[/cyan]")
        console.print("⌨️ [cyan]Ctrl+A/E: Start/End of line, Ctrl+R: Search history, Tab: Complete[/cyan]")
    
    if chat_memory:
        console.print(f"📚 [cyan]Loaded {len(chat_memory)} previous messages[/cyan]")
    if engagement_memory["targets"]:
        console.print(f"🎯 [blue]Active targets:[/blue] {', '.join(engagement_memory['targets'])}")

    while True:
        try:
            console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
            console.print(f"[bold cyan]└─>[/bold cyan] ", end="")
            
            # Get user input with readline support (history, editing)
            if history_enabled:
                user_input = get_input_with_history()
            else:
                user_input = input().strip()
            
            # Handle empty input by continuing to the next loop iteration
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                save_chat_history(chat_memory)
                if history_enabled:
                    save_command_history()
                console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                break

            # Add user input to chat memory
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })

            # Enforce memory limit
            if len(chat_memory) > MEMORY_LIMIT:
                chat_memory = chat_memory[-MEMORY_LIMIT:]

            # Extract targets and suggest attack plan
            extract_targets(user_input)
            attack_plan = suggest_attack_plan(user_input)
            if attack_plan:
                console.print(f"\n[yellow][Attack Plan][/yellow] {attack_plan}")

            # Generate reasoning about the task (but don't display it)
            reasoning = reasoning_engine.analyze_task(user_input)
            
            # NEW: Analyze intent first
            intent_analysis = intent_analyzer.analyze(user_input)

            # Nested try block for processing the actual command/logic
            try:
                # Use context optimizer to get optimized prompt
                full_prompt = context_optimizer.get_optimized_prompt(
                    chat_memory=chat_memory[-3:],
                    current_task=user_input,
                    base_prompt="You are Nikita 🐺, an Offline AI Security Assistant. Focus on the current task."
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task_id = progress.add_task("🐺 Reasoning...", total=None)
                    
                    # For direct agent identity or status queries, skip LLM call
                    if intent_analysis["intent"] in ["agent_identity", "agent_status"]:
                        response = intent_analysis["response"]
                        progress.stop()
                        console.print()
                        # Add console print statements to show who and how questions
                        console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                        console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                    else:
                        try:
                            output = llm(
                                full_prompt,
                                max_tokens=system_params['max_tokens'],
                                temperature=system_params['temperature'],
                                top_k=system_params['top_k'],
                                top_p=system_params['top_p'],
                                repeat_penalty=system_params['repeat_penalty'],
                                stop=["User:", "Task:", "\n\n"]
                            )
                        except KeyboardInterrupt:
                            progress.stop()
                            console.print("\n[yellow]Thinking interrupted. Back to prompt...[/yellow]")
                            continue
                            
                        progress.stop()
                        console.print()
                        response = output['choices'][0]['text'].strip()
                        
                        # Clean up response and execute command if present
                        response = response.translate(str.maketrans("", "", "`"))
                        
                        # Initialize command variable
                        cmd = None
                        
                        # First priority: Check intent-derived commands for hash identification
                        if intent_analysis["intent"] == "hash_identification" and intent_analysis.get("command"):
                            cmd = intent_analysis["command"]
                        
                        # Second priority: Check fine-tuning knowledge
                        elif not cmd:
                            fine_tuned_match = fine_tuning.get_command_for_task(user_input)
                            if fine_tuned_match:
                                cmd = fine_tuned_match["output"].split("\n")[0].replace("Command: ", "")
                        
                        # Third priority: Other intent-derived commands
                        elif not cmd and intent_analysis["intent"] not in ["agent_identity", "agent_status", "unknown"] and intent_analysis.get("command"):
                            cmd = intent_analysis["command"]
                        
                        # Extra handling for network scan command
                        if not cmd and ("scan" in user_input.lower() and "network" in user_input.lower()):
                            # Use direct nmap command for network scanning
                            cmd = "nmap -sn $(ip -4 addr show | grep -i 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1 | sed 's/[0-9]*$/0\\/24/')"
                        
                        # Fourth priority: Direct command execution via keywords
                        if not cmd and user_input.lower().startswith(('run ', 'execute ', 'show ')):
                            cmd = user_input.split(' ', 1)[1].strip()
                        
                        # Fifth priority: Handle scanning commands specifically
                        if not cmd and user_input.lower().startswith('scan '):
                            parts = user_input.split(' ', 1)
                            if len(parts) > 1 and parts[1].strip():
                                # Determine scan type
                                scan_text = parts[1].strip()
                                if "wifi" in scan_text.lower() or "wireless" in scan_text.lower():
                                    cmd = "nmcli dev wifi list"
                                elif re.search(r'(\d{1,3}\.){3}\d{1,3}', scan_text):
                                    # If IP is specified
                                    ip_match = re.search(r'(\d{1,3}\.){3}\d{1,3}', scan_text)
                                    target = ip_match.group(0)
                                    cmd = f"nmap -sV -sC {target}"
                                else:
                                    # Generic scan based on context
                                    if "vuln" in scan_text.lower():
                                        cmd = "nmap -sV --script vuln 127.0.0.1"
                                    elif "port" in scan_text.lower():
                                        cmd = "nmap -sV -p- 127.0.0.1"
                                    else:
                                        cmd = "nmap -sV 127.0.0.1"
                        
                        # Sixth priority: Extract command from LLM response
                        if not cmd and "how to" not in user_input.lower() and "what is" not in user_input.lower():
                            # Extract command patterns
                            command_pattern = re.search(r'^([a-zA-Z0-9_\-]+(?:\s+(?:-{1,2}[a-zA-Z0-9_\-]+|\S+))*)', response)
                            if command_pattern:
                                potential_cmd = command_pattern.group(1).strip().strip('"\'')
                                if ' ' in potential_cmd and potential_cmd.split()[0] in system_commands:
                                    cmd = potential_cmd
                        
                        # Execute the command if one was determined
                        if cmd:
                            validated_cmd, error = validate_command(cmd)
                            if validated_cmd:
                                try:
                                    # Execute the command
                                    console.print(f"[cyan]Executing: {validated_cmd}[/cyan]")
                                    
                                    cmd_output = subprocess.run(
                                        shlex.split(validated_cmd),
                                        capture_output=True,
                                        text=True,
                                        timeout=30
                                    )
                                    
                                    output_file = save_command_output(validated_cmd, cmd_output.stdout, cmd_output.stderr)
                                    
                                    # Display command output
                                    if cmd_output.stdout:
                                        console.print(f"[green]{cmd_output.stdout}[/green]")
                                    if cmd_output.stderr:
                                        console.print(f"[red]{cmd_output.stderr}[/red]")
                                    
                                    console.print(f"📝 [cyan]Output saved to:[/cyan] {output_file}")
                                    
                                    # Save execution to chat memory
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    chat_memory.append({
                                        "role": "assistant",
                                        "content": f"Executed: {validated_cmd}",
                                        "timestamp": timestamp,
                                        "output": cmd_output.stdout,
                                        "error": cmd_output.stderr
                                    })
                                    
                                    # NEW: Format a better response based on the command output
                                    formatted_response = intent_analyzer.format_command_response(
                                        validated_cmd, 
                                        cmd_output.stdout, 
                                        cmd_output.stderr
                                    )
                                    
                                    # If command produced empty or error output, make a note of it
                                    if not cmd_output.stdout.strip() or cmd_output.stderr.strip():
                                        if "Command executed successfully, but didn't produce any output" in formatted_response:
                                            console.print("[yellow]Note: Command did not produce output. Might need a different approach.[/yellow]")
                                    
                                    # Use the formatted response
                                    response = formatted_response
                                    
                                except subprocess.TimeoutExpired:
                                    console.print("[red]Error: Command timed out after 30 seconds[/red]")
                                    response = "The command timed out. Please try again or try a more specific command."
                                except Exception as e:
                                    console.print(f"[red]Error executing command: {e}[/red]")
                                    response = f"Error executing command: {str(e)}"
                
                        # Save chat memory and display response
                        chat_memory.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        save_chat_history(chat_memory)
                        
                        console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                        console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                        
            except ValueError as e:
                if "exceed context window" in str(e):
                    # Use a minimal prompt with just the essential information
                    minimal_prompt = f"You are Nikita, a security assistant. Task: {user_input}\nResponse:"
                    output = llm(minimal_prompt, max_tokens=system_params['max_tokens'])
                    response = output['choices'][0]['text'].strip()
                    
                    # Save the response to chat memory
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    chat_memory.append({
                        "role": "assistant",
                        "content": f"Reasoning:\n{reasoning}\n\nResponse: {response}",
                        "timestamp": timestamp
                    })
                    save_chat_history(chat_memory)
                    
                    console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                    console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                else:
                    raise e
            except Exception as e:
                console.print(f"\n[bold red]Error in command processing:[/bold red] {str(e)}")
                console.print("[yellow]Continuing to next interaction...[/yellow]")
                
        except KeyboardInterrupt:
            save_chat_history(chat_memory)
            if history_enabled:
                save_command_history()
            console.print("\n[bold red]Nikita:[/bold red] Caught interrupt signal. Exiting gracefully.\n")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            console.print("[yellow]Continuing to next interaction...[/yellow]\n")
            continue

def check_swap_usage():
    ram, swap, _, _ = get_system_info()
    if swap.used > (512 * 1024 * 1024) and not hasattr(check_swap_usage, "warning_shown"):
        console.print(f"⚠️  [bold red]High swap usage: {swap.used // (1024*1024)} MiB | Free RAM: {ram.available // (1024*1024)} MiB[/bold red]")
        check_swap_usage.warning_shown = True

# === Custom Input with History ===
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Nikita:[/bold red] Interrupted. Exiting.\n")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}")
    finally:
        # Ensure command history is saved when the program exits
        try:
            readline.write_history_file(COMMAND_HISTORY_FILE)
        except:
            pass

# === Command Enhancement ===
def harden_command(cmd):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    enhancements = {
        'nmap': [
            ('-sV', ' -sV'),
            ('-sC', ' -sC'),
            ('-sn', ' -sn'),
            ('-T4', ' -T4'),
            ('--stats-every', ' --stats-every 10s')
        ],
        'sqlmap': [
            ('--batch', ' --batch'),
            ('--random-agent', ' --random-agent')
        ],
        'gobuster': [
            ('-q', ' -q')
        ],
        'smbclient': [
            ('-N', '-N')
        ],
        'dig': [
            ('+short', ' +short')
        ],
        'hashcat': [
            ('--identify', ' --identify'),
            ('--quiet', ' --quiet'),
            ('--show', ' --show')
        ]
    }
    
    # Handle output paths first
    if cmd.startswith('nmap'):
        output_base = os.path.join(OUTPUT_DIR, f"nmap_scan_{timestamp}")
        if not any(opt in cmd for opt in ['-oA', '-oX', '-oN', '-oG', '-oS']):
            cmd += f" -oA {output_base}"
    elif cmd.startswith('sqlmap'):
        sql_output = os.path.join(OUTPUT_DIR, f"sqlmap_{timestamp}")
        os.makedirs(sql_output, exist_ok=True)
        if '--output-dir' not in cmd:
            cmd += f" --output-dir={sql_output}"
    elif cmd.startswith('gobuster'):
        if '-o' not in cmd:
            cmd += f" -o {os.path.join(OUTPUT_DIR, f'gobuster_{timestamp}.txt')}"
    
    # Extract IP range if present
    ip_range_match = re.search(r'(\d{1,3}\.){3}\d{1,3}(?:-\d{1,3})?', cmd)
    if ip_range_match:
        ip_range = ip_range_match.group(0)
        if '-' in ip_range:
            base_ip = ip_range.split('-')[0]
            cmd = cmd.replace(ip_range, base_ip + '/24')
    
    # Apply standard enhancements
    for tool, rules in enhancements.items():
        if cmd.startswith(tool):
            for check, add in rules:
                if check not in cmd:
                    cmd += add
    
    return cmd

def save_command_output(cmd, output, error=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"cmd_{timestamp}.txt")
    with open(output_file, 'w') as f:
        f.write(f"=== Command ===\n{cmd}\n\n")
        f.write(f"=== Output ===\n{output}\n")
        if error:
            f.write(f"\n=== Errors ===\n{error}\n")
    
    return output_file

def run_command(cmd):
    if cmd.count('.') < 3 and " " in cmd:
        console.print(f"❌ [red]Command incomplete:[/red] {cmd}")
        return

    try:
        # Harden and normalize command
        cmd = harden_command(cmd)
        console.print(f"⚡ [bold cyan]Running:[/bold cyan] {cmd}")
        
        cmd_list = shlex.split(cmd)
        result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        output_file = save_command_output(cmd, result.stdout, result.stderr)
        
        if result.stdout.strip():
            # Format nmap output for better readability
            if cmd.startswith("nmap"):
                output_lines = result.stdout.strip().split("\n")
                formatted_output = "\n".join(f"  {line}" if "open" in line else line for line in output_lines)
                console.print(f"🖥️  [green]{formatted_output}[/green]")
            else:
                console.print(f"🖥️  [green]{result.stdout.strip()}[/green]")
            console.print(f"📝 [cyan]Output saved to:[/cyan] {output_file}")
        else:
            console.print(f"⚠️  [yellow]No output returned.[/yellow]")
            
    except subprocess.TimeoutExpired:
        console.print("❌ [red]Command timed out after 5 minutes[/red]")
        save_command_output(cmd, "", "Command timed out after 5 minutes")
    except Exception as e:
        console.print(f"❌ [red]Error:[/red] {e}")
        save_command_output(cmd, "", str(e))

