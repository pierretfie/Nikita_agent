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
import readline  # Add readline for command history and editing
import warnings
import contextlib

# Determine the directory of the main script (Nikita_agent.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base directory for Nikita (where model and output files are)
NIKITA_BASE_DIR = os.path.join(os.path.expanduser("~"), "tinyllama")

# Construct absolute paths relative to NIKITA_BASE_DIR
MODEL_PATH = os.path.join(NIKITA_BASE_DIR, "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")
OUTPUT_DIR = os.path.join(NIKITA_BASE_DIR, "outputs")
HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "history.json")
CHAT_HISTORY_FILE = Path(os.path.join(NIKITA_BASE_DIR, "nikita_history.json"))
COMMAND_HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "command_history")

# Construct absolute paths to scripts (in the same directory as Nikita_agent.py)
PROMPT_TEMPLATE_FILE = os.path.join(SCRIPT_DIR, "modules", "prompt_template.txt")
FINE_TUNING_FILE = os.path.join(SCRIPT_DIR, "modules", "fine_tuning.json")

# Add the script directory to the Python path
sys.path.insert(0, SCRIPT_DIR)
from modules.intent_analyzer import IntentAnalyzer
from modules.resource_management import get_system_info, get_dynamic_params, optimize_memory_resources, optimize_cpu_usage, prewarm_model
from modules.history_manager import setup_command_history, save_command_history, get_input_with_history, load_chat_history, save_chat_history
from modules.context_optimizer import ContextOptimizer
from modules.command_handler import run_command
from modules.engagement_manager import extract_targets, suggest_attack_plan, engagement_memory
from modules.reasoning_engine import ReasoningEngine
from modules.tool_manager import ToolManager
from modules.gpu_manager import GPUManager


console = Console()

# Create necessary directories
os.makedirs(NIKITA_BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
#max tokens refer to the maximum number of tokens that the model can generate in a single response.
MAX_TOKENS = 2048  #  from 512
TEMPERATURE = 0.3  # Reduced from 0.7 for more focused responses
# Maximum number of messages to keep in memory
MEMORY_LIMIT = 20  # Set a reasonable limit for memory usage

# Get system parameters for model initialization
system_params = get_dynamic_params()

# Global variables for commands
system_commands = {}
categorized_commands = {}

# ===============================
# === COMMAND HISTORY ===========
# ===============================

def discover_system_commands():
    """Discover available system commands"""
    global system_commands, categorized_commands
    
    commands = {}
    categorized = {}
    
    # Common security tools
    security_tools = {
        "nmap": "Network scanner",
        "metasploit": "Exploitation framework",
        "hydra": "Login brute forcer",
        "hashcat": "Password cracker",
        "gobuster": "Directory bruteforcer",
        "wireshark": "Network analyzer",
        "aircrack-ng": "WiFi security tool",
        "burpsuite": "Web security tool",
        "sqlmap": "SQL injection tool"
    }
    
    # Populate some basic commands
    base_commands = {
        "ls": "List directory contents",
        "cd": "Change directory",
        "cat": "Display file contents",
        "grep": "Search text",
        "find": "Find files",
        "ip": "Show IP information",
        "ps": "Process status",
        "kill": "Terminate processes"
    }
    
    commands.update(base_commands)
    commands.update(security_tools)
    
    # Categorize commands
    categorized["file"] = ["ls", "cat", "find"]
    categorized["network"] = ["nmap", "ip", "wireshark"]
    categorized["process"] = ["ps", "kill"]
    categorized["security"] = ["metasploit", "hydra", "hashcat", "gobuster", "aircrack-ng", "burpsuite", "sqlmap"]
    
    system_commands = commands
    categorized_commands = categorized
    
    return commands, categorized

# Initialize system commands
system_commands, categorized_commands = discover_system_commands()

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
            with open(FINE_TUNING_FILE, "r") as f:
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

                # Don't display knowledge loading information to keep output clean
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

# ===============================
# === REASONING ENGINE =========
# ===============================

# Initialize reasoning engine
reasoning_engine = ReasoningEngine()

# ===============================
# === MEMORY ENGINE =============
# ===============================

# Memory for storing information
memory = {
    "targets": [],
    "credentials": [],
    "loot": [],
    "network_maps": [],
    "system_info": {}
}

# Chat history
chat_memory = []

# Engagement memory
engagement_memory = {
    "targets": [],
    "credentials": [],
    "loot": [],
    "network_maps": [],
    "attack_history": []
}

# ===============================
# === CONTEXT OPTIMIZER ========
# ===============================

# No need to redefine ContextOptimizer class since we're importing it
# from modules.context_optimizer

# ===============================
# === Model Setup ===
console.print("🧠 [bold red]Waking Nikita 🐺...[/bold red]")

# Optimize system resources
success, aggressive_mode = optimize_memory_resources()

# Get system parameters
system_params = get_dynamic_params()

# Memory usage statistics - using simplified format
ram, swap, cpu_count, ram_gb = get_system_info()
console.print(f"[green]⚙️ RAM Tier: {ram_gb:.1f}GB system | Using {int(system_params['memory_target_gb'])}GB | Target: {system_params['memory_target_pct']:.1f}%[/green]")
console.print(f"[green]📊 Current usage: {ram.used/1024/1024/1024:.1f}GB ({ram.percent:.1f}%) | Context: {system_params['context_limit']} tokens | Batch: {system_params['n_batch']}[/green]")

# Display memory optimization status if used aggressive mode
if aggressive_mode:
    console.print("💫 [green]Aggressive memory optimization activated[/green] ")

# Apply CPU optimizations
success, target_cores, current_load = optimize_cpu_usage()
console.print(f"[green]⚡ CPU affinity set to use {target_cores} cores based on current load ({current_load:.2f})[/green]  ")

# Ensure mlock is enabled for RAM optimization
try:
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
except:
    pass

# === GPU Power Check Function ===
def is_gpu_powerful(device_info):
    """Determine if a GPU is powerful based on its specifications"""
    if not device_info:
        return False
        
    # Memory check (in bytes)
    memory_gb = device_info['global_mem_size'] / (1024**3)
    memory_powerful = memory_gb >= 8  # 8GB or more is considered powerful
    
    # Compute units check
    compute_units = device_info['max_compute_units']
    compute_powerful = compute_units >= 16  # 16 or more compute units is powerful
    
    # Work group size check
    work_group_size = device_info['max_work_group_size']
    work_group_powerful = work_group_size >= 256  # 256 or more is powerful
    
    # Overall assessment
    is_powerful = (
        memory_powerful and 
        compute_powerful and 
        work_group_powerful
    )
    
    return {
        'is_powerful': is_powerful,
        'memory_gb': memory_gb,
        'compute_units': compute_units,
        'work_group_size': work_group_size,
        'details': {
            'memory_powerful': memory_powerful,
            'compute_powerful': compute_powerful,
            'work_group_powerful': work_group_powerful
        }
    }

# === Load Prompt Template ===
try:
    with open(PROMPT_TEMPLATE_FILE, "r") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    console.print(f"[red]Error:[/red] Prompt template file not found at {PROMPT_TEMPLATE_FILE}")
    PROMPT_TEMPLATE = "You are a helpful AI assistant."  # Default prompt
except Exception as e:
    console.print(f"[yellow]Could not load prompt template: {e}[/yellow]")
    PROMPT_TEMPLATE = "You are a helpful AI assistant."  # Default prompt

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
llm = None # Initialize llm to None
with suppress_stderr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Initialize GPU manager for parallel processing
            gpu_manager = GPUManager()
            gpu_manager.initialize()
            device_info = gpu_manager.get_device_info()
            # Check if device_info is valid before accessing keys
            if device_info:
                power_analysis = is_gpu_powerful(device_info)
                
                if power_analysis['is_powerful']:
                    # Use more GPU-intensive settings
                    work_split_ratio = 0.7  # 70% GPU, 30% CPU
                    tensor_split = [0.3, 0.7]  # More work on GPU
                else:
                    # Use more CPU-intensive settings
                    work_split_ratio = 0.3  # 30% GPU, 70% CPU
                    tensor_split = [0.7, 0.3]  # More work on CPU
                
                console.print(f"[green]⚡ Parallel Processing Configuration:[/green]")
                console.print(f"[green]  • Device: {device_info['name']}[/green]")
                console.print(f"[green]  • Compute Units: {device_info['max_compute_units']}[/green]")
                console.print(f"[green]  • Global Memory: {device_info['global_mem_size'] / (1024*1024):.1f} MB[/green]")
                console.print(f"[green]  • Max Work Group Size: {device_info['max_work_group_size']}[/green]")
                console.print(f"[green]  • Parallel Processing: Enabled[/green]")
            else:
                console.print("[red]Error getting device info. Cannot configure parallel processing.[/red]")
                # Set default splits if device info is unavailable
                work_split_ratio = 0.5
                tensor_split = [0.5, 0.5]

            # Initialize Llama model
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=system_params['context_limit'],
                n_threads=system_params['n_threads'], # Use all dynamic threads for CPU
                n_batch=system_params['n_batch'],
                # max_tokens=system_params['max_tokens'], # max_tokens is set during generation, not init
                use_mlock=True,
                use_mmap=True,
                low_vram=True,  # Keep low VRAM mode enabled
                verbose=False,
                f16_kv=True,
                seed=42,
                embedding=False,
                rope_scaling={"type": "linear", "factor": 0.25},
                n_gpu_layers=0,  # Set to 0 since GPU is not used for Llama layers
                vocab_only=False,
                # tensor_split=tensor_split, # Remove, not relevant for n_gpu_layers=0
                logits_all=False,
                last_n_tokens_size=32,
                cache=True
                # gpu_device=0, # Remove, not relevant
                # gpu_memory_utilization=0.85, # Remove, not relevant
                # use_opencl=True,  # Remove, let library decide based on build/availability if needed elsewhere
                # opencl_context=gpu_manager.context,  # Remove, not used by Llama with n_gpu_layers=0
                # opencl_queue=gpu_manager.queue,  # Remove, not used by Llama with n_gpu_layers=0
                # parallel_processing=True,  # Remove, Llama won't parallelize with GPU here
                # cpu_threads=system_params['n_threads'] // 2,  # Remove, use n_threads directly
                # gpu_threads=system_params['n_threads'] // 2,  # Remove
                # work_split_ratio=work_split_ratio # Remove
            )
            
            # Prewarm the model AFTER successful initialization
            prewarm_duration = prewarm_model(llm, base_prompt="You are Nikita, an AI Security Assistant.")
            console.print(f"✅ [green] Model prewarmed in {prewarm_duration:.2f} seconds[/green]")

        except Exception as e:
            console.print(f"[red]Error initializing model or GPU manager: {str(e)}[/red]")
            # Ensure GPU manager is cleaned up even if Llama init failed
            if 'gpu_manager' in locals() and gpu_manager is not None:
                gpu_manager.cleanup()
            sys.exit(1)

# Initialize model cache
MODEL_CACHE = {}

# Initialize fine tuning knowledge
fine_tuning = FinetuningKnowledge()

# Initialize system commands
system_commands, categorized_commands = discover_system_commands()

# Initialize intent analyzer
intent_analyzer = IntentAnalyzer(OUTPUT_DIR, system_commands)

# Initialize context optimizer - ensure llm exists if needed, or pass None/handle later
context_optimizer = ContextOptimizer(
    max_tokens=system_params['context_limit'],
    reserve_tokens=system_params['max_tokens'] # Assuming reserve_tokens is independent of llm
)

# Define a function to get responses with optimized caching
def get_cached_response(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """Get response from model with optimized caching and error handling"""
    try:
        # Check cache first
        cache_key = f"{prompt}_{max_tokens}_{temperature}"
        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key]

        # Generate new response with optimized settings
        try:
            output = llm(prompt, 
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["User:", "\nUser:", "USER:"],
                        echo=False,  # Disable echo for faster response
                        stream=False)  # Disable streaming for faster response
            
            # Cache the response
            if len(MODEL_CACHE) > 20:  # Limit cache size
                MODEL_CACHE.clear()
            MODEL_CACHE[cache_key] = output
            
            return output
        except Exception as e:
            console.print(f"[yellow]Model inference error: {str(e)}[/yellow]")
            return {"choices": [{"text": "I apologize, but I encountered an error processing your request."}]}
    except Exception as e:
        console.print(f"[yellow]Error: {str(e)}[/yellow]")
        return {"choices": [{"text": "I apologize, but I encountered an error processing your request."}]}

# Helper function for command confirmation and execution
def confirm_and_run_command(cmd):
    """Displays the command and asks for user confirmation before running."""
    if not cmd:
        console.print("[yellow]Attempted to run an empty command.[/yellow]")
        return

    console.print(f"\n[bold yellow]Proposed Command:[/bold yellow] [cyan]{shlex.quote(cmd)}[/cyan]")
    try:
        confirm = input("Execute this command? (Y/N): ").strip().lower()
        if confirm == 'y' or confirm == 'yes':
            console.print(f"[green]Executing command...[/green]")
            run_command(cmd) # Use the existing run_command function
        else:
            console.print("[yellow]Command execution skipped by user.[/yellow]")
    except EOFError:
        console.print("\n[yellow]Input stream closed. Command execution skipped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during command confirmation: {e}. Command execution skipped.[/red]")

# === UTILITY FUNCTIONS ===

# Note: The following functions have been moved to modules.engagement_manager
# - extract_targets
# - suggest_attack_plan

# Note: The following functions have been moved to modules.history_manager
# - load_chat_history
# - save_chat_history

# Note: The following functions have been moved to modules.command_handler
# - save_command_output
# - run_command # We keep importing this as the confirmation helper calls it

# === REPLACE main() FUNCTION ===
def main():
    """Main function to run the Nikita agent"""
    global chat_memory, llm, gpu_manager # Ensure llm and gpu_manager are accessible

    # Setup command history with readline
    history_enabled = setup_command_history()

    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        console.print(f"\n[bold red]Error:[/bold red] Model file not found at {MODEL_PATH}")
        console.print("[yellow]Please ensure the model file is placed in the correct location.[/yellow]")
        return
        
    # Check if llm was successfully initialized before proceeding
    if llm is None:
        console.print("[red]Model initialization failed. Exiting.[/red]")
        # Attempt cleanup even if init failed
        if 'gpu_manager' in globals() and gpu_manager is not None:
             gpu_manager.cleanup()
        return # Exit main function gracefully

    chat_memory = load_chat_history(memory_limit=MEMORY_LIMIT, chat_history_file=CHAT_HISTORY_FILE)

    # Print version banner
    console.print("\n[bold red]╔══════════════════════════════════════╗[/bold red]")
    console.print("[bold red]║[/bold red]     [bold white]NIKITA 🐺 AI AGENT v1.0[/bold white]      [bold red]║[/bold red]")
    console.print("[bold red]╚══════════════════════════════════════╝[/bold red]\n")

    # Import threading capabilities
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # Model prewarming is now done during initialization phase above
    # prewarm_duration = prewarm_model(llm, base_prompt="You are Nikita, an AI Security Assistant.")
    # console.print(f"✅ [cyan] Model prewarmed in {prewarm_duration:.2f} seconds[/cyan]")
    
    console.print("✅ [cyan]Nikita (Offline Operator Mode) Loaded[/cyan]")
    
    if chat_memory:
        pass
        #console.print(f"💬 Loaded {len(chat_memory)} previous chat messages")
    
    # More concise display of active targets if any
    if engagement_memory["targets"]:
        console.print(f"🎯 Active targets: {', '.join(engagement_memory['targets'])}")

    # Create a thread pool for background tasks
    executor = ThreadPoolExecutor(max_workers=2)

    # Prefetch common prompts to warm up cache
    def prefetch_common_tasks():
        """Prefetch common prompts with better error handling"""
        try:
            # Reduce number of prefetch prompts
            simple_prompts = [
                "You are Nikita, an AI Security Assistant. How can I help?",
                "You are Nikita, an AI Security Assistant. Respond briefly.",
                "You are Nikita, an AI Security Assistant. Analyze this security concern."
            ]
            
            for prompt in simple_prompts:
                try:
                    # Add delay between prefetches
                    time.sleep(0.5)
                    # Run inference with minimal tokens in background
                    _ = llm(prompt, max_tokens=1)
                except Exception as e:
                    console.print(f"[yellow]Prefetch warning for prompt: {str(e)}[/yellow]")
                    continue
            return True
        except Exception as e:
            console.print(f"[yellow]Prefetch error: {str(e)}[/yellow]")
            return False

    # Start prefetching in background with better error handling
    try:
        prefetch_future = executor.submit(prefetch_common_tasks)
        
        # Let prefetch run without timeout
        try:
            prefetch_success = prefetch_future.result()  # No timeout
            if prefetch_success:
                console.print("[cyan]✅ Common prompts prefetched successfully[/cyan]")
            else:
                console.print("[yellow]⚠️ Some prompts failed to prefetch[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Prefetch error: {str(e)}[/yellow]")
            # Cancel the prefetch if it's still running
            if not prefetch_future.done():
                prefetch_future.cancel()
    except Exception as e:
        console.print(f"[yellow]Failed to start prefetch: {str(e)}[/yellow]")

    # Initialize tool manager after other initializations
    tool_manager = ToolManager(fine_tuning_file=FINE_TUNING_FILE)

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
                save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
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

            # Parallelize non-critical tasks
            def background_processing():
                try:
                    extract_targets(user_input)
                    return suggest_attack_plan(user_input)
                except Exception as e:
                    return None

            # Start background processing
            attack_plan_future = executor.submit(background_processing)

            # NEW: Analyze intent first (this is faster)
            intent_analysis = intent_analyzer.analyze(user_input)

            # Get attack plan result if ready, otherwise continue without waiting
            try:
                attack_plan = attack_plan_future.result(timeout=0.1)  # Short timeout
                if attack_plan:
                    console.print(f"\n[yellow][Attack Plan][/yellow] {attack_plan}")
            except:
                # Continue without attack plan if it's taking too long
                pass

            # Generate reasoning - do quickly and in background if possible
            try:
                # Pass the result of intent analysis to the reasoning engine
                reasoning_future = executor.submit(reasoning_engine.analyze_task, user_input, intent_analysis=intent_analysis)

                # Get reasoning result with timeout and better error handling
                try:
                    reasoning_result = reasoning_future.result(timeout=0.5)  # Short timeout to avoid blocking
                except Exception as e:
                    console.print(f"[yellow]Reasoning timeout or error: {str(e)}[/yellow]")
                    reasoning_result = {"reasoning": {}, "follow_up_questions": []}
                    if not reasoning_future.done():
                        reasoning_future.cancel()
            except Exception as e:
                console.print(f"[yellow]Failed to start reasoning: {str(e)}[/yellow]")
                reasoning_result = {"reasoning": {}, "follow_up_questions": []}

            # Use context optimizer to get optimized prompt
            full_prompt = context_optimizer.get_optimized_prompt(
                chat_memory=chat_memory[-3:],
                current_task=user_input,
                base_prompt="You are Nikita 🐺, an Offline AI Security Assistant. Focus on the current task.",
                reasoning_context=reasoning_result.get("reasoning", {}),
                follow_up_questions=reasoning_result.get("follow_up_questions", []),
                tool_context=tool_manager.get_tool_context(reasoning_result.get("tool_name")) if reasoning_result.get("tool_name") else None
            )

            # --- DEBUG START ---
            print(f"\n{'='*20} DEBUG INFO {'='*20}")
            print(f"Intent Analysis: {intent_analysis}")
            print(f"--- Full Prompt Sent to LLM: ---\n{full_prompt}")
            print("---------------------------------")
            # --- DEBUG END ---

            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                start_time = time.time()
                task_id = progress.add_task("🐺 Reasoning...", total=None)
                
                # Update the timer every 0.1 seconds
                timer_running = True
                def update_timer():
                    while timer_running:
                        elapsed = time.time() - start_time
                        progress.update(task_id, description=f"🐺 Reasoning... [{elapsed:.1f}s]")
                        time.sleep(0.1)
                
                # Start the timer update thread
                timer_thread = threading.Thread(target=update_timer)
                timer_thread.daemon = True
                timer_thread.start()

                # Generate response using cached response function for better performance
                output = get_cached_response(full_prompt)

                # --- DEBUG START ---
                print(f"--- Raw Output Received from LLM: ---\n{output}")
                print("---------------------------------")
                # --- DEBUG END ---

                # Stop the timer thread and progress spinner
                timer_running = False
                timer_thread.join(timeout=0.2)  # Wait for thread to finish
                progress.stop()
                
                # Display total elapsed time
                total_time = time.time() - start_time
                console.print(f"⏱️ {total_time:.1f}s")
                console.print() # Add an empty line for better separation
                
                response = output['choices'][0]['text'].strip()
                
                # Process any commands in the response
                command_match = re.search(r'```(.*?)```', response, re.DOTALL) or re.search(r'`(.*?)`', response, re.DOTALL)
                
                executed_command_this_turn = False # Flag to avoid double execution

                if intent_analysis and intent_analysis.get("command") and intent_analysis.get("should_execute", False):
                    # Execute command from intent analysis (with confirmation)
                    cmd = intent_analysis["command"]
                    
                    # Check if it's a help request (don't need confirmation for help)
                    if cmd.lower().startswith(('help', 'man')):
                        tool_name = cmd.split()[-1]
                        if tool_name in system_commands:
                            tool_help = tool_manager.get_tool_help(tool_name)
                            if tool_help:
                                console.print(f"\n[bold cyan]Help for {tool_name}:[/bold cyan]")
                                if tool_help.get("source") == "man_page":
                                    console.print(f"[bold]Name:[/bold] {tool_help.get('name', 'N/A')}")
                                    console.print(f"[bold]Synopsis:[/bold] {tool_help.get('synopsis', 'N/A')}")
                                    console.print(f"[bold]Description:[/bold] {tool_help.get('description', 'N/A')}")
                                    if tool_help.get('options'):
                                        console.print(f"[bold]Options:[/bold] {tool_help['options']}")
                                    if tool_help.get('examples'):
                                        console.print(f"[bold]Examples:[/bold] {tool_help['examples']}")
                                else:
                                    console.print(tool_help.get('help_text', 'No help text found.'))
                            else:
                                console.print(f"[yellow]No help information available for {tool_name}[/yellow]")
                            executed_command_this_turn = True # Treat help display as handled
                        else:
                            # If help is for an unknown command, ask to run it
                            confirm_and_run_command(cmd)
                            executed_command_this_turn = True
                    else:
                        # Ask for confirmation for non-help commands
                        confirm_and_run_command(cmd)
                        executed_command_this_turn = True

                elif command_match and not executed_command_this_turn:
                    # Execute command from response (with confirmation)
                    cmd = command_match.group(1).strip()
                    
                    # Check if it's a help request (don't need confirmation)
                    if cmd.lower().startswith(('help', 'man')):
                        tool_name = cmd.split()[-1]
                        if tool_name in system_commands:
                            tool_help = tool_manager.get_tool_help(tool_name)
                            if tool_help:
                                console.print(f"\n[bold cyan]Help for {tool_name}:[/bold cyan]")
                                if tool_help.get("source") == "man_page":
                                    console.print(f"[bold]Name:[/bold] {tool_help.get('name', 'N/A')}")
                                    console.print(f"[bold]Synopsis:[/bold] {tool_help.get('synopsis', 'N/A')}")
                                    console.print(f"[bold]Description:[/bold] {tool_help.get('description', 'N/A')}")
                                    if tool_help.get('options'):
                                        console.print(f"[bold]Options:[/bold] {tool_help['options']}")
                                    if tool_help.get('examples'):
                                        console.print(f"[bold]Examples:[/bold] {tool_help['examples']}")
                                else:
                                    console.print(tool_help.get('help_text', 'No help text found.'))
                            else:
                                console.print(f"[yellow]No help information available for {tool_name}[/yellow]")
                            executed_command_this_turn = True
                        else:
                            # If help is for an unknown command, ask to run it
                            confirm_and_run_command(cmd)
                    else:
                        # Ask for confirmation for non-help commands
                        confirm_and_run_command(cmd)
                
                # Save the response to chat memory in background
                def save_response_to_memory():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    chat_memory.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": timestamp,
                        "reasoning_context": reasoning_result.get("reasoning", {}),
                        "follow_up_questions": reasoning_result.get("follow_up_questions", [])
                    })
                    save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
                
                # Execute save in background to avoid blocking
                executor.submit(save_response_to_memory)
                
                # Display the response with clear formatting
                console.print(f"[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                console.print(f"[bold magenta]└─>[/bold magenta] {response}")
                console.print() # Add an empty line after output for better readability
        except KeyboardInterrupt:
            # Make sure to stop the timer if it exists in this scope
            if 'timer_running' in locals():
                timer_running = False
                if 'timer_thread' in locals():
                    timer_thread.join(timeout=0.2)
                # Display total elapsed time
                if 'start_time' in locals():
                    total_time = time.time() - start_time
                    console.print(f"⏱️ {total_time:.1f}s")
            console.print("[yellow]Processing interrupted by user[/yellow]")
        except Exception as e:
            # Make sure to stop the timer if it exists in this scope
            if 'timer_running' in locals():
                timer_running = False
                if 'timer_thread' in locals():
                    timer_thread.join(timeout=0.2)
                # Display total elapsed time
                if 'start_time' in locals():
                    total_time = time.time() - start_time
                    console.print(f"⏱️ {total_time:.1f}s")
            console.print(f"[red]Error during command processing:[/red] {str(e)}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Command interrupted. Press Ctrl+C again to exit or Enter to continue.[/yellow]")
            try:
                # Give the user a chance to exit with another Ctrl+C or continue
                if input().strip().lower() in ["exit", "quit"]:
                    save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
                    if history_enabled:
                        save_command_history()
                    console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                    break
            except KeyboardInterrupt:
                save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
                if history_enabled:
                    save_command_history()
                console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                break
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {str(e)}")
            console.print("[yellow]Continuing to next prompt...[/yellow]")
    
    # Shutdown executor pool cleanly
    executor.shutdown(wait=False)

    # Explicitly clean up resources before exit
    print("Cleaning up resources...")
    # Try closing llama object first
    if 'llm' in globals() and llm is not None:
        try:
            print("Closing Llama model...")
            llm.close() # Call the library's close method
            print("Llama model closed.")
        except Exception as e:
            print(f"Error closing Llama model: {e}")
        llm = None # Ensure reference is gone
    
    # Clean up GPU manager after Llama
    if 'gpu_manager' in globals() and gpu_manager is not None:
        try:
            print("Cleaning up GPU Manager...")
            gpu_manager.cleanup()
            print("GPU Manager cleaned up.")
        except Exception as e:
            print(f"Error cleaning up GPU Manager: {e}")
        gpu_manager = None # Ensure reference is gone

    # Optional: Force garbage collection again after explicit cleanup
    import gc
    gc.collect()
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
