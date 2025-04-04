# Nikita Agent Modules

This directory contains the core modules for the Nikita Agent system.

## Core Modules

### Tool Manager (`tool_manager.py`)
Handles tool-related functionality including:
- Man page fetching and parsing
- Tool help information retrieval
- Tool context management
- Common usage patterns
- Caching for better performance

### Context Optimizer (`context_optimizer.py`)
Optimizes conversation context for LLM interactions by:
- Selecting relevant messages
- Handling token limits
- Improving prompt quality
- Managing context window

### Reasoning Engine (`reasoning_engine.py`)
Provides advanced reasoning capabilities:
- Task analysis
- Context understanding
- Follow-up question generation
- Security-focused reasoning

### Intent Analyzer (`intent_analyzer.py`)
Analyzes user intent and commands:
- Command detection
- Intent classification
- Security tool recognition
- Command validation

### Resource Management (`resource_management.py`)
Manages system resources:
- Memory optimization
- CPU usage control
- Model prewarming
- System parameter tuning

### History Manager (`history_manager.py`)
Manages conversation and command history:
- Command history tracking
- Chat history management
- History persistence
- Memory optimization

### Engagement Manager (`engagement_manager.py`)
Handles engagement-related functionality:
- Target tracking
- Attack planning
- Finding recording
- Engagement summaries

### Command Handler (`command_handler.py`)
Handles command execution:
- Command validation
- Safe execution
- Output capture
- Error handling

### Code Handler (`code_handler.py`)
Manages code-related operations:
- Code analysis
- Security checks
- Code execution
- Output formatting

### GPU Manager (`gpu_manager.py`)
Handles GPU resource management and configuration:
- GPU device detection and initialization
- Parallel processing setup (e.g., OpenCL)
- Resource allocation and cleanup

## Data Files

- `emotional_patterns.json`: Emotional analysis patterns
- `reasoning_datasets.json`: Reasoning engine datasets
- `human_like_patterns.json`: Human-like response patterns
- `fine_tuning.json`: Fine-tuning data for the model
- `prompt_template.txt`: Base prompt template

## Usage

Each module is designed to work independently while integrating seamlessly with the main Nikita Agent system. The modules can be imported and used as needed:

```python
from modules import ToolManager, ContextOptimizer, ReasoningEngine

# Initialize modules
tool_manager = ToolManager(fine_tuning_file="path/to/fine_tuning.json")
context_optimizer = ContextOptimizer()
reasoning_engine = ReasoningEngine()

# Use modules
tool_context = tool_manager.get_tool_context("nmap")
optimized_prompt = context_optimizer.get_optimized_prompt(...)
reasoning_result = reasoning_engine.analyze_task(...)
```

## Key Components

### Reasoning Engine

The reasoning engine (`reasoning_engine.py`) provides a structured framework for analyzing security tasks with these components:

1. A comprehensive reasoning template
2. Task categorization (Recon, Web, Wireless, Password)
3. Goal and constraint identification
4. Tool selection and parameter analysis
5. Risk and precaution assessment

The engine helps Nikita generate more thoughtful and security-aware responses by considering the full context of user requests.

### Engagement Manager

The engagement manager (`engagement_manager.py`) maintains awareness of the security engagement context:

1. Tracking target systems and networks
2. Maintaining a record of discovered credentials and information
3. Suggesting appropriate attack paths based on current knowledge
4. Providing phase-aware recommendations (reconnaissance → exploitation → post-exploitation)

This module helps maintain continuity across multiple interactions during security assessments.

## Extending Modules

To add new functionality:

1. Add new functions to the appropriate module based on function category
2. For entirely new categories of functionality, create a new module file
3. Update `__init__.py` if needed to expose the new functionality 

## GPU Setup for Google Colab

When running Nikita Agent in Google Colab, you'll need to properly set up the CUDA-enabled version of llama-cpp-python to utilize the GPU. Follow these steps:

```bash
# Uninstall current llama-cpp-python
pip uninstall -y llama-cpp-python

# Install CUDA dependencies
!apt-get update && apt-get install -y libcublas-dev-11-8 cuda-nvcc-11-8

# Install llama-cpp-python with CUDA support
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.6 --no-cache-dir

# Alternative installation command if the above fails
# !pip install llama-cpp-python-cu11==0.2.6 --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

After running these commands, restart your Colab kernel before running the Nikita agent code. This ensures the GPU is properly utilized for model inference, significantly improving performance. 