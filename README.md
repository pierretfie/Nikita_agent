# Nikita Agent 🐺

Nikita is an offline AI security assistant designed to help with security tasks and system operations. It uses a local language model to provide intelligent responses, command suggestions, and security analysis without requiring internet connectivity.

## Features

- **Offline Operation**: Runs completely offline using local LLMs
- **Intelligent Command Analysis**: Analyzes and enhances security commands
- **Context Optimization**: Maintains conversation context for relevant responses
- **Code & Command Execution**: Safely runs and manages code snippets and shell commands
- **Security Focus**: Specialized for security tasks with phase-aware recommendations
- **Resource Optimization**: Automatically adapts to available system resources
- **Structured Reasoning**: Analyzes tasks with a comprehensive reasoning framework
- **Intent Analysis**: Determines the user's intention to suggest appropriate actions
- **Engagement Memory**: Maintains awareness of targets and attack progress

## Installation

### 1. Download the AI Model (Mistral-7B)

Since Nikita runs fully offline, you need to download the **Mistral-7B-Instruct** model:

```bash
# Create the model directory
mkdir -p ~/Nikita_Agent_model

# Download the model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -O ~/Nikita_Agent_model/mistral.gguf
```

- This downloads **Mistral-7B-Instruct Q4_K_M GGUF**, optimized for CPU inference
- The **Q4_K_M** quantization ensures a balance between speed and accuracy

### 2. Install Dependencies

You can install the required Python packages using pip:

```bash
# Clone the repository
git clone https://github.com/pierretfie/Nikita_agent.git
cd Nikita_agent

# Install dependencies
pip install -r requirements.txt
```

### 3. Alternative: Build `llama.cpp` from Source (Optional)

For maximum performance, you can build `llama.cpp` from source instead of using the pre-built Python package:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_AVX2=OFF -DLLAMA_AVX=ON -DLLAMA_FMA=OFF -DLLAMA_BLAS=ON -DLLAMA_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**Build Flags Explained:**
| Flag | Description |
|------|-------------|
| `-DLLAMA_AVX2=OFF` | Disables AVX2 optimizations for compatibility with older CPUs |
| `-DLLAMA_AVX=ON` | Enables basic AVX optimizations (widely supported) |
| `-DLLAMA_FMA=OFF` | Disables Fused Multiply-Add instructions (for compatibility) |
| `-DLLAMA_BLAS=ON` | Enables BLAS optimizations for matrix operations |
| `-DLLAMA_OPENBLAS=ON` | Uses OpenBLAS for performance boost |

**Processor Compatibility Note:**
Different processors support different instruction sets. For maximum compatibility:
- For older or AMD processors, use the flags as shown above with `-DLLAMA_AVX2=OFF`
- For newer Intel processors (2013+), you can enable AVX2 with `-DLLAMA_AVX2=ON` for better performance
- For very old systems, you may need to use `-DLLAMA_AVX=OFF` as well

After building, install the Python bindings:

```bash
pip install --no-cache-dir --force-reinstall llama-cpp-python
```

## Usage

Run the agent:
```
python3 Nikita_agent.py
```

Example interactions:
- `scan my network` - Performs a network scan with optimized parameters
- `explain buffer overflow` - Provides security knowledge
- `help me with nmap` - Gives command suggestions and explanations

## Project Structure

The project is organized into two main parts:

### 1. Agent Code - `/path/to/Nikita_agent/`

```
Nikita_agent/
├── Nikita_agent.py       # Main entry point and application logic
├── modules/              # Modular components
│   ├── __init__.py       # Package initialization
│   ├── code_handler.py   # Code detection and execution
│   ├── command_handler.py # Command processing and optimization
│   ├── context_optimizer.py # Conversation context management
│   ├── engagement_manager.py # Target and attack phase tracking
│   ├── fine_tuning.json    # Training data for command suggestions
│   ├── history_manager.py # History and memory management
│   ├── intent_analyzer.py # Query intent analysis
│   ├── prompt_template.txt # System prompts and instructions
│   ├── reasoning_engine.py # Structured reasoning framework
│   ├── README.md         # Modules documentation
│   └── resource_management.py # System resource optimization
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

### 2. Runtime Data - `~/Nikita_Agent_model/`

```
~/Nikita_Agent_model/
├── mistral.gguf              # The language model file
├── outputs/                  # Directory for command output
│   ├── cmd_20240126_120000.txt
│   ├── nmap_scan_20240126_120005.txt
│   └── ... (other output files)
├── nikita_history.json       # Chat history
└── command_history           # Command history from readline
```

The agent code can be located anywhere on your system, while the runtime data is stored in your home directory for easier access and management.

## Configuration

Nikita is designed to work with minimal configuration. The key paths are:

```python
# Base directory for Nikita data (in your home directory)
NIKITA_BASE_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model")

# Model path
MODEL_PATH = os.path.join(NIKITA_BASE_DIR, "mistral.gguf")

# Command outputs directory
OUTPUT_DIR = os.path.join(NIKITA_BASE_DIR, "outputs")

# History files
CHAT_HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "nikita_history.json")
COMMAND_HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "command_history")
```

By default, the agent:

- Stores the model in `~/Nikita_Agent_model/mistral.gguf`
- Saves command outputs to `~/Nikita_Agent_model/outputs/`
- Maintains conversation history in `~/Nikita_Agent_model/nikita_history.json`
- Preserves command history in `~/Nikita_Agent_model/command_history`
- Automatically optimizes for your system's resources

These paths can be customized by editing the constants at the beginning of `Nikita_agent.py`.

## Modular Design

The agent is built with a modular architecture for maintainability:

- **Intent Analysis**: Determines user intent from queries
- **Resource Management**: Optimizes system resource usage
- **Command Handling**: Processes and enhances system commands
- **Code Handling**: Detects and safely executes code snippets
- **History Management**: Maintains conversation and command history
- **Context Optimization**: Improves responses through context
- **Reasoning Engine**: Provides structured analysis of security tasks
- **Engagement Manager**: Tracks targets and attack progression

## Key Components

### Reasoning Engine

The reasoning engine provides a structured framework for analyzing security tasks:

1. **Task Understanding**: Goals, context, and constraints
2. **Planning**: Required steps, dependencies, and execution order
3. **Tool Selection**: Primary and alternative tools with parameters
4. **Safety Analysis**: Risks, precautions, and fallback plans
5. **Execution Planning**: Command formulation with explanations
6. **Output Analysis**: Expected output, success indicators, and next steps

The engine categorizes tasks into domains like Reconnaissance, Web, Wireless, and Password tasks, adapting its reasoning accordingly.

### Resource Management

The agent automatically optimizes performance based on your system:

- **Memory Tier Detection**: Configures parameters based on available RAM
- **CPU Optimization**: Sets thread affinity based on current system load
- **Model Prewarming**: Reduces initial response latency
- **Dynamic Scaling**: Adjusts batch size and context limits for optimal performance

## Troubleshooting

### Missing `llama-cpp-python`
If you see:
```bash
ModuleNotFoundError: No module named 'llama_cpp'
```
Run:
```bash
pip install --no-cache-dir --force-reinstall llama-cpp-python
```

### `llama.cpp` Compilation Errors
Ensure dependencies are installed:
```bash
sudo apt update && sudo apt install -y build-essential cmake libopenblas-dev
```

### Processor Compatibility Issues
If you encounter errors related to unsupported CPU instructions:

1. Try rebuilding llama.cpp with more conservative flags:
```bash
cmake .. -DLLAMA_AVX2=OFF -DLLAMA_AVX=OFF -DLLAMA_FMA=OFF -DLLAMA_BLAS=ON
```

2. Or use a pre-built binary that's compatible with your CPU:
```bash
pip install --no-cache-dir --force-reinstall llama-cpp-python --no-binary llama-cpp-python
```

3. For ARM processors (like Apple M1/M2), use:
```bash
CMAKE_ARGS="-DLLAMA_METAL=ON" pip install --no-cache-dir --force-reinstall llama-cpp-python
```

### Slow Performance
- Use `htop` to check CPU usage
- Adjust the `n_threads` parameter for better parallelization
- Consider using a different quantization level (Q5_K_M for higher quality, Q3_K_M for faster performance)

## Security Considerations

- Runs fully offline (no internet dependency)
- Only executes commands after user confirmation
- Incorporates safeguards against malformed commands
- Uses reasoning engine to evaluate security implications

## Development

To extend the agent's capabilities:

1. Add new modules to the `modules/` directory
2. Update imports in `Nikita_agent.py`
3. Extend the relevant classes or functions

## Dependencies

- `llama_cpp_python`: LLM inference
- `rich`: Terminal UI and formatting
- `psutil`: System resource monitoring
- Additional dependencies in `requirements.txt`


---

*Nikita - Stay frosty.*