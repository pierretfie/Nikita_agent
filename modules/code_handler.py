#!/usr/bin/env python3
"""
Code Handler Module for Nikita Agent

Functions for detecting, running, and saving code snippets safely.
"""

import os
import re
import tempfile
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Try to import rich for pretty output if available
try:
    from rich.console import Console
    console = Console()
except ImportError:
    # Fallback to simple print if rich is not available
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = FallbackConsole()

# Default output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_python_code(text):
    """
    Detect if a string contains Python code.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        bool: True if the text appears to be Python code, False otherwise
    """
    # Look for Python keywords and patterns
    python_patterns = [
        r'^\s*def\s+\w+\s*\(',  # Function definitions
        r'^\s*class\s+\w+',     # Class definitions
        r'^\s*import\s+\w+',    # Import statements
        r'^\s*from\s+\w+\s+import', # From imports
        r'^\s*for\s+\w+\s+in\s+', # For loops
        r'^\s*if\s+.+:',        # If statements
        r'^\s*while\s+.+:',     # While loops
        r'^\s*try:',            # Try blocks
        r'^\s*except',          # Except blocks
        r'^\s*with\s+.+:',      # With statements
    ]
    
    lines = text.strip().split('\n')
    
    # Check for indentation (a Python characteristic)
    has_indentation = any(line.startswith('    ') or line.startswith('\t') for line in lines)
    
    # Count pattern matches
    pattern_matches = 0
    for pattern in python_patterns:
        for line in lines:
            if re.match(pattern, line):
                pattern_matches += 1
                break
    
    # Heuristic: if multiple patterns match or we see indentation and at least one pattern,
    # it's likely Python code
    return pattern_matches >= 2 or (has_indentation and pattern_matches >= 1)

def run_python_code(code, save_output=True):
    """
    Run Python code in a controlled environment and capture the output.
    
    Args:
        code (str): The Python code to execute
        save_output (bool): Whether to save the output to a file
        
    Returns:
        tuple: (output, error_message, output_file_path)
    """
    # Create a temporary file to run the code
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code)
    
    try:
        # Run the Python code with a timeout
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        output = result.stdout
        error = result.stderr
        
        # Save output if requested
        output_file_path = None
        if save_output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file_path = os.path.join(OUTPUT_DIR, f"python_code_{timestamp}.txt")
            
            with open(output_file_path, 'w') as f:
                f.write(f"=== Python Code ===\n{code}\n\n")
                f.write(f"=== Output ===\n{output}\n")
                if error:
                    f.write(f"\n=== Errors ===\n{error}\n")
        
        return output, error, output_file_path
        
    except subprocess.TimeoutExpired:
        return "", "Code execution timed out after 30 seconds", None
    except Exception as e:
        return "", f"Error running code: {str(e)}", None
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def extract_code_from_text(text):
    """
    Extract code blocks from text (markdown-style).
    
    Args:
        text (str): Text that may contain code blocks
        
    Returns:
        list: List of extracted code blocks
    """
    # Look for triple backtick code blocks
    code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', text)
    
    # If none found, try single backtick blocks
    if not code_blocks:
        code_blocks = re.findall(r'`(.*?)`', text)
    
    return code_blocks

if __name__ == "__main__":
    # Simple self-test
    test_code = """
def hello():
    print("Hello, world!")
    
hello()
"""
    
    print(f"Is Python code: {is_python_code(test_code)}")
    output, error, path = run_python_code(test_code, save_output=False)
    print(f"Output: {output}")
    print(f"Error: {error}") 