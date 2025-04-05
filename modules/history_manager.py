#!/usr/bin/env python3
"""
History Manager Module for Nikita Agent

Functions for managing command history, chat memory, and user input with rich
terminal UI features.
"""

import os
import json
import readline
import sys
from pathlib import Path
from datetime import datetime

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

# Default paths
NIKITA_BASE_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model")
CHAT_HISTORY_FILE = Path(os.path.join(NIKITA_BASE_DIR, "nikita_history.json"))
COMMAND_HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "command_history")

# Command completer for readline
class CommandCompleter:
    def __init__(self, commands):
        """
        Initialize command completer for readline auto-completion.
        
        Args:
            commands (list): List of command strings to complete
        """
        self.commands = commands
    
    def complete(self, text, state):
        """
        Return the state'th completion for text.
        
        Args:
            text (str): Text to complete
            state (int): State of completion (0 for first match, etc.)
            
        Returns:
            str: Completion match or None if no more matches
        """
        # Return all matching commands
        matches = [cmd for cmd in self.commands if cmd.startswith(text)]
        try:
            return matches[state]
        except IndexError:
            return None

def load_chat_history(memory_limit=20, chat_history_file=None):
    """
    Load chat history from file.
    
    Args:
        memory_limit (int): Maximum number of messages to load
        chat_history_file (Path, optional): Path to chat history file. 
                                            Defaults to CHAT_HISTORY_FILE.
        
    Returns:
        list: Chat history as a list of message dictionaries
    """
    # Use provided chat_history_file or default
    history_file = chat_history_file or CHAT_HISTORY_FILE
    
    if isinstance(history_file, str):
        history_file = Path(history_file)
        
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                return history[-memory_limit:]
        except Exception as e:
            console.print(f"[yellow]Could not load chat history: {e}[/yellow]")
    return []

def save_chat_history(messages, chat_history_file=None):
    """
    Save chat history to file.
    
    Args:
        messages (list): List of message dictionaries to save
        chat_history_file (Path, optional): Path to chat history file.
                                            Defaults to CHAT_HISTORY_FILE.
    """
    # Use provided chat_history_file or default
    history_file = chat_history_file or CHAT_HISTORY_FILE
    
    if isinstance(history_file, str):
        history_file = Path(history_file)
        
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Could not save chat history: {e}[/yellow]")

def setup_keyboard_shortcuts():
    """
    Configure advanced keyboard shortcuts for command editing.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if sys.platform != 'win32':
        try:
            # Command history navigation
            readline.parse_and_bind(r'"\e[A": previous-history')  # Up arrow
            readline.parse_and_bind(r'"\e[B": next-history')  # Down arrow

            # Cursor movement
            readline.parse_and_bind(r'"\e[C": forward-char')  # Right arrow
            readline.parse_and_bind(r'"\e[D": backward-char')  # Left arrow

            # Word navigation
            readline.parse_and_bind(r'"\e[1;5C": forward-word')  # Ctrl+Right
            readline.parse_and_bind(r'"\e[1;5D": backward-word')  # Ctrl+Left

            # Line navigation
            readline.parse_and_bind(r'"\C-a": beginning-of-line')  # Ctrl+A
            readline.parse_and_bind(r'"\C-e": end-of-line')  # Ctrl+E

            # History search
            readline.parse_and_bind(r'"\C-r": reverse-search-history')  # Ctrl+R

            # Advanced editing
            readline.parse_and_bind(r'"\C-k": kill-line')  # Ctrl+K (delete to end)
            readline.parse_and_bind(r'"\C-u": unix-line-discard')  # Ctrl+U (delete to start)

            #console.print("🔤 [green]Keyboard shortcuts enabled[/green]")
            return True
        except Exception as e:
            console.print(f"[yellow]Keyboard shortcuts setup failed: {e}[/yellow]")

    return False

def setup_command_history(system_commands=None):
    """
    Configure readline for command history and editing.
    
    Args:
        system_commands (dict, optional): Dictionary of available commands for auto-completion
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(COMMAND_HISTORY_FILE), exist_ok=True)
        
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

        # Use the system_commands for auto-completion if provided
        if system_commands:
            completer = CommandCompleter(list(system_commands.keys()))
            readline.set_completer(completer.complete)

        # Setup advanced keyboard shortcuts
        setup_keyboard_shortcuts()

        #console.print("🔄 [cyan]Command history and editing enabled[/cyan]")
        return True
    except Exception as e:
        console.print(f"[yellow]Command history setup failed: {e}[/yellow]")
        console.print("[yellow]Continuing without command history support[/yellow]")
        return False

def save_command_history():
    """
    Save command history to file.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        readline.write_history_file(COMMAND_HISTORY_FILE)
        return True
    except Exception as e:
        console.print(f"[yellow]Could not save command history: {e}[/yellow]")
        return False

def get_input_with_history():
    """
    Get user input with readline history support and better error handling.
    
    Returns:
        str: User input string
    """
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

def add_to_chat_memory(chat_memory, role, content, memory_limit=15):
    """
    Add a message to chat memory with timestamp.
    
    Args:
        chat_memory (list): List of chat messages
        role (str): Message role ('user' or 'assistant')
        content (str): Message content
        memory_limit (int): Maximum messages to keep
        
    Returns:
        list: Updated chat memory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    chat_memory.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })
    
    # Enforce memory limit
    if len(chat_memory) > memory_limit:
        chat_memory = chat_memory[-memory_limit:]
    
    return chat_memory

if __name__ == "__main__":
    # Simple self-test
    print("History Manager Module Self-Test")
    
    # Test command history setup (manual testing only)
    print("Command history can be tested manually")
    
    # Test chat history
    test_chat = []
    test_chat = add_to_chat_memory(test_chat, "user", "Hello Nikita")
    test_chat = add_to_chat_memory(test_chat, "assistant", "Hello! How can I help you?")
    
    print("Chat memory:")
    for msg in test_chat:
        print(f"[{msg['role']}] [{msg['timestamp']}]: {msg['content']}")
    
    # Test saving and loading
    save_chat_history(test_chat)
    loaded = load_chat_history()
    
    print("\nLoaded successfully:", len(loaded) == len(test_chat)) 