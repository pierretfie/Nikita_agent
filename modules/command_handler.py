#!/usr/bin/env python3
"""
Command Handler Module for Nikita Agent

Functions for running system commands, hardening commands with security best practices,
and saving command outputs.
"""

import os
import subprocess
import shlex
import re
from datetime import datetime
import psutil
import json
from typing import Dict, List, Tuple, Optional

# Try to import rich for pretty output if available
try:
    from rich.console import Console
    from rich.prompt import Confirm
    console = Console()
except ImportError:
    # Fallback to simple print if rich is not available
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
        def confirm(self, prompt: str) -> bool:
            response = input(f"{prompt} (y/n): ").lower()
            return response in ['y', 'yes']
    console = FallbackConsole()

# Default output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Command risk levels and categories
RISK_LEVELS = {
    "LOW": 1,      # Basic system info, non-destructive
    "MEDIUM": 2,   # Network scanning, file operations
    "HIGH": 3,     # System modifications, security tools
    "CRITICAL": 4  # Potentially dangerous operations
}

COMMAND_CATEGORIES = {
    "system_info": {
        "risk_level": "LOW",
        "commands": ["uname", "ls", "ps", "df", "free", "top", "uptime"]
    },
    "network_scan": {
        "risk_level": "MEDIUM",
        "commands": ["nmap", "ping", "traceroute", "netstat", "ss"]
    },
    "security_tools": {
        "risk_level": "HIGH",
        "commands": ["sqlmap", "gobuster", "hashcat", "john", "hydra"]
    },
    "system_modification": {
        "risk_level": "CRITICAL",
        "commands": ["rm", "mv", "cp", "chmod", "chown", "useradd", "usermod"]
    }
}

def get_command_risk_level(cmd: str) -> Tuple[str, str]:
    """
    Determine the risk level and category of a command.
    
    Args:
        cmd (str): The command to analyze
        
    Returns:
        tuple: (risk_level, category)
    """
    cmd_base = cmd.split()[0].lower()
    
    # Check command categories
    for category, info in COMMAND_CATEGORIES.items():
        if cmd_base in info["commands"]:
            return info["risk_level"], category
            
    # Default to MEDIUM risk if unknown
    return "MEDIUM", "unknown"

def requires_confirmation(cmd: str) -> bool:
    """
    Check if a command requires user confirmation based on its risk level.
    
    Args:
        cmd (str): The command to check
        
    Returns:
        bool: True if confirmation is required
    """
    risk_level, _ = get_command_risk_level(cmd)
    return RISK_LEVELS[risk_level] >= RISK_LEVELS["MEDIUM"]

def get_confirmation_message(cmd: str, risk_level: str, category: str) -> str:
    """
    Generate a confirmation message for a command.
    
    Args:
        cmd (str): The command to confirm
        risk_level (str): The risk level of the command
        category (str): The category of the command
        
    Returns:
        str: Confirmation message
    """
    return f"""⚠️  Command requires confirmation:
Command: {cmd}
Risk Level: {risk_level}
Category: {category}

Type 'yes' to proceed or 'no' to cancel."""

def harden_command(cmd):
    """
    Enhance commands with smart defaults and output paths for better security and logging.
    
    Args:
        cmd (str): The command to harden
        
    Returns:
        str: The hardened command with additional safety parameters
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Tool-specific enhancements
    enhancements = {
        'nmap': [
            ('-sV', ' -sV'),  # Version detection
            ('-sC', ' -sC'),  # Default scripts
            ('-sn', ' -sn'),  # Ping scan
            ('-T4', ' -T4'),  # Timing template
            ('--stats-every', ' --stats-every 10s')  # Progress updates
        ],
        'sqlmap': [
            ('--batch', ' --batch'),  # Non-interactive mode
            ('--random-agent', ' --random-agent')  # Random user agent
        ],
        'gobuster': [
            ('-q', ' -q')  # Quiet mode
        ],
        'smbclient': [
            ('-N', '-N')  # No password prompt
        ],
        'dig': [
            ('+short', ' +short')  # Short output
        ],
        'hashcat': [
            ('--identify', ' --identify'),  # Hash type identification
            ('--quiet', ' --quiet'),  # Less verbose output
            ('--show', ' --show')  # Show cracked passwords
        ]
    }

    # Add output paths for tools that support it
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

    # Fix common IP range syntax
    ip_range_match = re.search(r'(\d{1,3}\.){3}\d{1,3}(?:-\d{1,3})?', cmd)
    if ip_range_match:
        ip_range = ip_range_match.group(0)
        if '-' in ip_range:
            base_ip = ip_range.split('-')[0]
            cmd = cmd.replace(ip_range, base_ip + '/24')

    # Apply tool-specific enhancements
    for tool, rules in enhancements.items():
        if cmd.startswith(tool):
            for check, add in rules:
                if check not in cmd:
                    cmd += add

    return cmd

def save_command_output(cmd, output, error=None):
    """
    Save command output to a file for later reference.
    
    Args:
        cmd (str): The command that was executed
        output (str): Command stdout
        error (str, optional): Command stderr
        
    Returns:
        str: Path to the output file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"cmd_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"=== Command ===\n{cmd}\n\n")
        f.write(f"=== Output ===\n{output}\n")
        if error:
            f.write(f"\n=== Errors ===\n{error}\n")

    return output_file

def run_command(cmd: str, timeout: int = 300, harden: bool = True) -> Tuple[bool, str, str, Optional[str]]:
    """
    Run a system command with enhanced safety and output handling.
    
    Args:
        cmd (str): Command to run
        timeout (int): Command timeout in seconds
        harden (bool): Whether to apply hardening to the command
        
    Returns:
        tuple: (success, output, error, output_file_path)
    """
    # Validate command
    if cmd.count('.') < 3 and " " in cmd and any(x in cmd for x in ["nmap", "scan", "ping"]):
        console.print(f"❌ [red]Command incomplete:[/red] {cmd}")
        return False, "", "Command appears incomplete", None

    # Check command safety
    is_safe, reason = is_command_safe(cmd)
    if not is_safe:
        console.print(f"❌ [red]Command blocked:[/red] {reason}")
        return False, "", reason, None

    # Get command risk level and category
    risk_level, category = get_command_risk_level(cmd)
    
    # Check if confirmation is required
    if requires_confirmation(cmd):
        confirmation_msg = get_confirmation_message(cmd, risk_level, category)
        console.print(f"\n[bold yellow]{confirmation_msg}[/bold yellow]")
        
        while True:
            response = input("> ").lower().strip()
            if response == 'yes':
                console.print("[green]✓ Command approved[/green]")
                break
            elif response == 'no':
                console.print("❌ [yellow]Command execution cancelled by user[/yellow]")
                return False, "", "Command cancelled by user", None
            else:
                console.print("[red]Invalid input. Please type 'yes' or 'no'[/red]")

    try:
        # Harden and normalize command if requested
        if harden:
            cmd = harden_command(cmd)
        
        console.print(f"⚡ [bold cyan]Running:[/bold cyan] {cmd}")

        # Split command for subprocess
        cmd_list = shlex.split(cmd)
        
        # Execute the command with timeout
        result = subprocess.run(
            cmd_list, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )

        # Save output to file
        output_file = save_command_output(cmd, result.stdout, result.stderr)

        # Format and display output
        if result.stdout.strip():
            # Special formatting for nmap output
            if cmd.startswith("nmap"):
                output_lines = result.stdout.strip().split("\n")
                formatted_output = "\n".join(
                    f"  {line}" if "open" in line else line 
                    for line in output_lines
                )
                console.print(f"🖥️  [green]{formatted_output}[/green]")
            else:
                console.print(f"🖥️  [green]{result.stdout.strip()}[/green]")
            
            console.print(f"📝 [cyan]Output saved to:[/cyan] {output_file}")
        else:
            console.print(f"⚠️  [yellow]No output returned.[/yellow]")

        return True, result.stdout, result.stderr, output_file

    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout} seconds"
        console.print(f"❌ [red]{error_msg}[/red]")
        save_command_output(cmd, "", error_msg)
        return False, "", error_msg, None
    
    except Exception as e:
        error_msg = str(e)
        console.print(f"❌ [red]Error:[/red] {error_msg}")
        save_command_output(cmd, "", error_msg)
        return False, "", error_msg, None

def is_command_safe(cmd):
    """
    Basic safety check for commands to prevent dangerous operations.
    
    Args:
        cmd (str): Command to check
        
    Returns:
        tuple: (is_safe, reason)
    """
    # List of potentially dangerous commands
    dangerous_patterns = [
        r'\brm\s+-rf\b',           # rm -rf
        r'\brm\s+.*--no-preserve-root\b',  # rm with no-preserve-root
        r'\bmkfs\b',                # mkfs
        r'\bdd\s+if=.*of=/dev/',    # dd to device
        r'\bformat\b',              # format
        r'>\s*/dev/sd[a-z]',        # redirect to block device
        r'>\s*/dev/null.*2>&1\s*&\s*disown',  # background cmd hiding output
    ]
    
    # Check all dangerous patterns
    for pattern in dangerous_patterns:
        if re.search(pattern, cmd, re.IGNORECASE):
            return False, f"Command matches dangerous pattern: {pattern}"
    
    return True, "Command appears safe"

if __name__ == "__main__":
    # Simple self-test
    print("Command Handler Module Self-Test")
    
    test_cmd = "echo 'Hello, Nikita!'"
    hardened = harden_command(test_cmd)
    print(f"Original: {test_cmd}")
    print(f"Hardened: {hardened}")
    
    success, output, error, path = run_command(test_cmd, harden=False)
    print(f"Success: {success}")
    print(f"Output: {output}")
    print(f"Error: {error}")
    print(f"Output Path: {path}") 