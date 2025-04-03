#!/usr/bin/env python3
"""
Tool Manager Module for Nikita Agent

Handles tool-related functionality including man pages, help information,
and tool context management.
"""

import subprocess
import re
import json
from pathlib import Path
from rich.console import Console

console = Console()

class ToolManager:
    def __init__(self, fine_tuning_file=None):
        """
        Initialize the tool manager.
        
        Args:
            fine_tuning_file (str, optional): Path to fine-tuning data file
        """
        self.fine_tuning_file = fine_tuning_file
        self.tool_cache = {}  # Cache for tool information
        self.common_usage = {
            "nmap": {
                "basic_scan": "nmap -sV -sC <target>",
                "stealth_scan": "nmap -sS -T2 <target>",
                "os_detection": "nmap -O <target>",
                "service_scan": "nmap -sV <target>"
            },
            "metasploit": {
                "search": "msfconsole -q -x 'search <exploit>'",
                "exploit": "msfconsole -q -x 'use <exploit>; set RHOSTS <target>; exploit'"
            },
            "hydra": {
                "ssh_brute": "hydra -l <user> -P <wordlist> ssh://<target>",
                "http_post": "hydra -l <user> -P <wordlist> <target> http-post-form"
            }
        }

    def get_tool_manpage(self, tool_name):
        """Fetch and parse man page for a security tool"""
        try:
            # Run man command and capture output
            result = subprocess.run(['man', tool_name], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch man page for {tool_name}: {str(e)}[/yellow]")
            return None

    def parse_manpage(self, manpage_content):
        """Parse man page content to extract useful information"""
        if not manpage_content:
            return None
        
        # Extract common sections
        sections = {
            "name": re.search(r"NAME\n\s*(.*?)\n", manpage_content),
            "synopsis": re.search(r"SYNOPSIS\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL),
            "description": re.search(r"DESCRIPTION\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL),
            "options": re.search(r"OPTIONS\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL),
            "examples": re.search(r"EXAMPLES\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL)
        }
        
        parsed = {}
        for section, match in sections.items():
            if match:
                parsed[section] = match.group(1).strip()
        
        return parsed

    def get_tool_help(self, tool_name):
        """Get help information for a security tool"""
        # Check cache first
        if tool_name in self.tool_cache:
            return self.tool_cache[tool_name]
        
        # First try to get man page
        manpage = self.get_tool_manpage(tool_name)
        if manpage:
            parsed = self.parse_manpage(manpage)
            if parsed:
                help_info = {
                    "source": "man_page",
                    "name": parsed.get("name", ""),
                    "synopsis": parsed.get("synopsis", ""),
                    "description": parsed.get("description", ""),
                    "options": parsed.get("options", ""),
                    "examples": parsed.get("examples", "")
                }
                self.tool_cache[tool_name] = help_info
                return help_info
        
        # Fallback to --help if man page not available
        try:
            result = subprocess.run([tool_name, '--help'], capture_output=True, text=True)
            if result.returncode == 0:
                help_info = {
                    "source": "help_flag",
                    "help_text": result.stdout
                }
                self.tool_cache[tool_name] = help_info
                return help_info
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get help for {tool_name}: {str(e)}[/yellow]")
        
        return None

    def get_tool_context(self, tool_name):
        """Get comprehensive context for a security tool"""
        context = {
            "man_page": None,
            "fine_tuning": None,
            "common_usage": None
        }
        
        # Get man page information
        tool_help = self.get_tool_help(tool_name)
        if tool_help:
            context["man_page"] = tool_help
        
        # Get fine-tuning data if file is available
        if self.fine_tuning_file and Path(self.fine_tuning_file).exists():
            try:
                with open(self.fine_tuning_file, "r") as f:
                    fine_tuning_data = json.load(f)
                    tool_data = [entry for entry in fine_tuning_data if entry.get("tool_used") == tool_name]
                    if tool_data:
                        context["fine_tuning"] = tool_data
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load fine-tuning data for {tool_name}: {str(e)}[/yellow]")
        
        # Get common usage patterns
        if tool_name in self.common_usage:
            context["common_usage"] = self.common_usage[tool_name]
        
        return context

    def clear_cache(self):
        """Clear the tool information cache"""
        self.tool_cache.clear()

if __name__ == "__main__":
    # Simple self-test
    tool_manager = ToolManager()
    
    # Test with nmap
    nmap_context = tool_manager.get_tool_context("nmap")
    print("Nmap Context:")
    print(json.dumps(nmap_context, indent=2)) 