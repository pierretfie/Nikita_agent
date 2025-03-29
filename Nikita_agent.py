#!/usr/bin/env python3

from llama_cpp import Llama
import subprocess
import shlex
import psutil
import readline
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt

console = Console()

# Load Llama model (fine-tuned for low-RAM systems)
llm = Llama(
    model_path="/home/eclipse/Nikita_Agent_model/mistral.gguf",
    n_ctx=768,        # Optimal context size for your RAM
    n_threads=4,      # Match your CPU threads
    n_batch=64,       # Reduce RAM usage, increase efficiency
    use_mlock=True,   # Prevents model swap-out
    low_vram=True,    # Enables low-memory optimizations
    embedding=False,  # Speeds up inference
    verbose=False     # Suppress unneeded debug info
)

# Updated Prompt Template
PROMPT_TEMPLATE = """
You are Nikita, an AI Offensive Security Agent in Kali Linux.
Your job is to assist pentesters by generating Linux security commands.

🔹 Rules:
- Use ONLY these tools: nmap, gobuster, amass, smbclient, metasploit, sqlmap, whois, dig.
- Output ONLY the final command (no extra words or explanations).
- Never output partial/incomplete commands.

Examples:
User: Scan 192.168.0.1 for open ports.
AI: nmap -sS -p- 192.168.0.1

User: Find subdomains of example.com
AI: amass enum -d example.com

User: {}
AI:
"""

def check_swap_usage():
    """Detects excessive swap usage & suggests fixes."""
    swap = psutil.swap_memory()
    if swap.used > (600 * 1024 * 1024):  # If swap > 600MB, warn user
        console.print(f"[bold red]⚠ WARNING: Swap usage is high ({swap.used // (1024*1024)} MiB).[/bold red]")
        console.print("[yellow]Try closing apps or running:[/yellow] [cyan]sudo swapoff -a && sudo swapon -a[/cyan]")
        
        
def query_ai(task):
    """Query AI and smartly extract a usable command if present."""
    full_prompt = PROMPT_TEMPLATE.format(task)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        transient=True
    ) as progress:
        task_gen = progress.add_task("[cyan]Generating...", total=100)
        try:
            output = llm(full_prompt, max_tokens=100, stream=False)
            response = output['choices'][0]['text'].strip()
            for _ in range(100):
                progress.update(task_gen, advance=1)

        except Exception as e:
            console.print(f"[red]⚠ AI Error:[/red] {e}")
            return None

    console.print(f"[bold magenta]Nikita:[/bold magenta] {response}")

    # ✅ Soft-filter: Grab first valid command inside the response
    for line in response.splitlines():
        for tool in ["nmap", "gobuster", "amass", "smbclient", "metasploit", "sqlmap", "whois", "dig"]:
            if line.strip().startswith(tool):
                return line.strip()

    # If no valid command was detected, fallback
    return response


def run_command(cmd):
    """Execute the generated command in a subprocess."""
    try:
        console.print(f"[bold cyan]▶ Running: {cmd}[/bold cyan]")
        cmd_list = shlex.split(cmd)
        result = subprocess.run(cmd_list, capture_output=True, text=True)

        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
    except Exception as e:
        console.print(f"[red]⚠ Error:[/red] {e}")
        
        
def main():
    console.print("[bold green]KaliAI - Nikita (Operator-Aware Conversational Mode) Activated[/bold green]")
    while True:
        user_input = input("Sudo> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        check_swap_usage()
        cmd = query_ai(user_input)
        
        # 🟣 Only auto-run if a valid command is detected
        if any(cmd.startswith(tool) for tool in ["nmap", "gobuster", "amass", "smbclient", "metasploit", "sqlmap", "whois", "dig"]):
            run_command(cmd)
        else:
            console.print(f"[yellow]Operator Notice:[/yellow] {cmd}")


if __name__ == "__main__":
    main()
