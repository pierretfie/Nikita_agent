import os
import time
import re
import random
import psutil
import json
import subprocess
import shlex
import socket
import getpass
from llama_cpp import Llama
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from datetime import datetime

# ===============================
# === CONFIG ====================
# ===============================

MODEL_PATH ="/home/eclipse/Nikita_Agent_model/mistral.gguf"  # Fixed absolute path
MAX_TOKENS = 512
TEMPERATURE = 0.7
OUTPUT_DIR = "/home/eclipse/Nikita_Agent_model/outputs"
HISTORY_FILE = "/home/eclipse/Nikita_Agent_model/history.json"

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

# ===============================
# === SYSTEM INFO ===============
# ===============================

def get_system_info():
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_count = os.cpu_count()
    ram_gb = ram.total / (1024 ** 3)
    
    # Get hostname and username
    hostname = socket.gethostname()
    username = getpass.getuser()
    
    # Get IP addresses
    ip_addresses = []
    try:
        for interface, snics in psutil.net_if_addrs().items():
            for snic in snics:
                if snic.family == socket.AF_INET:
                    ip_addresses.append(snic.address)
    except:
        ip_addresses = ["Unable to retrieve IP"]

    # Update memory with system info
    memory["system_info"] = {
        "hostname": hostname,
        "username": username,
        "ip_addresses": ip_addresses,
        "ram_gb": ram_gb,
        "cpu_count": cpu_count
    }
    
    return ram, swap, cpu_count, ram_gb

# ===============================
# === DYNAMIC PROMPT ============
# ===============================

def get_dynamic_prompt():
    ram, swap, cpu_count, ram_gb = get_system_info()
    system_info = memory["system_info"]
    
    return f"""
You are Nikita 🐺 — an advanced Offensive Security AI Assistant with dry humor and Red Team operator attitude.

=== System Info ===
- Hostname: {system_info['hostname']}
- Username: {system_info['username']}
- IP Addresses: {', '.join(system_info['ip_addresses'])}
- RAM: {ram_gb:.2f} GB
- CPU Cores: {cpu_count}

=== Personality ===
- Blunt, sarcastic, but loyal to the operator.
- Drops subtle jokes or remarks every now and then.
- Prioritizes precision, elegance, and OPSEC.

=== Mission ===
- Reconnaissance
- Exploitation
- Post-Exploitation
- Pivoting
- Red Team Adversary Simulation

=== Operator Notes ===
1. Be professional but fun.
2. Prefer chained, efficient commands.
3. When appropriate, comment on the operator's choice.
4. Provide suggestions, don't hold back.
5. NEVER include prefixes like 'RESPONSE:', 'OUTPUT:', etc.
6. Keep responses short and direct.
7. For commands, give ONLY the command.
"""

# ===============================
# === TARGET EXTRACTOR ==========
# ===============================

def extract_targets(task):
    targets = re.findall(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', task)
    for t in targets:
        if t not in memory["targets"]:
            memory["targets"].append(t)
    return targets

# ===============================
# === ATTACK PLAN GENERATOR =====
# ===============================

def suggest_attack_plan(task):
    if "recon" in task.lower():
        return "Recommended: nmap -sC -sV -oA recon_scan <target> | tee recon_scan.txt | grep open"
    if "priv esc" in task.lower():
        return "Recommended: wget https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh -O linpeas.sh && chmod +x linpeas.sh && ./linpeas.sh | tee linpeas.txt"
    if "pivot" in task.lower():
        return "Recommended: ssh -D 9050 user@target -f -C -q -N (SOCKS Proxy Pivot)"
    if "web exploit" in task.lower():
        return "Recommended: ffuf -u http://<target>/FUZZ -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt"
    return ""

# ===============================
# === SELF-HEALING COMMAND ======
# ===============================

def harden_command(cmd):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if cmd.startswith("nmap"):
        output_base = os.path.join(OUTPUT_DIR, f"nmap_scan_{timestamp}")
        if "-sV" not in cmd:
            cmd += " -sV"
        if "--stats-every" not in cmd:
            cmd += " --stats-every 10s"
        if "-oN" not in cmd:
            cmd += f" -oN {output_base}.txt"
        if "-oX" not in cmd:
            cmd += f" -oX {output_base}.xml"
    
    if cmd.startswith("sqlmap"):
        if "--batch" not in cmd:
            cmd += " --batch"
        if "--output-dir" not in cmd:
            output_dir = os.path.join(OUTPUT_DIR, f"sqlmap_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            cmd += f" --output-dir={output_dir}"
    
    if cmd.startswith("gobuster"):
        if "-o" not in cmd:
            cmd += f" -o {os.path.join(OUTPUT_DIR, f'gobuster_{timestamp}.txt')}"
    
    if cmd.startswith("smbclient") and "-N" not in cmd:
        cmd = cmd.replace("smbclient", "smbclient -N")
    
    if cmd.startswith("dig") and "+short" not in cmd:
        cmd += " +short"
        
    # Convert IP ranges to CIDR
    ip_range_match = re.search(r'(\d{1,3}\.){3}\d{1,3}-\d{1,3}', cmd)
    if ip_range_match:
        ip_range = ip_range_match.group(0)
        base_ip = ip_range.split('-')[0]
        cmd = cmd.replace(ip_range, base_ip + '/24')
    
    return cmd

# ===============================
# === BANTER INJECTOR ===========
# ===============================

def inject_banter(response_text):
    banters = [
        "Stay frosty.",
        "Classic move, operator.",
        "This ain't my first rodeo.",
        "Slick... real slick.",
        "Another day, another target.",
        "You know the drill."
    ]
    if random.random() < 0.4:  # 40% chance
        response_text += f"\n# {random.choice(banters)}"
    return response_text

# ===============================
# === LLM SETUP =================
# ===============================

def setup_llm():
    ram, swap, cpu_count, _ = get_system_info()
    
    # More aggressive optimization parameters
    if swap.used > (512 * 1024 * 1024):
        n_ctx = 1024  # Increased from 512
        n_batch = 64   # Increased minimum batch size
        n_threads = max(1, cpu_count - 1)
        print("🟡 Low RAM Mode: Optimizing for memory")
    else:
        n_ctx = 2048
        n_batch = min(512, cpu_count * 32)  # Increased batch size
        n_threads = cpu_count
        print("🟢 Normal Mode: Optimizing for performance")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        f16_kv=True,
        verbose=False
    )
    
    # Prewarm with system info to cache common tokens
    llm(f"System info: {memory['system_info']}", max_tokens=1)
    return llm

# ===============================
# === MAIN LOOP =================
# ===============================

console = Console()

def main():
    global chat_memory
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load history if exists
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                chat_memory = json.load(f)
        except:
            chat_memory = []

    try:
        llm = setup_llm()
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        return

    # Print version banner
    console.print("\n[bold red]╔══════════════════════════════════════╗[/bold red]")
    console.print("[bold red]║[/bold red]     [bold white]NIKITA AI AGENT v2.0[/bold white]      [bold red]║[/bold red]")
    console.print("[bold red]╚══════════════════════════════════════╝[/bold red]\n")
    
    console.print("[bold magenta]🐺 Nikita is online. Let's get hostile.[/bold magenta]\n")

    while True:
        console.print("\n[bold cyan]┌──([bold white]SUDO[/bold white])[/bold cyan]")
        console.print(f"[bold cyan]└─> [/bold cyan]", end="")
        task = input().strip()
        
        if task.lower() in ["exit", "quit"]:
            # Save history before exit
            with open(HISTORY_FILE, 'w') as f:
                json.dump(chat_memory[-50:], f)  # Keep last 50 messages
            console.print("\n[bold red]Nikita:[/bold red] Exiting. Try not to get caught.\n")
            break

        extract_targets(task)
        attack_plan = suggest_attack_plan(task)
        if attack_plan:
            console.print(f"\n[yellow][Attack Plan][/yellow] {attack_plan}")

        context = "\n".join(chat_memory[-5:])  # Last 5 messages for context
        full_prompt = f"{get_dynamic_prompt()}\n\nRecent Context:\n{context}\n\nInput: {task}\n\nOutput:"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Thinking...", total=None)
            try:
                response = llm(
                    full_prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stop=["Input:", "Output:", "\n\n"]  # Prevent response continuation
                )
                response_text = response["choices"][0]["text"].strip()
                
                # Clean up response
                response_text = response_text.split('\n')[0]  # Take first line only
                for prefix in ["RESPONSE:", "OUTPUT:", "RESULT:", "Answer:"]:
                    if response_text.startswith(prefix):
                        response_text = response_text[len(prefix):].strip()
                
                if response_text.startswith("nmap"):
                    response_text = harden_command(response_text)
                
                response_text = inject_banter(response_text)
                
                chat_memory.append(f"Sudo: {task}\nResponse: {response_text}")
                console.print(f"\n[bold red]┌──([bold white]NIKITA 🐺[/bold white])[/bold red]")
                console.print(f"[bold red]└─> [/bold red]{response_text}\n")
                
                # Save output for commands
                if any(response_text.startswith(tool) for tool in ["nmap", "gobuster", "sqlmap", "sudo"]):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(OUTPUT_DIR, f"cmd_{timestamp}.txt")
                    with open(output_file, 'w') as f:
                        f.write(f"Command: {response_text}\n")
                        try:
                            result = subprocess.run(
                                shlex.split(response_text),
                                capture_output=True,
                                text=True,
                                timeout=300  # 5 minute timeout
                            )
                            f.write(f"\nOutput:\n{result.stdout}")
                            if result.stdout:
                                console.print(f"[green]{result.stdout}[/green]")
                            console.print(f"[cyan]Output saved to: {output_file}[/cyan]")
                        except Exception as e:
                            f.write(f"\nError: {str(e)}")
                            console.print(f"[red]Error running command: {e}[/red]")
                
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")

if __name__ == "__main__":
    main()
