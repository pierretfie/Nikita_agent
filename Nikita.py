#!/usr/bin/env python3

import subprocess
import shlex
import openai
from rich.console import Console

console = Console()

openai.api_key = "YOUR_API_KEY"  # or load from env

def ai_parser(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Convert this user input into a Kali Linux command: {prompt}",
        max_tokens=100
    )
    cmd = response.choices[0].text.strip()
    return cmd

def run_command(cmd):
    try:
        console.print(f"[bold cyan]Running: {cmd}[/bold cyan]")
        cmd_list = shlex.split(cmd)
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        console.print(result.stdout)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

def main():
    console.print("[bold green]KaliAI - AI Cyber Assistant[/bold green]")
    while True:
        user_input = input("Agent> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        ai_command = ai_parser(user_input)
        run_command(ai_command)

if __name__ == "__main__":
    main()
