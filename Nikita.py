#!/usr/bin/env python3

import subprocess
import shlex
from openai import OpenAI
from rich.console import Console

console = Console()

client = OpenAI(api_key="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")

def ai_parser(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that converts user input into Kali Linux commands."},
            {"role": "user", "content": prompt}
        ]
    )
    cmd = response.choices[0].message.content.strip()
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
    console.print("[bold green]NIKITA - AI Cyber Assistant[/bold green]")
    while True:
        user_input = input("Agent> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        ai_command = ai_parser(user_input)
        run_command(ai_command)

if __name__ == "__main__":
    main()
