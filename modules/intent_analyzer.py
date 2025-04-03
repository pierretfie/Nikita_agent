import re
import random
import subprocess
import shlex
import time
import json
import os
from functools import lru_cache  # Add caching

class IntentAnalyzer:
    def __init__(self, output_dir, system_commands):
        # Define intent categories
        self.output_dir = output_dir
        self.system_commands = system_commands # Store accessible commands
        
        # Load patterns from JSON file
        try:
            patterns_file = os.path.join(os.path.dirname(__file__), "intent_patterns.json")
            with open(patterns_file, "r") as f:
                patterns_data = json.load(f)
                self.intent_categories = patterns_data["intent_categories"]
                self.command_mappings = patterns_data["command_mappings"]
                self.response_templates = patterns_data["response_templates"]
                self.follow_up_patterns = patterns_data["follow_up_patterns"]
                self.quality_indicators = patterns_data["quality_indicators"]
        except Exception as e:
            print(f"Warning: Could not load intent patterns: {e}")
            # Fallback to empty patterns if file loading fails
            self.intent_categories = {}
            self.command_mappings = {}
            self.response_templates = {}
            self.follow_up_patterns = {}
        self.quality_indicators = {
            "empty_output": r'^\s*$',
                "error_patterns": [],
                "success_patterns": {}
        }

        self.command_execution_time = {} #Track time
        self.interaction_pattern = """When a user asks about something that involves system commands or actions:
1. First explain the concept or answer their question
2. If there's a relevant command or action that could help, ask if they'd like to see it
3. Only show or execute commands after getting explicit user consent
4. Keep responses conversational and interactive

Example pattern:
User: "What is X?"
Assistant: "X is [explanation]. Would you like to see how to [related action]? Just say 'yes' and I'll show you."
"""

    def analyze(self, user_input):
        """Analyze user input for intent and context"""
        input_lower = user_input.lower()
        
        # Initialize analysis result
        analysis = {
            "intent": None,
            "context": {},
            "command": None,
            "should_execute": False,
            "personal_reference": None,
            "emotional_context": None,
            "technical_context": None
        }
        
        # Check for personal references
        personal_patterns = {
            r"network\s+(\w+)": "network_contact",
            r"(\w+)\s+from\s+network": "network_contact",
            r"(\w+)\s+in\s+network": "network_contact"
        }
        
        for pattern, ref_type in personal_patterns.items():
            match = re.search(pattern, input_lower)
            if match:
                analysis["personal_reference"] = {
                    "name": match.group(1),
                    "type": ref_type,
                    "context": "network"
                }
                break
        
        # Analyze emotional context
        emotional_indicators = {
            "urgency": ["urgent", "asap", "quick", "hurry"],
            "frustration": ["frustrated", "angry", "annoyed"],
            "concern": ["worried", "concerned", "troubled"],
            "curiosity": ["wonder", "curious", "interested"]
        }
        
        for emotion, indicators in emotional_indicators.items():
            if any(indicator in input_lower for indicator in indicators):
                analysis["emotional_context"] = emotion
                break
        
        # Analyze technical context
        technical_indicators = {
            "network": ["network", "ip", "connection", "wifi", "ethernet"],
            "security": ["security", "secure", "protection", "vulnerability"],
            "system": ["system", "computer", "machine", "device"]
        }
        
        for context, indicators in technical_indicators.items():
            if any(indicator in input_lower for indicator in indicators):
                analysis["technical_context"] = context
                break
        
        # Determine intent based on context
        if analysis["personal_reference"]:
            if analysis["emotional_context"] == "urgency":
                analysis["intent"] = "urgent_network_contact"
            elif analysis["emotional_context"] == "concern":
                analysis["intent"] = "network_contact_concern"
            else:
                analysis["intent"] = "network_contact_query"
        
        # Generate appropriate response based on analysis
        if analysis["personal_reference"]:
            name = analysis["personal_reference"]["name"]
            if analysis["emotional_context"] == "urgency":
                analysis["response"] = f"I understand you need urgent assistance regarding {name} from the network team. How can I help?"
            elif analysis["emotional_context"] == "concern":
                analysis["response"] = f"I understand you're concerned about {name} from the network team. Let me help you with that."
            else:
                analysis["response"] = f"I understand you're looking for information about {name} from the network team. What specific information do you need?"
        
        return analysis

    @lru_cache(maxsize=128)
    def _determine_command(self, intent, query):
        """Determine appropriate command based on intent and query (Cached)"""
        mapping = self.command_mappings.get(intent, {})

        # Check for exact matches
        for keyword, command in mapping.items():
            if keyword in query:
                # Extract potential targets
                target_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', query)
                target = target_match.group(0) if target_match else "127.0.0.1"

                # Extract potential network
                network_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', query)
                network = network_match.group(0) if network_match else "192.168.1.0/24"

                # Extract file
                file_match = re.search(r'([/A-Za-z0-9_\-\.]+)', query)
                file = file_match.group(0) if file_match else "/usr/share/wordlists/rockyou.txt"


                # Replace placeholders
                command = command.replace("{TARGET}", target)
                command = command.replace("{NETWORK}", network)
                command = command.replace("{FILE}", file)

                return command

        # Handle special cases
        if intent == "wifi_operations" and "scan" in query:
            return "nmcli dev wifi list"

        if intent == "system_info" and "ip" in query:
            if "eth0" in query or "ethernet" in query:
                return "ip -4 addr show eth0"
            elif "wlan" in query or "wifi" in query:
                return "ip -4 addr show wlan0"
            else:
                return "ip -4 addr show"

        if intent == "security_scan":
            # Extract IP if present
            ip_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', query)
            target = ip_match.group(0) if ip_match else "127.0.0.1"

            if "port" in query or "service" in query:
                return f"nmap -sV -sC {target}"
            elif "vuln" in query:
                return f"nmap -sV --script vuln {target}"
            elif "web" in query:
                return f"nikto -h {target}"
            else:
                return f"nmap -sV {target}"

        # Wordlist info
        if intent == "wordlist_info":
            if "location" in query or "where" in query:
                return "which rockyou.txt"
            elif "size" in query:
                return "ls -l /usr/share/wordlists/rockyou.txt"
            else:
                 return "ls -l /usr/share/wordlists/rockyou.txt"

        return None

    def _get_agent_response(self, intent):
        """Get an appropriate response for agent-related queries"""
        templates = self.response_templates.get(intent, ["I'm Nikita, your security assistant."])
        return random.choice(templates)

    def format_command_response(self, command, output, error=None, save_command_output_func=None, system_commands = None):
        """Format a helpful response for command execution results with enhanced output validation"""
        if save_command_output_func is not None:
            save_command_output_func(command, output, error) # Pass function

        # Handle hashcat --identify output specifically
        if command.startswith("hashcat --identify"):
            if not output.strip():
                return "❌ No hash was found in the specified file. Please verify the file contains a valid hash."
            if "No hash-mode match found" in output:
                return "❌ Could not identify the hash type. The file might not contain a valid hash format."

            # Look for hash mode matches
            mode_matches = re.findall(r'(\d+) \| ([^|]+)\|([^\n]+)', output)
            if mode_matches:
                response = "✅ Identified possible hash types:\n"
                for mode_id, name, category in mode_matches:
                    response += f"• Mode {mode_id}: {name.strip()} ({category.strip()})\n"
                return response
            else:
                return "⚠️ Hash identification completed but no standard hash formats were detected."

        # Check for empty output or errors
        if not output.strip():
            if error and error.strip():
                return f"Command encountered an error: {error.strip()}"
            return "Command executed successfully, but didn't produce any output. This might indicate the command didn't find what it was looking for."

        # Check for error patterns in output
        for pattern in self.quality_indicators["error_patterns"]:
            if re.search(pattern, output, re.IGNORECASE) or (error and re.search(pattern, error, re.IGNORECASE)):
                return f"Command executed but encountered issues: {re.search(pattern, output or error, re.IGNORECASE).group(0)}"

        # Get command base for specialized formatting
        cmd_base = command.split()[0]
        cmd_args = " ".join(command.split()[1:]) if len(command.split()) > 1 else ""

        # Validate output quality based on command type
        success_patterns = self.quality_indicators["success_patterns"]

        # Check if command output contains expected patterns
        if cmd_base in success_patterns:
            valid_output = any(re.search(pattern, output) for pattern in success_patterns[cmd_base])
            if not valid_output:
                return f"Command executed, but the output doesn't contain expected information. This might indicate {cmd_base} couldn't retrieve the requested data."

        # Handle specific commands with specialized formatting
        if cmd_base == "ip" and "addr" in command:
            # Extract and format IP addresses
            interfaces = []
            current_iface = None
            ip_addresses = []

            for line in output.strip().split('\n'):
                # Detect interface lines
                if ': ' in line and '<' in line and '>' in line:
                    current_iface = line.split(':', 1)[0].strip().strip(':')
                    state = "UP" if "UP" in line else "DOWN"
                    interfaces.append(f"{current_iface} ({state})")
                # Detect IPv4 address lines
                elif current_iface and "inet " in line:
                    ip = line.strip().split()[1].split('/')[0]
                    ip_addresses.append(f"{ip} ({current_iface})")

            if ip_addresses:
                formatted_output = "Found these IP addresses:\n"
                for i, ip in enumerate(ip_addresses, 1):
                    formatted_output += f"{i}. {ip}\n"
                return formatted_output.strip()
            elif interfaces:
                return f"Found network interfaces:\n" + "\n".join(interfaces) + "\nBut no IP addresses were assigned."
            else:
                return "No active network interfaces or IP addresses were found."

        # Ping command with improved failure detection
        elif cmd_base == "ping":
            target = command.split()[-1]
            if "0 received" in output or "100% packet loss" in output:
                return f"❌ Target {target} is not responding to ping. This could indicate the host is offline, blocking ICMP packets, or there's a network connectivity issue."
            elif "icmp_seq=" in output:
                # Extract ping stats
                stats_match = re.search(r'(\d+) packets transmitted, (\d+) received', output)
                time_match = re.search(r'min/avg/max/(?:mdev|stddev) = ([\d.]+)/([\d.]+)/([\d.]+)', output)

                if stats_match:
                    sent, received = stats_match.groups()
                    loss_percent = 100 - (int(received) / int(sent) * 100)

                    status = "✅ up and responding normally" if loss_percent == 0 else f"⚠️ responding with {loss_percent}% packet loss"

                    if time_match and len(time_match.groups()) >= 2:
                        avg_time = time_match.groups()[1]
                        return f"Target {target} is {status} (avg response time: {avg_time}ms)"

                    return f"Target {target} is {status} (received {received}/{sent} packets)"

            return f"Ping to {target} completed, but the results are inconclusive."

        # Nmap command with improved output analysis
        elif cmd_base == "nmap":
            target = [arg for arg in command.split() if not arg.startswith('-')][-1]

            if "0 hosts up" in output or "All 1000 scanned ports on" in output and "closed" in output:
                return f"❌ Scan of {target} found no open ports or services. Target may be offline, firewalled, or not running any services on scanned ports."

            # Handle host discovery scans (-sn) for network device queries
            if "-sn" in command:
                # Count the hosts found
                hosts_up = re.findall(r'Host is up', output)
                host_count = len(hosts_up)

                # Extract IP addresses
                ip_addresses = re.findall(r'Nmap scan report for (?:[a-zA-Z0-9-]+\s)?\(?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\)?', output)

                # Format the response
                if host_count > 0:
                    response = f"✅ Found {host_count} devices on your network:\n"
                    for i, ip in enumerate(ip_addresses, 1):
                        # Try to extract hostnames if available
                        hostname_match = re.search(rf'Nmap scan report for ([a-zA-Z0-9-_.]+)\s+\(?{ip}\)?', output)
                        if not hostname_match:
                            # Try alternative pattern without parentheses
                            hostname_match = re.search(rf'Nmap scan report for ([a-zA-Z0-9-_.]+)(?!\s+\({ip}\))', output)
                        hostname = hostname_match.group(1) if hostname_match else ""

                        # Add hostname if available
                        if hostname and hostname != ip:
                            response += f"{i}. {ip} ({hostname})\n"
                        else:
                            response += f"{i}. {ip}\n"

                    # Add note about localhost
                    if "127.0.0.1" in ip_addresses:
                        response += "\nNote: This includes your local machine (127.0.0.1)."

                    return response
                else:
                    return "No devices were found on your network. This might indicate network connectivity issues or nmap couldn't detect any responding hosts."

            # Extract open ports with improved formatting
            open_ports = []
            for line in output.split('\n'):
                if 'open' in line and '/tcp' in line:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        port = parts[0].split('/')[0]
                        service = parts[2]
                        version = ' '.join(parts[3:]) if len(parts) > 3 else ""
                        open_ports.append(f"Port {port}: {service} {version}")

            if open_ports:
                return f"✅ Scan of {target} found {len(open_ports)} open ports:\n" + "\n".join(open_ports)
            else:
                # Look for other useful information
                host_status = "Host appears to be up" if "Host is up" in output else "Host status unclear"
                latency = re.search(r'latency: ([\d.]+)s', output)
                latency_info = f" (latency: {latency.group(1)}s)" if latency else ""

                return f"⚠️ Scan of {target} completed. {host_status}{latency_info}. No open ports were found in the scan range."

        # General info
        elif cmd_base in self.system_commands:
            if output.strip():
                return f"{output.strip()}"

    def _build_result(self, intent, pattern, cmd=None, response=None, should_execute=True):
        """Build a result dictionary for the analyzed intent"""
        result = {
            "intent": intent,
            "pattern": pattern,
            "command": cmd,
            "response": response,
            "should_execute": should_execute,
            "follow_up": self._get_follow_up(intent, response)
        }
        return result

    def _get_follow_up(self, intent, response):
        """Generate an interactive follow-up based on the intent and response"""
        if intent in self.follow_up_patterns:
            # Extract topic or location from response if possible
            topic = self._extract_topic_from_response(response)
            if topic:
                return random.choice(self.follow_up_patterns[intent]).format(
                    topic=topic,
                    location=topic
                )
        return None

    def _extract_topic_from_response(self, response):
        """Extract the main topic from a response for follow-up questions"""
        if not response:
            return None
            
        # Try to find the main subject of the response
        # This is a simple implementation that can be enhanced
        words = response.split()
        if len(words) > 0:
            # Look for capitalized words or words after "about" or "of"
            for i, word in enumerate(words):
                if word.lower() in ["about", "of"] and i + 1 < len(words):
                    return words[i + 1].strip('.,!?')
                if word[0].isupper() and len(word) > 2:
                    return word.strip('.,!?')
                    
        return None

    def _involves_command(self, query):
        """Check if the query involves a system command or action"""
        # Implement the logic to determine if a query involves a system command or action
        # This is a placeholder and should be replaced with the actual implementation
        return False

    def _is_ambiguous_name(self, name):
        """Check if a name is likely to be ambiguous using reasoning"""
        # Common patterns that suggest ambiguity
        ambiguity_indicators = [
            r'^(john|james|michael|david|robert|william|thomas|james|charles|joseph|thomas|daniel|paul|mark|donald|george|kenneth|steven|edward|brian|ronald|anthony|kevin|jason|matthew|gary|timothy|jose|larry|jeffrey|frank|scott|eric|stephen|andrew|raymond|gregory|joshua|jerry|dennis|walter|patrick|peter|harold|douglas|henry|carl|arthur|ryan|roger|joe|juan|jack|albert|jonathan|justin|terry|gerald|keith|samuel|willie|ralph|lawrence|nicholas|roy|benjamin|bruce|brandon|adam|harry|fred|wayne|billy|steve|louis|jeremy|aaron|randy|howard|eugene|carlos|russell|bobby|victor|martin|ernest|phillip|todd|jesse|craig|alan|shawn|clarence|sean|philip|chris|johnny|earl|jimmy|antonio|rodney|terry|evan|austin|jesus|nathan|kyle|stanley|adam|harry|fred|wayne|billy|steve|louis|jeremy|aaron|randy|howard|eugene|carlos|russell|bobby|victor|martin|ernest|phillip|todd|jesse|craig|alan|shawn|clarence|sean|philip|chris|johnny|earl|jimmy|antonio|rodney|terry|evan|austin|jesus|nathan|kyle|stanley)$',
            r'^(mary|patricia|jennifer|linda|elizabeth|barbara|susan|jessica|sarah|karen|nancy|lisa|betty|margaret|sandra|ashley|kimberly|emily|donna|michelle|dorothy|carol|amanda|melissa|deborah|stephanie|rebecca|sharon|laura|cynthia|amy|angela|helen|anna|brenda|pamela|nicole|emma|samantha|katherine|christine|debra|rachel|catherine|carolyn|janet|ruth|maria|heather|diane|virginia|julie|joyce|victoria|kelly|christina|lauren|joan|evelyn|olivia|judith|megan|cheryl|martha|megan|andrea|hannah|brenda|marie|sara|alice|julia|judy|abigail|maria|anne|jacqueline|kayla|alexis|lori|kimberly|heather|teresa|diana|natalie|sandra|brenda|denise|tammy|irene|jane|lori|rachel|marilyn|andrea|kathryn|louise|sara|anne|jacqueline|wanda|bonnie|julia|ruby|lois|tina|nellie|deborah|wanda|frances|elizabeth|julia|irene|adriana|hannah|marilyn|diana|gloria|jean|kelly|rose|catherine|gloria|lauren|sylvia|josephine|katie|gladys|marion|martha|gloria|tiffany|maxine|irma|jackie|jenny|kay|leona|lori|lynn|margie|may|mildred|nancy|nina|nora|pearl|phyllis|renee|roberta|robin|rosa|shirley|stacey|theresa|tina|tonya|vera|wanda|wendy)$',
            r'^(smith|johnson|williams|brown|jones|garcia|miller|davis|rodriguez|martinez|hernandez|lopez|gonzalez|wilson|anderson|thomas|taylor|moore|jackson|martin|lee|perez|thompson|white|harris|sanchez|clark|ramirez|lewis|robinson|walker|young|allen|king|wright|scott|torres|nguyen|hill|flores|green|adams|nelson|baker|hall|rivera|campbell|mitchell|carter|roberts|turner|phillips|evans|torres|parker|collins|edwards|stewart|morris|murphy|cook|rogers|gutierrez|ortiz|morgan|cooper|peterson|bailey|reed|kelly|howard|ramos|kim|cox|ward|richardson|watson|brooks|chavez|wood|james|bennett|gray|reyes|cruz|hughes|price|myers|long|foster|sanders|ross|morales|powell|sullivan|russell|ortiz|jenkins|gutierrez|perry|butler|barnes|fisher)$'
        ]
        
        # Check if the name matches any common patterns that suggest ambiguity
        name_lower = name.lower()
        for pattern in ambiguity_indicators:
            if re.match(pattern, name_lower):
                return True
                
        # Check for common name combinations that might be ambiguous
        name_parts = name_lower.split()
        if len(name_parts) > 1:
            # If it's a common first name + common last name combination
            if any(re.match(ambiguity_indicators[0], name_parts[0])) and any(re.match(ambiguity_indicators[2], name_parts[-1])):
                return True
                
        return False