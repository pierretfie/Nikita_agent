import re
import random
import subprocess
import shlex
import time
import json
import os
from functools import lru_cache  # Add caching
from collections import Counter
from modules.engagement_manager import extract_targets, get_default_network # Added get_default_network

# Define path for intent patterns relative to this file's location
INTENT_PATTERNS_FILE = os.path.join(os.path.dirname(__file__), "intent_patterns.json")
OUTPUT_DIR = "outputs" # Default output dir, consider making this configurable

class IntentAnalyzer:
    def __init__(self, output_dir=OUTPUT_DIR, system_commands=None):
        self.output_dir = output_dir
        self.patterns = self._load_patterns()
        self.system_commands = system_commands if system_commands else {}
        self.command_map = self._build_command_map()

    def _load_patterns(self):
        try:
            with open(INTENT_PATTERNS_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Intent patterns file not found at {INTENT_PATTERNS_FILE}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {INTENT_PATTERNS_FILE}")
            return {}
        except Exception as e:
            print(f"Error loading intent patterns: {e}")
            return {}

    def _build_command_map(self):
        """Build a map from intent keywords to specific commands."""
        command_map = {}
        # Use patterns if available
        if self.patterns and "intent_patterns" in self.patterns:
            for category, data in self.patterns["intent_patterns"].items():
                if "keywords" in data:
                    command = data.get("suggested_command", "") # Get suggested command
                    for keyword in data["keywords"]:
                        # Map keyword to potential command structure
                        command_map[keyword] = {
                            "base_command": command.split(" ")[0] if command else keyword, # Default base to keyword if no suggestion
                            "intent": category,
                            "example_params": data.get("example_params", ""),
                            "description": data.get("description", "")
                        }
        # Fallback or supplement with system commands
        for cmd, desc in self.system_commands.items():
            if cmd not in command_map: # Avoid overwriting pattern-based map
                 command_map[cmd] = {
                     "base_command": cmd,
                     "intent": "command_execution", # Default intent
                     "example_params": "",
                     "description": desc
                 }
        return command_map

    def _get_last_assistant_question(self, chat_memory):
        """Helper to find the last question asked by the assistant."""
        if not chat_memory:
            return None
        for i in range(len(chat_memory) - 1, -1, -1):
            entry = chat_memory[i]
            if entry.get("role") == "assistant":
                content = entry.get("content", "")
                if content.strip().endswith("?"):
                    # Basic check for choice questions
                    if " or " in content.lower() and ("public" in content.lower() and "private" in content.lower()):
                         return {"type": "choice", "question": content, "options": ["public", "private"]}
                    # Add other question type recognitions here if needed
                    return {"type": "general", "question": content}
                else:
                    return None # Last assistant message wasn't a question
        return None

    def analyze(self, user_input, chat_memory=None):
        """
        Analyze user input for intent, context, potential commands, and conversational state.
        Includes enhanced logic for informational queries and answers to questions.
        """
        analysis = {
            "intent": "unknown",
            "context": {},
            "command": None,
            "should_execute": False,
            "personal_reference": None, # Placeholder
            "emotional_context": None, # Placeholder
            "technical_context": None, # Placeholder
            "response": None, # Placeholder
            "targets": [],
            "follow_up_suggestions": [],
            "is_answer": False, # Flag if input is likely an answer
            "extracted_answer": None # Store extracted answer value
        }
        user_input_lower = user_input.lower().strip()

        if not user_input_lower:
            analysis["intent"] = "empty_input"
            return analysis

        # 1. Check if it's an answer to a previous question
        last_question = self._get_last_assistant_question(chat_memory)
        if last_question:
            # Check for specific answers to known question types
            if last_question["type"] == "choice" and last_question["options"] == ["public", "private"]:
                if "private" in user_input_lower:
                    analysis["is_answer"] = True
                    analysis["extracted_answer"] = "private"
                    analysis["intent"] = "answer_to_question" # Specific intent for answers
                elif "public" in user_input_lower:
                    analysis["is_answer"] = True
                    analysis["extracted_answer"] = "public"
                    analysis["intent"] = "answer_to_question"

            # Add more sophisticated answer checking here if needed
            # e.g., using fuzzy matching or checking against expected formats

        # If it's identified as an answer, prioritize that intent
        if analysis["is_answer"]:
             # Targets might still be relevant from the original question context
             if chat_memory:
                 # Look back for the user query that prompted the question
                 for i in range(len(chat_memory) - 2, -1, -1): # Start before the last assistant message
                     entry = chat_memory[i]
                     if entry.get("role") == "user":
                         original_query = entry.get("content", "")
                         analysis["targets"] = extract_targets(original_query) # Re-extract from original
                         break
             return analysis # Return early, reasoning engine will handle combining answer with original task

        # 2. Check for informational queries BEFORE command patterns
        # Simple patterns for informational queries
        informational_triggers = ["what is", "what's", "explain", "tell me about", "define"]
        if any(user_input_lower.startswith(trigger) for trigger in informational_triggers):
            analysis["intent"] = "informational_query"
            # Extract the topic of the query
            for trigger in informational_triggers:
                if user_input_lower.startswith(trigger):
                    analysis["context"]["topic"] = user_input.split(trigger, 1)[1].strip().rstrip("?")
                    break
            # No command needed for informational queries
            analysis["command"] = None
            analysis["should_execute"] = False
            # Generate informational follow-up suggestions if applicable
            # (Could be enhanced in reasoning engine based on topic)
            analysis["follow_up_suggestions"].append(f"Would you like to know more about related concepts to {analysis['context'].get('topic', 'this')}?")
            return analysis # Return early for informational queries

        # 3. If not an answer or informational query, proceed with command/general intent analysis
        # Extract potential targets first
        analysis["targets"] = extract_targets(user_input) # Extract targets

        # Intent matching based on patterns
        best_match_score = 0
        matched_intent = "general_query" # Default intent
        matched_data = {}

        if self.patterns and "intent_patterns" in self.patterns:
            for intent, data in self.patterns["intent_patterns"].items():
                score = 0
                keywords = data.get("keywords", [])
                regex_patterns = data.get("regex_patterns", [])

                # Keyword scoring
                for keyword in keywords:
                    if keyword in user_input_lower:
                        score += data.get("keyword_weight", 1) # Use specified weight or default

                # Regex scoring
                for pattern in regex_patterns:
                    if re.search(pattern, user_input_lower, re.IGNORECASE):
                        score += data.get("regex_weight", 2) # Regex matches are stronger

                # Check if score is better than current best
                if score > best_match_score:
                    best_match_score = score
                    matched_intent = intent
                    matched_data = data

        # Assign the best matched intent if score is above a threshold
        analysis["intent"] = matched_intent if best_match_score > 0 else "general_query"
        analysis["context"]["matched_pattern_data"] = matched_data # Store matched data for reasoning

        # Determine command based on intent and input
        analysis["command"], analysis["should_execute"] = self._determine_command(
            user_input,
            analysis["intent"],
            analysis["targets"] # Pass targets to command determination
        )

        # Generate follow-up suggestions based on intent and targets
        if analysis["intent"] in ["security_scan", "vulnerability_assessment"] and analysis["targets"]:
            for target in analysis["targets"]:
                 target_type = "IP/CIDR" if re.match(r'^\d{1,3}(\.\d{1,3}){3}(/\d{1,2})?$', target) else "Hostname/Other"
                 if target_type == "IP/CIDR":
                    analysis["follow_up_suggestions"].append(f"Scan {target} for common vulnerabilities?")
                    analysis["follow_up_suggestions"].append(f"Check open ports on {target}?")
                 else:
                    analysis["follow_up_suggestions"].append(f"Perform DNS enumeration for {target}?")
                    analysis["follow_up_suggestions"].append(f"Check web server configuration for {target}?")
        elif analysis["intent"] == "general_query" and analysis["targets"]:
             analysis["follow_up_suggestions"].append(f"Would you like to perform an action on the detected target(s): {', '.join(analysis['targets'])}?")


        # Basic emotional/personal/technical context (placeholders, can be expanded)
        # These would typically involve more complex NLP or pattern matching
        # analysis["emotional_context"] = self._analyze_emotion(user_input_lower)
        # analysis["personal_reference"] = self._find_personal_references(user_input_lower)
        # analysis["technical_context"] = self._assess_technicality(user_input_lower)

        return analysis

    def _determine_command(self, user_input, intent, targets):
        """
        Determine the command to execute based on intent, user input, and targets.
        Enhanced to use targets and default networks.
        """
        user_input_lower = user_input.lower()
        command = None
        should_execute = False # Default to not executing unless explicitly inferred

        # --- Command Logic ---
        # Use command_map for keyword-based command generation
        words = user_input_lower.split()
        base_command = None
        cmd_data = None

        # Find command keyword
        for word in words:
            if word in self.command_map:
                cmd_data = self.command_map[word]
                base_command = cmd_data.get("base_command", word)
                break
            # Check aliases if defined in patterns
            if self.patterns and "tool_aliases" in self.patterns:
                 for tool, aliases in self.patterns["tool_aliases"].items():
                     if word in aliases:
                         base_command = tool
                         # Find corresponding cmd_data for the actual tool name
                         if tool in self.command_map:
                             cmd_data = self.command_map[tool]
                         break
            if base_command: break # Exit outer loop once command keyword is found

        if base_command:
            command_parts = [base_command]
            target_specified = bool(targets)

            # Add target(s) if relevant to the command/intent
            if targets and intent in ["security_scan", "vulnerability_assessment", "network_enumeration", "command_execution"]:
                command_parts.extend(targets) # Add all found targets
            elif not target_specified and intent in ["security_scan", "network_enumeration"]:
                # If a scan is intended but no target given, use default network
                default_network = get_default_network()
                if default_network:
                    print(f"[IntentAnalyzer] No target specified for scan, using default network: {default_network}")
                    command_parts.append(default_network)
                    targets.append(default_network) # Add to analysis targets
                else:
                    print("[IntentAnalyzer] Scan intended, but no target specified and could not determine default network.")
                    # Maybe ask user for target in reasoning phase

            # Add parameters based on intent and keywords (simple example)
            if intent == "security_scan":
                if "deep" in user_input_lower or "thorough" in user_input_lower:
                    if base_command == "nmap": command_parts.insert(1, "-A") # Example for nmap
                elif "quick" in user_input_lower or "fast" in user_input_lower:
                     if base_command == "nmap": command_parts.insert(1, "-T4 -F") # Example for nmap
                # Add default scan options if none specified
                elif base_command == "nmap" and len(command_parts) == (len(targets) + 1): # Only command + target(s)
                     command_parts.insert(1, "-sV -T4") # Default versatile scan

            elif intent == "vulnerability_assessment":
                 if base_command == "nmap":
                     # Add vuln script if not already added
                     if "--script=vuln" not in command_parts and "-A" not in command_parts:
                         command_parts.insert(1, "--script=vuln")

            # Extract potential parameters mentioned by user (simple approach)
            potential_params = [word for word in words if word.startswith('-')]
            if potential_params:
                 # Insert params after base command but before targets
                 command_parts = [base_command] + potential_params + targets


            command = " ".join(command_parts)

            # Determine if the command should be executed directly
            # More conservative approach: only execute if intent is clearly execution-focused
            # and not just a request or query about a command.
            if intent == "command_execution": # Only execute if the primary intent is direct execution
                should_execute = True
            elif intent == "security_scan" and "run" in user_input_lower or "execute" in user_input_lower:
                 should_execute = True # Execute scans if explicitly told to run/execute


        # Handle cases where intent is command_request but no specific command identified
        # (e.g., "get my ip") - Reasoning engine should handle these based on intent.
        if intent == "command_request" and not command:
            # Let the reasoning engine figure out the command (like 'ip addr')
            pass # No command formulated here, but intent is captured

        # Refine command using shlex if needed (optional, good for complex commands)
        # try:
        #     command = " ".join(shlex.split(command))
        # except:
        #     pass # Keep original command string if shlex fails

        return command, should_execute

    # --- Placeholder methods for future expansion ---
    # def _analyze_emotion(self, text): return None
    # def _find_personal_references(self, text): return None
    # def _assess_technicality(self, text): return None

    @lru_cache(maxsize=128)
    def _determine_command(self, intent, query):
        """Determine appropriate command based on intent and query (Cached)"""
        mapping = self.command_mappings.get(intent, {})
        
        # Check for exact matches
        for keyword, command in mapping.items():
            if keyword in query:
                # Extract potential targets
                target_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', query)
                target = target_match.group(0) if target_match else None
                
                # Extract potential network
                network_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', query)
                network = network_match.group(0) if network_match else None
                
                # If no target/network specified, use default network
                if not target and not network and "scan" in query.lower():
                    from modules.engagement_manager import get_default_network
                    network = get_default_network()
                
                # Replace placeholders
                if target:
                    command = command.replace("{TARGET}", target)
                if network:
                    command = command.replace("{NETWORK}", network)
                
                return command
        
        # Handle special cases
        if intent == "security_scan":
            # Extract IP if present
            ip_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', query)
            target = ip_match.group(0) if ip_match else None
            
            # If no target specified, use default network
            if not target and "scan" in query.lower():
                from modules.engagement_manager import get_default_network
                target = get_default_network()
            
            if "port" in query or "service" in query:
                return f"nmap -sV -sC {target}"
            elif "vuln" in query:
                return f"nmap -sV --script vuln {target}"
            elif "web" in query:
                return f"nikto -h {target}"
            elif "hosts" in query or "discovery" in query:
                return f"nmap -sn {target}"
            else:
                return f"nmap -sV {target}"
        
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