"""
Reasoning Engine Module for Nikita Agent

Provides structured reasoning for tasks and security-related decision making.
"""

import re
import shlex
import os
import json
from .engagement_manager import engagement_memory
import random

# Try to import rich for pretty output if available
try:
    from rich.console import Console
    console = Console()
except ImportError:
    # Fallback to simple print if rich is not available
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = FallbackConsole()

class ReasoningEngine:
    def __init__(self):
        """Initialize the reasoning engine for task analysis"""
        # Load human-like patterns
        self.patterns = self._load_patterns()
        
        # Load reasoning datasets
        self.datasets = self._load_datasets()
        
        # Load emotional patterns
        self.emotional_patterns = self._load_emotional_patterns()
        
        # Personality traits
        self.personality = {
            "tone": "professional yet friendly",
            "empathy_level": "high",
            "expertise_style": "helpful and educational",
            "conversation_style": "natural and engaging"
        }
        
        # Natural conversation patterns
        self.conversation_patterns = self.patterns.get("conversation_patterns", {})
        
        # Security domain knowledge
        self.security_domain = self.patterns.get("security_domain", {})
        
        # Engagement patterns
        self.engagement_patterns = self.patterns.get("engagement_patterns", {})

        # Initialize security datasets
        self.security_concepts = self.datasets.get("security_concepts", {})
        self.technical_terminology = self.datasets.get("technical_terminology", {})
        self.security_tools = self.datasets.get("security_tools", {})
        self.vulnerability_patterns = self.datasets.get("vulnerability_patterns", {})
        self.security_metrics = self.datasets.get("security_metrics", {})
        self.response_patterns = self.datasets.get("response_patterns", {})

        self.reasoning_template = """
Thought Process:
1. UNDERSTAND: {task}
   - Goal: {goal}
   - Context: {context}
   - Constraints: {constraints}
   - Ambiguity Level: {ambiguity_level}
   - User Expertise: {user_expertise}
   - Emotional Context: {emotional_context}
   - Conversation Style: {conversation_style}

2. PLAN:
   - Required steps: {steps}
   - Dependencies: {dependencies}
   - Order: {order}
   - Interactive Elements: {interactive_elements}
   - Engagement Strategy: {engagement_strategy}

3. TOOLS:
   - Primary tool: {primary_tool}
   - Alternative tools: {alternative_tools}
   - Parameters needed: {parameters}

4. SAFETY:
   - Risks: {risks}
   - Precautions: {precautions}
   - Fallback plan: {fallback}

5. EXECUTION:
   Command: {command}
   Explanation: {explanation}
   Natural Language: {natural_language}

6. ANALYSIS:
   - Expected output: {expected_output}
   - Success indicators: {success_indicators}
   - Next steps: {next_steps}
   - Follow-up Questions: {follow_up_questions}
   - Engagement Opportunities: {engagement_opportunities}
"""

    def _load_patterns(self):
        """Load human-like patterns from JSON file"""
        try:
            patterns_file = os.path.join(os.path.dirname(__file__), "human_like_patterns.json")
            with open(patterns_file, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load human-like patterns: {e}[/yellow]")
            return {}

    def _load_datasets(self):
        """Load reasoning datasets from JSON file"""
        try:
            datasets_file = os.path.join(os.path.dirname(__file__), "reasoning_datasets.json")
            with open(datasets_file, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load reasoning datasets: {e}[/yellow]")
            return {}

    def _load_emotional_patterns(self):
        """Load emotional patterns from JSON file"""
        try:
            patterns_file = os.path.join(os.path.dirname(__file__), "emotional_patterns.json")
            with open(patterns_file, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load emotional patterns: {e}[/yellow]")
            return {}

    def _get_pattern(self, category, subcategory=None, context=None):
        """Get a random pattern from the specified category"""
        try:
            if subcategory and context:
                patterns = self.patterns[category][subcategory][context]
            elif subcategory:
                patterns = self.patterns[category][subcategory]
            else:
                patterns = self.patterns[category]
            
            if isinstance(patterns, list):
                return random.choice(patterns)
            return patterns
        except:
            return ""

    def _get_dataset(self, category, subcategory=None, context=None):
        """Get data from the specified dataset category"""
        try:
            if subcategory and context:
                data = self.datasets[category][subcategory][context]
            elif subcategory:
                data = self.datasets[category][subcategory]
            else:
                data = self.datasets[category]
            
            if isinstance(data, list):
                return data
            return data
        except:
            return []

    def analyze_task(self, task, intent_analysis=None):
        """Analyze the task and generate appropriate reasoning, using provided intent analysis if available."""
        # Get intent analysis details if provided, otherwise perform basic analysis
        primary_intent = intent_analysis.get("intent") if intent_analysis else None
        command = intent_analysis.get("command") if intent_analysis else None
        emotional_context = intent_analysis.get("emotional_context") if intent_analysis else self._analyze_emotional_context(task)
        technical_context = intent_analysis.get("technical_context") if intent_analysis else None
        personal_context = intent_analysis.get("personal_reference") if intent_analysis else None
        answered_context = intent_analysis.get("answered_context") if intent_analysis else None

        # Initialize reasoning template
        reasoning = {
            "task_analysis": {
                "primary_intent": primary_intent, # Use intent from analysis
                "secondary_intents": [],
                "personal_context": personal_context, # Use personal context from analysis
                "technical_context": None, # Will be populated based on intent/keywords
                "emotional_context": emotional_context, # Use emotional context from analysis
                "answered_context": answered_context # Track if this is an answer to a previous question
            },
            "response_strategy": {
                "approach": None,
                "tone": None,
                "technical_level": None,
                "follow_up_questions": []
            },
            "execution_plan": {
                "steps": [],
                "priority": "normal",
                "dependencies": [],
                "command": command # Include command from analysis
            }
        }

        # Determine primary intent if not provided by analyzer
        if not primary_intent:
            security_keywords = ["scan", "hack", "exploit", "attack", "security", "detect", "bypass", "evade", "tool", "best", "command", "run", "execute"]
            if any(keyword in task.lower() for keyword in security_keywords) or command:
                primary_intent = "security" # Default to security if keywords or command present
            else:
                primary_intent = "general_query" # Fallback
            reasoning["task_analysis"]["primary_intent"] = primary_intent

        # Refine reasoning based on primary intent
        if primary_intent in ["command_execution", "command_request", "help_request", "security"]:
            reasoning["response_strategy"]["approach"] = "technical"
            reasoning["response_strategy"]["tone"] = "professional"
            reasoning["response_strategy"]["technical_level"] = "advanced"

            # Add security-specific context
            reasoning["task_analysis"]["technical_context"] = {
                "domain": technical_context or "security", # Use provided or default
                "complexity": "advanced",
                "risk_level": "high"
            }

            # Add tool-specific context if asking about tools or a command is present
            if "tool" in task.lower() or "best" in task.lower() or command:
                focus = "tool_recommendation" if "best" in task.lower() else "tool_usage"
                reasoning["task_analysis"]["technical_context"]["focus"] = focus

                # Add follow-up based on whether it's help or execution and if there's an answered context
                if answered_context:
                    # If this is an answer to a previous question, generate appropriate follow-up
                    if answered_context.get("answered_question", "").lower().startswith(("what", "how", "which", "when", "where", "why")):
                        # For "what" questions about tools/commands
                        if "what" in answered_context["answered_question"].lower():
                            reasoning["response_strategy"]["follow_up_questions"] = [
                                "Would you like to see an example of how to use this?",
                                "Would you like me to explain more about its features?",
                                "Would you like to know about alternative tools?"
                            ]
                        # For "how" questions about tools/commands
                        elif "how" in answered_context["answered_question"].lower():
                            reasoning["response_strategy"]["follow_up_questions"] = [
                                "Would you like me to show you the command syntax?",
                                "Would you like to see some common use cases?",
                                "Would you like to know about best practices?"
                            ]
                        # For "which" questions about tools/commands
                        elif "which" in answered_context["answered_question"].lower():
                            reasoning["response_strategy"]["follow_up_questions"] = [
                                "Would you like to know more about why this is the best choice?",
                                "Would you like to see a comparison with other options?",
                                "Would you like to know about its specific advantages?"
                            ]
                        else:
                            # Default follow-up questions for other types of questions
                            reasoning["response_strategy"]["follow_up_questions"] = [
                                "Would you like me to explain more about this?",
                                "Would you like to see some practical examples?",
                                "Would you like to know about related concepts?"
                            ]
                    else:
                        # For non-question answers, generate context-appropriate follow-up
                        reasoning["response_strategy"]["follow_up_questions"] = [
                            "Would you like to proceed with this option?",
                            "Would you like to know more about what this means?",
                            "Would you like to explore other options?"
                        ]
                else:
                    # Original follow-up questions for new queries
                    if primary_intent == "help_request":
                        reasoning["response_strategy"]["follow_up_questions"] = [
                            f"What specifically about '{command.split()[-1] if command else 'this command'}' would you like help with?",
                            "Are you looking for usage examples or explanations of options?"
                        ]
                    elif primary_intent in ["command_request", "command_execution"]:
                        reasoning["response_strategy"]["follow_up_questions"] = [
                            "What is the target or scope for this command?",
                            "Are there any specific parameters you want to use?"
                        ]
                    else: # General tool query
                        reasoning["response_strategy"]["follow_up_questions"] = [
                            "What specific security requirements do you have?",
                            "Are you looking for a specific type of security tool?",
                            "Do you need a tool for a particular security task?"
                        ]

                # Add tool recommendations if appropriate (example)
                if "scan" in task.lower() and "detect" in task.lower():
                    reasoning["execution_plan"]["recommended_tools"] = [
                        {
                            "name": "Nmap",
                            "description": "Advanced network scanning tool with stealth capabilities",
                            "features": ["SYN scan", "Version detection", "OS fingerprinting", "Stealth mode"],
                            "command": "nmap -sS -T2 -p- --max-retries 1 --randomize-hosts --scan-delay 5s {TARGET}"
                        },
                        {
                            "name": "Masscan",
                            "description": "Fast port scanner with minimal detection footprint",
                            "features": ["Asynchronous scanning", "Low detection rate", "High performance"],
                            "command": "masscan {TARGET} -p1-65535 --rate=1000"
                        }
                    ]
                elif "vulnerability" in task.lower():
                     # (Vulnerability tool recommendations remain the same)
                     pass # Placeholder

            # Add security-specific steps
            reasoning["execution_plan"]["steps"] = [
                "analyze security requirements",
                "identify potential risks",
                "develop security strategy",
                "implement security measures" if primary_intent == "command_execution" else "formulate command/provide help",
                "verify security implementation" if primary_intent == "command_execution" else "confirm understanding"
            ]
        elif primary_intent in ["network_contact_query", "urgent_network_contact", "network_contact_concern"]:
             # Handle personal reference specific reasoning
             reasoning["task_analysis"]["technical_context"] = {"domain": "communication"}
             reasoning["response_strategy"]["approach"] = "personal_reference"
             reasoning["response_strategy"]["tone"] = "professional_empathy"
             reasoning["response_strategy"]["follow_up_questions"] = [
                 f"What specific information do you need about {personal_context['name'] if personal_context else 'them'}?",
                 "Is this regarding a specific network issue or task?",
                 "Would you like me to help you contact them or find information about their work?"
             ]
             reasoning["execution_plan"]["steps"] = ["clarify request", "gather contact info (if requested)", "provide assistance"]

        else: # General query or other intents
            reasoning["response_strategy"]["approach"] = "informative"
            reasoning["response_strategy"]["tone"] = "helpful"
            reasoning["response_strategy"]["technical_level"] = "moderate"
            reasoning["task_analysis"]["technical_context"] = {"domain": technical_context or "general"}
            reasoning["execution_plan"]["steps"] = ["understand query", "provide information", "ask clarifying questions"]

        # Combine reasoning components into a single dictionary for the final prompt
        final_reasoning_context = {
             "reasoning": reasoning, # Keep the nested structure
             "follow_up_questions": reasoning["response_strategy"].get("follow_up_questions", [])
        }

        return final_reasoning_context

    def _determine_goal(self, task):
        """Determine the goal based on task description"""
        task_lower = task.lower()
        
        if re.search(r'(scan|enumerate|discover|find|list)', task_lower):
            return "Discover and enumerate network resources or information"
        elif re.search(r'(check|verify|confirm|test)', task_lower):
            return "Verify system status or confirm operational state"
        elif re.search(r'(analyze|examine|study|investigate)', task_lower):
            return "Analyze system data or investigate specific conditions"
        elif re.search(r'(show|display|get|what)', task_lower):
            return "Retrieve and display specific system information"
        elif re.search(r'(ip|address|network)', task_lower):
            return "Identify network configuration or addressing information"
        else:
            return "Execute requested operation and provide relevant information"

    def _analyze_context(self, task):
        """Analyze context of the task"""
        # Get active targets from engagement memory
        targets = engagement_memory.get("targets", [])
        return f"User request in context of system state and {'active targets: ' + ', '.join(targets) if targets else 'no active targets'}"

    def _identify_constraints(self, task):
        """Identify constraints for the task"""
        constraints = ["Current user permissions", "System resource availability"]
        
        # Add task-specific constraints
        if "remote" in task.lower():
            constraints.append("Network connectivity to remote systems")
        if "scan" in task.lower():
            constraints.append("Scan performance and target responsiveness")
            
        return ", ".join(constraints)

    def _determine_dependencies(self, category):
        """Determine dependencies based on category"""
        dependencies = {
            "Recon": "Network access, required scan permissions",
            "Web": "HTTP client libraries, web server access",
            "Wireless": "Wireless interface in monitor mode",
            "Password": "Dictionary files, processing capability",
            "General": "Basic system tools and utilities"
        }
        return dependencies.get(category, dependencies["General"])

    def _get_risks(self, category):
        """Get risks associated with the category"""
        risks = {
            "Recon": "Target detection of scanning activity, false positives",
            "Web": "Unexpected service disruption, detected intrusion attempts",
            "Wireless": "Regulatory compliance issues, detection by monitors",
            "Password": "Account lockouts, audit log generation",
            "General": "Command timeout, unexpected output format"
        }
        return risks.get(category, risks["General"])

    def _get_precautions(self, category):
        """Get precautions for the category"""
        precautions = {
            "Recon": "Use proper scan timing, verify scope authorization",
            "Web": "Validate input, check for WAF/security controls",
            "Wireless": "Ensure regulatory compliance, verify isolated testing",
            "Password": "Monitor for lockout policies, use incremental approach",
            "General": "Validate commands before execution, review output carefully"
        }
        return precautions.get(category, precautions["General"])

    def _determine_expected_output(self, category, tool):
        """Determine expected output based on category and tool"""
        if category == "Recon" and tool == "nmap":
            return "Port status, service versions, host information"
        elif category == "Web":
            return "HTTP responses, discovered endpoints, potential vulnerabilities"
        elif "ping" in tool:
            return "Response time, packet statistics, host availability"
        elif "ip" in tool:
            return "Network interface information, addressing details"
        else:
            return "Command-specific output relevant to the task"

    def _determine_success_indicators(self, category):
        """Determine success indicators based on category"""
        indicators = {
            "Recon": "Discovered hosts, identified services, mapped network",
            "Web": "Accessible endpoints, identified technologies, vulnerability confirmation",
            "Wireless": "Captured packets, identified networks, successful authentication",
            "Password": "Successful authentication, cracked hashes, identified weaknesses",
            "General": "Clean command execution, relevant output, actionable information"
        }
        return indicators.get(category, indicators["General"])

    def _determine_next_steps(self, category):
        """Determine next steps based on category"""
        next_steps = {
            "Recon": "Target specific services, perform deeper analysis on open ports",
            "Web": "Follow up on identified endpoints, test vulnerabilities, gather more information",
            "Wireless": "Analyze captured data, attempt authentication if appropriate",
            "Password": "Use identified credentials, attempt privilege escalation",
            "General": "Analyze output, refine approach based on results"
        }
        return next_steps.get(category, next_steps["General"])

    def _get_category_steps(self, category):
        """Get standard steps for a given category"""
        steps_by_category = {
            "Recon": [
                "Verify target scope and permissions",
                "Perform initial host discovery",
                "Identify open ports and services",
                "Gather service versions and details"
            ],
            "Web": [
                "Check target website availability",
                "Identify web technologies used",
                "Scan for common vulnerabilities",
                "Test discovered endpoints"
            ],
            "Wireless": [
                "Put interface in monitor mode",
                "Scan for target networks",
                "Capture required handshakes/data",
                "Analyze captured data"
            ],
            "Password": [
                "Identify authentication mechanism",
                "Prepare wordlist/attack method",
                "Execute brute force attempt",
                "Monitor for successful attempts"
            ]
        }
        return steps_by_category.get(category, ["Analyze requirements", "Plan approach", "Execute safely", "Verify results"])

    def _guess_category(self, task):
        """Guess the category based on task keywords"""
        task_lower = task.lower()
        categories = {
            "Recon": ["scan", "enumerate", "discover", "find", "list"],
            "Web": ["http", "website", "url", "web", "port 80", "port 443"],
            "Wireless": ["wifi", "wireless", "wpa", "handshake", "deauth"],
            "Password": ["crack", "brute", "password", "hash", "login"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in task_lower for keyword in keywords):
                return category
        return "General"

    def _guess_primary_tool(self, task):
        """Guess the primary tool needed based on task description"""
        task_lower = task.lower()
        tool_keywords = {
            "nmap": ["scan", "port", "service", "version"],
            "hashcat": ["crack", "hash", "password"],
            "aircrack-ng": ["wifi", "wpa", "wireless", "handshake"],
            "gobuster": ["directory", "web", "brute", "website"],
            "hydra": ["brute", "login", "password", "ssh", "ftp"]
        }
        
        for tool, keywords in tool_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return tool
        return "To be determined"

    def _get_additional_tools(self, category):
        """Get commonly paired tools for a category"""
        tool_pairs = {
            "Recon": ["dig", "whois", "traceroute"],
            "Web": ["curl", "burpsuite", "nikto"],
            "Wireless": ["airmon-ng", "airodump-ng", "wireshark"],
            "Password": ["john", "hashcat", "crunch"]
        }
        return tool_pairs.get(category, [])

    def _extract_parameters(self, command):
        """Extract parameters from a command string"""
        if not command or command == "To be determined based on further analysis":
            return []
            
        params = []
        parts = shlex.split(command)
        for part in parts[1:]:  # Skip the command name
            if part.startswith("-"):
                params.append(part)
        return params 

    def _analyze_ambiguity(self, task):
        """Analyze the level of ambiguity in the task using sophisticated pattern matching"""
        task_lower = task.lower()
        
        # Use security concepts from datasets for better context analysis
        attack_techniques = self._get_dataset("security_concepts", "attack_techniques")
        defense_mechanisms = self._get_dataset("security_concepts", "defense_mechanisms")
        
        # Complex ambiguous patterns with context
        ambiguous_patterns = {
            "High": [
                r'who is (.+)',
                r'what is (.+)',
                r'where is (.+)',
                r'how to (.+)',
                r'which (.+)',
                r'when should (.+)',
                r'why does (.+)',
                r'can you explain (.+)',
                r'what does (.+) mean',
                r'how does (.+) work'
            ],
            "Medium": [
                r'it is (.+)',
                r'this is (.+)',
                r'that is (.+)',
                r'they are (.+)',
                r'those are (.+)',
                r'this (.+)',
                r'that (.+)',
                r'these (.+)',
                r'those (.+)'
            ],
            "Low": [
                r'run (.+)',
                r'execute (.+)',
                r'scan (.+)',
                r'check (.+)',
                r'verify (.+)',
                r'analyze (.+)',
                r'test (.+)',
                r'find (.+)'
            ]
        }
        
        # Check for ambiguous patterns with context
        for level, patterns in ambiguous_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, task_lower)
                if match:
                    # Additional context analysis using security concepts
                    context = match.group(1) if match.groups() else ""
                    
                    # Check if context contains security-related terms
                    security_terms = set(attack_techniques + defense_mechanisms)
                    context_words = set(context.lower().split())
                    
                    # If context contains security terms, it's likely more specific
                    if context_words & security_terms:
                        return "Low - Security-specific context"
                    
                    if self._is_ambiguous_context(context):
                        return f"{level} - Requires clarification"
                    elif self._needs_context(context):
                        return "Medium - May need context"
        
        return "Low - Clear and specific"

    def _is_ambiguous_context(self, context):
        """Check if the context is ambiguous based on multiple factors"""
        # Common ambiguous terms in security context
        ambiguous_terms = {
            "security": ["firewall", "antivirus", "encryption", "authentication"],
            "network": ["port", "protocol", "service", "connection"],
            "system": ["process", "service", "daemon", "application"],
            "access": ["permission", "right", "privilege", "authorization"]
        }
        
        # Check for ambiguous terms in context
        for category, terms in ambiguous_terms.items():
            if any(term in context.lower() for term in terms):
                return True
                
        # Check for multiple possible interpretations
        if len(context.split()) > 3:  # Longer phrases might be ambiguous
            return True
            
        return False

    def _needs_context(self, context):
        """Determine if the context needs additional information"""
        # Terms that often need context
        context_needed = [
            "it", "this", "that", "them", "those",
            "the system", "the network", "the service",
            "the application", "the process", "the file"
        ]
        
        return any(term in context.lower() for term in context_needed)

    def _assess_user_expertise(self, task):
        """Assess user expertise using sophisticated pattern matching"""
        task_lower = task.lower()
        
        # Use technical terminology from datasets
        networking_terms = self._get_dataset("technical_terminology", "networking")
        system_terms = self._get_dataset("technical_terminology", "system")
        
        # Use security tools from datasets
        reconnaissance_tools = self._get_dataset("security_tools", "reconnaissance")
        exploitation_tools = self._get_dataset("security_tools", "exploitation")
        
        # Comprehensive technical terms by category
        technical_terms = {
            "Advanced": [
                # Security concepts
                *self._get_dataset("security_concepts", "attack_techniques", "exploitation"),
                *self._get_dataset("security_concepts", "attack_techniques", "post_exploitation"),
                # Technical implementations
                *system_terms,
                *networking_terms,
                # Advanced tools
                *reconnaissance_tools,
                *exploitation_tools
            ],
            "Intermediate": [
                # Basic security concepts
                *self._get_dataset("security_concepts", "defense_mechanisms", "prevention"),
                *self._get_dataset("security_concepts", "defense_mechanisms", "detection"),
                # Common tools
                *self._get_dataset("security_tools", "reconnaissance", "network")
            ],
            "Beginner": [
                # Basic terms
                "how to", "what is", "explain", "help me",
                "can you", "please", "show me", "tell me",
                "guide", "tutorial", "step by step", "example",
                # Simple concepts
                "password", "login", "account", "file",
                "folder", "program", "website", "internet"
            ]
        }
        
        # Count matches for each expertise level
        expertise_counts = {
            level: sum(1 for term in terms if term in task_lower)
            for level, terms in technical_terms.items()
        }
        
        # Determine expertise based on counts and patterns
        if expertise_counts["Advanced"] > 2:
            return "Advanced - Technical terminology used"
        elif expertise_counts["Intermediate"] > 3:
            return "Intermediate - Mixed terminology"
        elif expertise_counts["Beginner"] > 2:
            return "Beginner - Basic query structure"
        else:
            return "Intermediate - Mixed terminology"

    def _analyze_emotional_context(self, task):
        """Analyze emotional context using sophisticated pattern matching"""
        task_lower = task.lower()
        
        # Get emotional patterns
        emotional_indicators = self.emotional_patterns.get("emotional_indicators", {})
        emotional_analysis = self.emotional_patterns.get("emotional_analysis", {})
        personal_references = self.emotional_patterns.get("personal_references", {})
        
        # Initialize emotional context
        context = {
            "emotion": "neutral",
            "intensity": "low",
            "context_match": False,
            "response": "",
            "compound_emotions": [],
            "context_weights": {},
            "emotional_flow": [],
            "personal_reference": None,
            "fact_verified": False,
            "follow_up": False,
            "previous_context": None
        }
        
        # Check for follow-up questions
        follow_up_indicators = ["yes", "yeah", "yep", "sure", "okay", "ok", "alright"]
        if any(indicator in task_lower for indicator in follow_up_indicators):
            context["follow_up"] = True
            # Use previous context if available
            if hasattr(self, 'previous_context'):
                context["previous_context"] = self.previous_context
        
        # Analyze emotional indicators
        for emotion, intensities in emotional_indicators.items():
            for intensity, data in intensities.items():
                if any(indicator in task_lower for indicator in data["indicators"]):
                    context["emotion"] = emotion
                    context["intensity"] = intensity
                    context["compound_emotions"].append(emotion)
                    context["response"] = random.choice(data["responses"])
                    context["context_match"] = True
        
        # Analyze emotional flow
        emotional_flow = emotional_analysis.get("emotional_flow", {}).get("transition_patterns", {})
        for flow_name, data in emotional_flow.items():
            if any(indicator in task_lower for indicator in data["indicators"]):
                context["emotional_flow"].append({
                    "name": flow_name,
                    "response": data["response"]
                })
        
        # Analyze context weights
        context_analysis = emotional_analysis.get("context_analysis", {})
        for ctx_type, data in context_analysis.items():
            weight = data.get("weight", 1.0)
            indicators = data.get("indicators", [])
            matches = sum(1 for indicator in indicators if indicator in task_lower)
            if matches > 0:
                context["context_weights"][ctx_type] = weight * matches
        
        # Check for content creator references using patterns
        content_creators = personal_references.get("content_creators", {})
        patterns = content_creators.get("patterns", {})
        validation_rules = patterns.get("validation_rules", {})
        
        # First check for known content creators
        for creator_id, creator_data in validation_rules.items():
            verification_indicators = creator_data.get("verification_indicators", [])
            if any(indicator in task_lower for indicator in verification_indicators):
                context["personal_reference"] = {
                    "type": "content_creator",
                    "data": creator_data
                }
                context["fact_verified"] = True
                
                # Handle specific queries about works
                if "books" in task_lower and "known_works" in creator_data:
                    works = creator_data["known_works"]
                    if "books" in works:
                        context["response"] = f"{creator_data['name']} has written several books including: {', '.join(works['books'])}"
                        return context
                
                # Use standard response template
                response_templates = content_creators.get("response_templates", [])
                if response_templates:
                    template = random.choice(response_templates)
                    context["response"] = template.format(
                        name=creator_data["name"],
                        role=creator_data.get("expertise", [""])[0],
                        field=creator_data.get("expertise", [""])[0],
                        topics=", ".join(creator_data.get("expertise", [])[:3])
                    )
                return context
        
        # If no verified match, use pattern-based extraction
        name_patterns = patterns.get("name_extraction", [])
        role_indicators = patterns.get("role_indicators", {})
        
        # Extract name using patterns
        extracted_name = None
        for pattern in name_patterns:
            match = re.search(pattern, task_lower)
            if match:
                extracted_name = match.group(1)
                break
        
        if extracted_name:
            # Determine role and field based on context
            role = "general"
            field = "general"
            topics = []
            
            # Analyze context to determine role and field
            for role_type, indicators in role_indicators.items():
                if any(indicator in task_lower for indicator in indicators):
                    role = role_type
                    field = role_type
                    topics.extend(indicators)
                    break
            
            # Generate response using templates
            response_templates = content_creators.get("response_templates", [])
            if response_templates:
                template = random.choice(response_templates)
                context["response"] = template.format(
                    name=extracted_name,
                    role=role,
                    field=field,
                    topics=", ".join(topics[:3])
                )
                context["personal_reference"] = {
                    "type": "content_creator",
                    "name": extracted_name,
                    "role": role,
                    "field": field,
                    "topics": topics
                }
                context["context_match"] = True
                return context
        
        # If no match found and fact checking is enabled
        fact_checking = content_creators.get("fact_checking", {})
        if fact_checking.get("enabled", False):
            context["response"] = fact_checking.get("fallback_response", "I apologize, but I need to verify this information. Could you provide more context about who you're asking about?")
        
        # Store context for follow-up questions
        self.previous_context = context
        
        return context

    def _determine_conversation_style(self, task):
        """Determine conversation style using patterns from JSON file"""
        task_lower = task.lower()
        
        # Get style indicators from loaded patterns
        style_indicators = self.emotional_patterns.get("style_indicators", {})
        
        # Analyze style indicators
        for style, patterns in style_indicators.items():
            if any(pattern in task_lower for pattern in patterns):
                return style
        
        # Default to friendly and informative
        return "Friendly and informative"

    def _determine_engagement_strategy(self, task):
        """Determine the best engagement strategy using enhanced emotional analysis"""
        emotional_context = self._analyze_emotional_context(task)
        user_expertise = self._assess_user_expertise(task)
        
        # Build strategy based on emotional context and expertise
        strategy_parts = []
        
        # Add emotional support based on compound emotions
        if emotional_context["compound_emotions"]:
            for emotion in emotional_context["compound_emotions"]:
                if "frustration" in emotion:
                    strategy_parts.append("Supportive approach with clear, step-by-step guidance")
                elif "urgency" in emotion:
                    strategy_parts.append("Efficient, focused approach with clear priorities")
                elif "curiosity" in emotion:
                    strategy_parts.append("Educational approach with detailed explanations")
                elif "concern" in emotion:
                    strategy_parts.append("Reassuring approach with security-focused guidance")
        
        # Add expertise-based elements
        if user_expertise == "Beginner - Basic query structure":
            strategy_parts.append("Educational elements with clear explanations")
        elif user_expertise == "Advanced - Technical terminology used":
            strategy_parts.append("Technical discussion with advanced concepts")
        
        # Add context-aware elements based on weights
        if emotional_context["context_weights"]:
            max_weight = max(emotional_context["context_weights"].values())
            for ctx_type, weight in emotional_context["context_weights"].items():
                if weight == max_weight:
                    if ctx_type == "security_context":
                        strategy_parts.append("Security-focused guidance and best practices")
                    elif ctx_type == "technical_context":
                        strategy_parts.append("Technical implementation details and examples")
                    elif ctx_type == "learning_context":
                        strategy_parts.append("Educational approach with clear examples")
        
        # Add emotional flow elements
        if emotional_context["emotional_flow"]:
            for flow in emotional_context["emotional_flow"]:
                strategy_parts.append(flow["response"])
        
        return " | ".join(strategy_parts) if strategy_parts else "Balanced approach with clear guidance"

    def _generate_natural_language(self, task):
        """Generate natural language for the response using enhanced emotional analysis"""
        emotional_context = self._analyze_emotional_context(task)
        user_expertise = self._assess_user_expertise(task)
        
        # Build response based on emotional context
        response_parts = []
        
        # Add emotional acknowledgment
        if emotional_context["response"]:
            response_parts.append(emotional_context["response"])
        
        # Add compound emotion responses
        if emotional_context["compound_emotions"]:
            for emotion in emotional_context["compound_emotions"]:
                if emotion in self.emotional_patterns.get("emotional_analysis", {}).get("compound_emotions", {}):
                    response_parts.append(
                        self.emotional_patterns["emotional_analysis"]["compound_emotions"][emotion]["response"]
                    )
        
        # Add expertise-based content
        if user_expertise == "Beginner - Basic query structure":
            content = self._get_pattern("security_domain", "explanations", "beginner")
        else:
            content = self._get_pattern("security_domain", "explanations", "advanced")
        
        # Add content with appropriate tone based on emotional context
        if emotional_context["emotion"] == "frustration":
            response_parts.append("Let me help you with this step by step.")
        elif emotional_context["emotion"] == "urgency":
            response_parts.append("I'll help you address this quickly and effectively.")
        elif emotional_context["emotion"] == "curiosity":
            response_parts.append("I'll explain this in detail to help you understand.")
        elif emotional_context["emotion"] == "concern":
            response_parts.append("Let's address this together to ensure everything is secure.")
        
        # Add context-specific content based on weights
        if emotional_context["context_weights"]:
            max_weight = max(emotional_context["context_weights"].values())
            for ctx_type, weight in emotional_context["context_weights"].items():
                if weight == max_weight:
                    if ctx_type == "security_context":
                        response_parts.append("Let me explain the security implications.")
                    elif ctx_type == "technical_context":
                        response_parts.append("I'll provide technical details and examples.")
                    elif ctx_type == "learning_context":
                        response_parts.append("Let me guide you through this.")
        
        # Add emotional flow responses
        if emotional_context["emotional_flow"]:
            for flow in emotional_context["emotional_flow"]:
                response_parts.append(flow["response"])
        
        return " ".join(response_parts) if response_parts else "I'm here to help you with this."

    def _identify_engagement_opportunities(self, task):
        """Identify opportunities for natural engagement"""
        opportunities = []
        topic = self._extract_topic(task)
        
        # Get topic flow transitions
        for flow in self.patterns["conversation_patterns"]["transitions"]["topic_flow"]:
            if flow["from"] == topic:
                opportunities.extend(flow["patterns"])
        
        # Add expertise-based transitions
        expertise = self._assess_user_expertise(task)
        if expertise == "Beginner - Basic query structure":
            opportunities.extend(self.patterns["conversation_patterns"]["transitions"]["expertise_level"]["beginner"])
        else:
            opportunities.extend(self.patterns["conversation_patterns"]["transitions"]["expertise_level"]["advanced"])
        
        return "\n   - ".join(opportunities)

    def _extract_topic(self, task):
        """Extract the main topic from the task"""
        # Simple topic extraction - can be enhanced
        words = task.lower().split()
        for word in words:
            if word not in ["what", "how", "why", "where", "when", "who", "is", "are", "was", "were"]:
                return word
        return "the topic"

    def _find_related_topic(self, topic):
        """Find a related topic for natural conversation flow"""
        # Simple related topic mapping - can be enhanced
        related_topics = {
            "scanning": "network security",
            "password": "authentication",
            "web": "application security",
            "network": "system security",
            "vulnerability": "penetration testing"
        }
        return related_topics.get(topic, "security best practices")

# Initialize reasoning engine
reasoning_engine = ReasoningEngine() 