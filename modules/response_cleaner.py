#!/usr/bin/env python3
"""
Response Cleaner Module for Nikita Agent

Cleans and formats LLM responses to ensure they are human-readable
and free from metadata, JSON artifacts, and other unwanted content.
"""

import re
import json
from typing import Dict, Any, List, Union, Optional

class ResponseCleaner:
    """
    Cleans and formats raw LLM responses to ensure they are human-readable
    and consistent with Nikita's expected output format.
    """
    
    def __init__(self):
        """Initialize the response cleaner"""
        self.json_pattern = re.compile(r'^\s*\{.*\}\s*$', re.DOTALL)
        self.command_pattern = re.compile(r'```(?:\w+)?\s*([^`]+)```|`([^`]+)`')
        self.metadata_sections = [
            "response_strategy", "execution_plan", "context", "reasoning",
            "technical_context", "emotional_context", "personal_context",
            "domain", "intent", "answered_context", "follow_up_questions"
        ]
        
    def clean_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Clean and format a raw LLM response.
        
        Args:
            raw_response (str): Raw response from the LLM
            
        Returns:
            dict: Dictionary containing cleaned response and extracted metadata
                - 'clean_text': The cleaned response text
                - 'commands': List of extracted commands
                - 'metadata': Any extracted metadata
        """
        if not raw_response:
            return {
                'clean_text': "I apologize, but I couldn't generate a response. Please try again.",
                'commands': [],
                'metadata': {}
            }
            
        # Try to parse as JSON first
        json_data = self._extract_json(raw_response)
        
        if json_data:
            return self._process_json_response(json_data)
        else:
            return self._process_text_response(raw_response)
            
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from text if present"""
        if self.json_pattern.match(text):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
                
        # Try to find JSON within text
        try:
            # Look for JSON-like structures
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
            
        return None
        
    def _process_json_response(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON-formatted response"""
        clean_text = ""
        metadata = {}
        commands = []
        
        # Extract the main response text
        if 'response' in json_data and isinstance(json_data['response'], dict):
            if 'text' in json_data['response']:
                clean_text = json_data['response']['text']
            
            # Move context to metadata
            if 'context' in json_data['response']:
                metadata['context'] = json_data['response']['context']
                
        elif 'text' in json_data:
            clean_text = json_data['text']
        elif 'output' in json_data:
            clean_text = json_data['output']
        elif 'content' in json_data:
            clean_text = json_data['content']
        elif 'message' in json_data:
            clean_text = json_data['message']
        elif 'answer' in json_data:
            clean_text = json_data['answer']
        
        # If we still don't have clean text, use the first string value we find
        if not clean_text:
            for key, value in json_data.items():
                if isinstance(value, str) and len(value) > 20:
                    clean_text = value
                    break
        
        # If still no clean text, use the entire JSON as a fallback
        if not clean_text:
            clean_text = "I apologize, but I couldn't format my response properly. Here's the raw data:\n\n" + json.dumps(json_data, indent=2)
            
        # Extract metadata
        for section in self.metadata_sections:
            if section in json_data:
                metadata[section] = json_data[section]
                
        # Extract commands from the clean text
        commands = self._extract_commands(clean_text)
        
        return {
            'clean_text': clean_text,
            'commands': commands,
            'metadata': metadata
        }
        
    def _process_text_response(self, text: str) -> Dict[str, Any]:
        """Process a plain text response"""
        # Split into lines for processing
        lines = text.split('\n')
        clean_lines = []
        metadata = {}
        skip_section = False
        current_section = None
        section_content = []
        
        # Process line by line
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            section_match = re.match(r'^#+\s*(.*?)\s*:?\s*$', line)
            if section_match:
                section_name = section_match.group(1).lower()
                
                # If ending a metadata section, store it
                if current_section in self.metadata_sections:
                    metadata[current_section] = '\n'.join(section_content)
                    section_content = []
                
                # Check if this is a metadata section
                if any(meta in section_name for meta in self.metadata_sections):
                    current_section = section_name
                    skip_section = True
                    continue
                else:
                    current_section = None
                    skip_section = False
            
            # If in a metadata section, collect content
            if skip_section or current_section in self.metadata_sections:
                section_content.append(line)
                continue
                
            # Skip JSON-like lines
            if line.startswith('{') and line.endswith('}'):
                continue
                
            # Skip empty lines at the beginning
            if not clean_lines and not line:
                continue
                
            # Add the line to clean lines
            clean_lines.append(line)
            
        # Store the last section if it was metadata
        if current_section in self.metadata_sections:
            metadata[current_section] = '\n'.join(section_content)
            
        # Join clean lines back together
        clean_text = '\n'.join(clean_lines)
        
        # Extract commands
        commands = self._extract_commands(clean_text)
        
        return {
            'clean_text': clean_text,
            'commands': commands,
            'metadata': metadata
        }
        
    def _extract_commands(self, text: str) -> List[str]:
        """Extract commands from code blocks in text"""
        commands = []
        
        # Find all code blocks
        matches = self.command_pattern.findall(text)
        for match in matches:
            # Each match is a tuple with groups from the regex
            command = match[0] if match[0] else match[1]
            if command:
                commands.append(command.strip())
                
        return commands
        
    def format_for_display(self, cleaned_response: Dict[str, Any]) -> str:
        """
        Format the cleaned response for display to the user.
        
        Args:
            cleaned_response (dict): The cleaned response dictionary
            
        Returns:
            str: Formatted text ready for display
        """
        return cleaned_response['clean_text']


if __name__ == "__main__":
    # Simple self-test
    cleaner = ResponseCleaner()
    
    # Test with JSON response
    json_response = '''
    {
      "response_strategy": {
        "approach": "informative",
        "tone": "helpful",
        "technical_level": "moderate",
        "follow_up_questions": []
      },
      "execution_plan": {
        "steps": [
          "understand query",
          "provide information",
          "ask clarifying questions"
        ],
        "priority": "normal",
        "dependencies": [],
        "command": null
      },
      "response": {
        "text": "To attack a target like google.com, you need to understand the scope of your operations and obtain explicit authorization. Once you have that in place, you can start gathering information about the target.",
        "context": {
          "domain": "general",
          "intent": "information_request",
          "personal_context": null,
          "technical_context": {
            "task_type": "targeted attack",
            "target": "google.com"
          },
          "emotional_context": null,
          "answered_context": null
        }
      }
    }
    '''
    
    cleaned = cleaner.clean_response(json_response)
    print("Cleaned JSON Response:")
    print(cleaner.format_for_display(cleaned))
    print("\nExtracted Commands:", cleaned['commands'])
    print("\nMetadata:", cleaned['metadata'])
    
    # Test with text response
    text_response = '''
    # Response Strategy
    This is a helpful response
    
    To use nmap for scanning, you can run:
    ```
    nmap -sS -p 1-1000 192.168.1.1
    ```
    
    Or for a simple ping:
    `ping -c 4 google.com`
    
    # Technical Context
    This is some technical context
    '''
    
    cleaned = cleaner.clean_response(text_response)
    print("\n\nCleaned Text Response:")
    print(cleaner.format_for_display(cleaned))
    print("\nExtracted Commands:", cleaned['commands'])
    print("\nMetadata:", cleaned['metadata'])
