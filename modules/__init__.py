"""
Nikita Agent Modules - Core functionality components
"""

from .code_handler import is_python_code, run_python_code
from .command_handler import run_command, save_command_output
from .context_optimizer import ContextOptimizer
from .history_manager import setup_command_history, save_command_history, get_input_with_history, load_chat_history, save_chat_history
from .intent_analyzer import IntentAnalyzer
from .resource_management import get_system_info, get_dynamic_params, optimize_memory_resources, optimize_cpu_usage, prewarm_model
from .engagement_manager import extract_targets, suggest_attack_plan, record_finding, get_engagement_summary, engagement_memory
from .reasoning_engine import ReasoningEngine
from .tool_manager import ToolManager

__all__ = [
    # Code handling
    'is_python_code', 'run_python_code',
    
    # Command handling
    'run_command', 'save_command_output',
    
    # Context optimization
    'ContextOptimizer',
    
    # History management
    'setup_command_history', 'save_command_history', 'get_input_with_history',
    'load_chat_history', 'save_chat_history',
    
    # Intent analysis
    'IntentAnalyzer',
    
    # Resource management
    'get_system_info', 'get_dynamic_params', 'optimize_memory_resources', 
    'optimize_cpu_usage', 'prewarm_model',
    
    # Engagement management
    'extract_targets', 'suggest_attack_plan', 'record_finding',
    'get_engagement_summary', 'engagement_memory',
    
    # Reasoning engine
    'ReasoningEngine',
    
    # Tool manager
    'ToolManager'
] 