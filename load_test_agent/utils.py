"""
Utility functions for the load test agent.
"""
import inspect
from typing import get_origin, get_args, Union, Optional, Literal, Any
from types import NoneType


def function_to_schema(func) -> dict:
    """
    Convert a Python function to an OpenAI function schema.
    
    Enhanced version based on OpenAI build-hours demo_util.py with support for:
    - Optional parameters
    - Union types
    - Literal types (for enums)
    - Better descriptions from docstrings
    """
    # Basic type mapping
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
        NoneType: "null",
    }
    
    def get_json_type(annotation):
        """Convert Python type annotation to JSON Schema type."""
        # Handle None and NoneType
        if annotation is None or annotation is NoneType:
            return "null"
        
        # Handle Optional types (Union with None)
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            # Check if it's Optional (Union with None)
            non_none_args = [arg for arg in args if arg is not NoneType and arg is not type(None)]
            if len(non_none_args) == 1:
                return get_json_type(non_none_args[0])
            # For other unions, default to string
            return "string"
        
        # Handle Literal types
        if origin is Literal:
            return "string"  # Literals are typically strings in our case
        
        # Direct type mapping
        if annotation in type_map:
            return type_map[annotation]
        
        # Default to string for unknown types
        return "string"
    
    def get_enum_values(annotation):
        """Extract enum values from Literal type."""
        origin = get_origin(annotation)
        if origin is Literal:
            return list(get_args(annotation))
        # Check if it's Optional[Literal]
        if origin is Union:
            args = get_args(annotation)
            for arg in args:
                if get_origin(arg) is Literal:
                    return list(get_args(arg))
        return None
    
    def parse_docstring(docstring):
        """Parse docstring to extract parameter descriptions."""
        if not docstring:
            return {}, ""
        
        lines = docstring.strip().split('\n')
        main_description = []
        param_descriptions = {}
        in_args_section = False
        
        for line in lines:
            line = line.strip()
            if line.lower() in ['args:', 'arguments:', 'parameters:']:
                in_args_section = True
                continue
            elif line.lower() in ['returns:', 'return:', 'raises:', 'example:', 'examples:']:
                in_args_section = False
                continue
            
            if in_args_section and ':' in line:
                # Parse parameter description
                param_part = line.split(':', 1)
                if len(param_part) == 2:
                    param_name = param_part[0].strip()
                    param_desc = param_part[1].strip()
                    param_descriptions[param_name] = param_desc
            elif not in_args_section and line and not line.startswith(' '):
                main_description.append(line)
        
        return param_descriptions, ' '.join(main_description)
    
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )
    
    # Parse docstring for descriptions
    param_descriptions, main_description = parse_docstring(func.__doc__)
    
    parameters = {}
    required = []
    
    for param in signature.parameters.values():
        # Skip 'self' parameter for methods
        if param.name == 'self':
            continue
            
        param_info = {}
        
        # Get the type
        if param.annotation != inspect._empty:
            param_info["type"] = get_json_type(param.annotation)
            
            # Check for enum values (Literal types)
            enum_values = get_enum_values(param.annotation)
            if enum_values:
                param_info["enum"] = enum_values
        else:
            # Default to string if no type annotation
            param_info["type"] = "string"
        
        # Add description if available
        if param.name in param_descriptions:
            param_info["description"] = param_descriptions[param.name]
        elif param_info.get("enum"):
            # Add enum values to description if no explicit description
            param_info["description"] = f"One of: {', '.join(param_info['enum'])}"
        
        parameters[param.name] = param_info
        
        # Check if parameter is required (no default value)
        if param.default == inspect._empty:
            required.append(param.name)
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": main_description or func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }