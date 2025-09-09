import mlflow
from functools import wraps
from typing import Callable, Optional, Any
import os

# Configuration for which tracing backend to use
TRACING_BACKEND = os.getenv("TRACING_BACKEND", "mlflow")

def generic_trace(name: Optional[str] = None, span_type: Optional[str] = None, **kwargs):
    """
    Generic decorator that wraps different tracing decorators based on configuration.
    
    Args:
        name: Optional name for the trace/span
        span_type: Optional span type (e.g., "TOOL", "CHAIN", "LLM")
        **kwargs: Additional kwargs to pass to the underlying tracing decorator
    
    Returns:
        The appropriate decorator based on TRACING_BACKEND
    """
    def decorator(func: Callable) -> Callable:
        # Apply the appropriate tracing decorator based on backend
        if TRACING_BACKEND == "mlflow":
            # Build kwargs for mlflow.trace
            mlflow_kwargs = {}
            if name:
                mlflow_kwargs["name"] = name
            if span_type:
                mlflow_kwargs["span_type"] = span_type
            # Add any additional kwargs
            mlflow_kwargs.update(kwargs)
            
            # Use mlflow.trace decorator with all kwargs
            traced_func = mlflow.trace(**mlflow_kwargs)(func) if mlflow_kwargs else mlflow.trace(func)
            return traced_func
        elif TRACING_BACKEND == "opentelemetry":
            # Placeholder for OpenTelemetry or other tracing backends
            # You can add support for other tracing libraries here
            @wraps(func)
            def wrapper(*args, **kwargs):
                # For now, just pass through without tracing
                return func(*args, **kwargs)
            return wrapper
        elif TRACING_BACKEND == "none":
            # No tracing - pass through
            return func
        else:
            # Default behavior - pass through
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
    
    # Handle both @generic_trace and @generic_trace() usage
    if callable(name):
        # Called as @generic_trace without parentheses
        func = name
        name = None
        return decorator(func)
    else:
        # Called as @generic_trace() or @generic_trace(name="something", span_type="TOOL")
        return decorator

def start_tracing():
    if TRACING_BACKEND == "mlflow":
        mlflow.openai.autolog()


def log_user_session(user_id, session_id):
    if TRACING_BACKEND == "mlflow":
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.user": user_id,      # Links this trace to a specific user
                "mlflow.trace.session": session_id, # Groups this trace with others in the same conversation
            }
            )