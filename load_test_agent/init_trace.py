import mlflow
from functools import wraps
from typing import Callable, Optional, Any
import os

# Configuration for which tracing backend to use
TRACING_BACKEND = os.getenv("TRACING_BACKEND", "mlflow")

# Try to import OpenTelemetry/Traceloop if available
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from traceloop.sdk import Traceloop
    TRACELOOP_AVAILABLE = True
except ImportError:
    TRACELOOP_AVAILABLE = False
    trace = None
    Traceloop = None

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
        elif TRACING_BACKEND in ["opentelemetry", "traceloop"]:
            # OpenTelemetry/Traceloop tracing
            if not TRACELOOP_AVAILABLE:
                # If Traceloop not installed, pass through
                return func
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get or create a tracer
                tracer = trace.get_tracer(__name__)
                
                # Determine span name
                span_name = name or func.__name__
                
                # Create span attributes
                attributes = {}
                if span_type:
                    attributes["span.type"] = span_type
                # Add any additional attributes from kwargs
                for key, value in kwargs.items():
                    if key not in ["name", "span_type"]:
                        attributes[key] = str(value)
                
                # Start a new span
                with tracer.start_as_current_span(span_name, attributes=attributes) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
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
    """
    Initialize tracing based on the configured backend.
    
    For Traceloop, you can set these environment variables:
    - TRACELOOP_API_KEY: Your Traceloop API key
    - TRACELOOP_BASE_URL: Custom base URL (optional)
    - TRACELOOP_HEADERS: Additional headers as JSON (optional)
    """
    if TRACING_BACKEND == "mlflow":
        mlflow.openai.autolog()
    elif TRACING_BACKEND in ["opentelemetry", "traceloop"]:
        if TRACELOOP_AVAILABLE:
            # Initialize Traceloop with configuration from environment
            api_key = os.getenv("TRACELOOP_API_KEY")
            base_url = os.getenv("TRACELOOP_BASE_URL")
            headers = os.getenv("TRACELOOP_HEADERS")
            
            traceloop_kwargs = {}
            if api_key:
                traceloop_kwargs["api_key"] = api_key
            if base_url:
                traceloop_kwargs["base_url"] = base_url
            if headers:
                import json
                try:
                    traceloop_kwargs["headers"] = json.loads(headers)
                except json.JSONDecodeError:
                    pass
            
            # Initialize Traceloop
            Traceloop.init(**traceloop_kwargs)
            print(f"Traceloop initialized with backend: {TRACING_BACKEND}")
        else:
            print(f"Warning: {TRACING_BACKEND} backend selected but traceloop-sdk not installed. Install with: pip install traceloop-sdk")


def log_user_session(user_id, session_id):
    """
    Log user session information to the current trace.
    
    Args:
        user_id: The user/customer ID
        session_id: The session ID for grouping related traces
    """
    if TRACING_BACKEND == "mlflow":
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.user": user_id,      # Links this trace to a specific user
                "mlflow.trace.session": session_id, # Groups this trace with others in the same conversation
            }
            )
    elif TRACING_BACKEND in ["opentelemetry", "traceloop"]:
        if TRACELOOP_AVAILABLE:
            # Get current span and add attributes
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("user.id", user_id)
                current_span.set_attribute("session.id", session_id)