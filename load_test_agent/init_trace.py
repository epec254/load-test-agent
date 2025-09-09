import mlflow
from functools import wraps
from typing import Callable, Optional, Any
import os

# Configuration for which tracing backend to use
TRACING_BACKEND = os.getenv("TRACING_BACKEND", "mlflow")

print(TRACING_BACKEND)

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task, agent, tool
from traceloop.sdk.tracing import get_tracer
from traceloop.sdk.tracing.tracing import set_association_properties
from databricks.sdk import WorkspaceClient
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


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
         
            print('dfdsfds')
            # Use Traceloop's specific decorators based on span_type
            traceloop_name = name or func.__name__
            
            if span_type == "TOOL":
                # Use @tool decorator for tools
                return tool(name=traceloop_name)(func)
            elif span_type == "AGENT":
                # Use @agent decorator for agents
                return agent(name=traceloop_name)(func)
            elif span_type == "TASK":
                # Use @task decorator for tasks
                return task(name=traceloop_name)(func)
            elif span_type in ["WORKFLOW", "CHAIN"]:
                # Use @workflow decorator for workflows/chains
                return workflow(name=traceloop_name)(func)
            else:
                # Default to workflow decorator for general tracing
                return workflow(name=traceloop_name)(func)
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

        os.environ['TRACELOOP_BASE_URL']="https://opentelemetry-collector-app-8544796052846287.aws.databricksapps.com/v1/traces"
        # Set up Databricks Workspace Client with OAuth authentication
        databricks_workspace_client = WorkspaceClient()

        # Set up OpenTelemetry Tracer
        # provider = TracerProvider()
        exporter = OTLPSpanExporter(
            headers={
                "content-type": "application/x-protobuf",
                # Retrieve the Databricks OAuth token from the Databricks Workspace Client
                # and set it as a header
                **databricks_workspace_client.config.authenticate()
            },
            endpoint=os.environ['TRACELOOP_BASE_URL']
        )
        # provider.add_span_processor(BatchSpanProcessor()
        # trace.set_tracer_provider(provider)
        # tracer = trace.get_tracer(__name__)

        
    
        # print('jjj')
        # # Initialize Traceloop with configuration from environment
        

        # print(os.environ['TRACELOOP_BASE_URL'])
        
        # traceloop_kwargs = {}
        # if api_key:
        #     traceloop_kwargs["api_key"] = api_key
        # if base_url:
        #     traceloop_kwargs["base_url"] = base_url
        # if headers:
        #     import json
        #     try:
        #         traceloop_kwargs["headers"] = json.loads(headers)
        #     except json.JSONDecodeError:
        #         pass
        
        # Initialize Traceloop
        
        Traceloop.init(exporter=exporter, disable_batch=True)


        print(f"Traceloop initialized with backend: {TRACING_BACKEND}")


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
        
        set_association_properties({
            "user_id": user_id,
            "session_id": session_id,
            "chat_id": session_id  # chat_id can be same as session_id for conversations
        })