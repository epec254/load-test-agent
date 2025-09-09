from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task, agent, tool
from traceloop.sdk.tracing import get_tracer
from traceloop.sdk.tracing.tracing import set_association_properties
import os 
from databricks.sdk import WorkspaceClient

databricks_workspace_client = WorkspaceClient()
auth_headers = databricks_workspace_client.config.authenticate()

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry Tracer
# provider = TracerProvider()

os.environ['TRACELOOP_BASE_URL']="https://opentelemetry-collector-app-8544796052846287.aws.databricksapps.com/v1/traces"

# Create OTLP exporter with debug info
otlp_exporter = OTLPSpanExporter(
    endpoint=os.environ['TRACELOOP_BASE_URL'],
    headers={
        "content-type": "application/x-protobuf",
        # Retrieve the Databricks OAuth token from the Databricks Workspace Client
        # and set it as a header
        **auth_headers
    }
)



Traceloop.init(exporter=otlp_exporter)

@workflow(name="suggest_answers")
def suggest_answers(question: str):
  return ''

# Works seamlessly with async functions too
@workflow(name="summarize")
def summarize(long_text: str):
  return ''


summarize(long_text="sdfdsfds")