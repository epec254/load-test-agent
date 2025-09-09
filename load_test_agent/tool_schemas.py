"""
Pydantic models for tool parameters using OpenAI's function calling helpers.
"""
from typing import Literal, Union
from pydantic import BaseModel, Field
from openai.lib._tools import pydantic_function_tool


class GetBillingHistoryParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")


class GetChargeDetailsParams(BaseModel):
    charge_id: str = Field(description="The ID of the charge.")


class GetActivePromotionsParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")


class GetAccountBalanceParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")


class GetSavedPaymentMethodsParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")


class ProcessPaymentParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")
    amount: float = Field(description="The payment amount.")
    payment_method_id: str = Field(description="The ID of the payment method to use.")


class GetNetworkStatusParams(BaseModel):
    location: str = Field(description="The location to check network status for.")


class RunRemoteDeviceDiagnosticsParams(BaseModel):
    device_imei: str = Field(description="The IMEI of the device to run diagnostics on.")


class CreateSupportTicketParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")
    issue_description: str = Field(description="Description of the issue.")


class SearchKnowledgeBaseParams(BaseModel):
    query: str = Field(description="The search query.")


class GetInsuranceCoverageParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")
    device_imei: str = Field(description="The IMEI of the device.")


class ProcessInsuranceClaimParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")
    device_imei: str = Field(description="The IMEI of the device.")
    claim_type: Literal["screen_damage", "water_damage", "theft_loss"] = Field(
        description="The type of claim (e.g., screen_damage, water_damage, theft_loss)."
    )
    incident_description: str = Field(description="Description of the incident.")


class GetDataUsageParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")


class EscalateToHumanAgentParams(BaseModel):
    customer_id: str = Field(description="The ID of the customer.")
    ticket_id: str = Field(description="The ID of the support ticket.")


# Union type of all tool parameter models
ToolParams = Union[
    GetBillingHistoryParams,
    GetChargeDetailsParams,
    GetActivePromotionsParams,
    GetAccountBalanceParams,
    GetSavedPaymentMethodsParams,
    ProcessPaymentParams,
    GetNetworkStatusParams,
    RunRemoteDeviceDiagnosticsParams,
    CreateSupportTicketParams,
    SearchKnowledgeBaseParams,
    GetInsuranceCoverageParams,
    ProcessInsuranceClaimParams,
    GetDataUsageParams,
    EscalateToHumanAgentParams,
]

# Comprehensive model with all possible fields (union of all fields)
class AllToolParams(BaseModel):
    """Union of all possible tool parameters across all tools."""
    # Customer-related fields
    customer_id: str | None = Field(None, description="The ID of the customer.")
    
    # Charge/Payment-related fields
    charge_id: str | None = Field(None, description="The ID of the charge.")
    amount: float | None = Field(None, description="The payment amount.")
    payment_method_id: str | None = Field(None, description="The ID of the payment method to use.")
    
    # Location/Device-related fields
    location: str | None = Field(None, description="The location to check network status for.")
    device_imei: str | None = Field(None, description="The IMEI of the device.")
    
    # Support-related fields
    issue_description: str | None = Field(None, description="Description of the issue.")
    ticket_id: str | None = Field(None, description="The ID of the support ticket.")
    query: str | None = Field(None, description="The search query.")
    
    # Insurance-related fields
    claim_type: Literal["screen_damage", "water_damage", "theft_loss"] | None = Field(
        None, description="The type of claim (e.g., screen_damage, water_damage, theft_loss)."
    )
    incident_description: str | None = Field(None, description="Description of the incident.")


# Generate tool definitions using pydantic_function_tool
TOOL_DEFINITIONS = [
    pydantic_function_tool(GetBillingHistoryParams, name="get_billing_history", description="Retrieves the billing history for a given customer."),
    pydantic_function_tool(GetChargeDetailsParams, name="get_charge_details", description="Retrieves details for a specific charge."),
    pydantic_function_tool(GetActivePromotionsParams, name="get_active_promotions", description="Retrieves active promotions for a given customer."),
    pydantic_function_tool(GetAccountBalanceParams, name="get_account_balance", description="Retrieves the current account balance for a given customer."),
    pydantic_function_tool(GetSavedPaymentMethodsParams, name="get_saved_payment_methods", description="Retrieves the saved payment methods for a given customer."),
    pydantic_function_tool(ProcessPaymentParams, name="process_payment", description="Processes a payment for a customer."),
    pydantic_function_tool(GetNetworkStatusParams, name="get_network_status", description="Retrieves the network status for a given location."),
    pydantic_function_tool(RunRemoteDeviceDiagnosticsParams, name="run_remote_device_diagnostics", description="Runs remote diagnostics on a device."),
    pydantic_function_tool(CreateSupportTicketParams, name="create_support_ticket", description="Creates a support ticket."),
    pydantic_function_tool(SearchKnowledgeBaseParams, name="search_knowledge_base", description="Searches the knowledge base for a given query."),
    pydantic_function_tool(GetInsuranceCoverageParams, name="get_insurance_coverage", description="Retrieves insurance coverage for a device."),
    pydantic_function_tool(ProcessInsuranceClaimParams, name="process_insurance_claim", description="Processes an insurance claim."),
    pydantic_function_tool(GetDataUsageParams, name="get_data_usage", description="Retrieves data usage for a customer."),
    pydantic_function_tool(EscalateToHumanAgentParams, name="escalate_to_human_agent", description="Escalates the issue to a human agent."),
]