from dataclasses import dataclass, field
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage
import boto3, os

@dataclass
class BedrockRequest:
    model:            str   = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    system_prompt:    str   = "You are a pharmacovigilance expert."
    input:            str   = ""   # ← user input goes here
    inference_config: dict  = field(default_factory=lambda: {
        "max_tokens": 1000,
        "temperature": 0.0,
        "top_p": 0.9
    })

    def get_response(self) -> str:
        """Send input and return model response."""
        llm = ChatBedrockConverse(
            model=self.model,
            region_name="us-east-1",
            max_tokens=self.inference_config["max_tokens"],
            temperature=self.inference_config["temperature"],
            top_p=self.inference_config["top_p"]
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.input)
        ]

        response = llm.invoke(messages)
        return response.content




# Simple call
req = BedrockRequest(input="What is an adverse event?")
print(req.get_response())

# Override anything as needed
req = BedrockRequest(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    system_prompt="You are a MedDRA coding expert. Return only LLT codes.",
    input="Patient reported stomach pain and nausea after taking Drug X",
    inference_config={"max_tokens": 500, "temperature": 0.0, "top_p": 0.9}
)
print(req.get_response())
