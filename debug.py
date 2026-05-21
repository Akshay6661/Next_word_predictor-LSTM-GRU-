from langchain_aws import ChatBedrock
import boto3

# Option A — Explicit credentials in code (quick test)
llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    client=boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="AKIA...",        # ← your Access Key ID
        aws_secret_access_key="abc123...",  # ← your Secret Access Key
    )
)
