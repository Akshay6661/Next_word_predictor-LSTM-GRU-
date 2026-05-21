import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY"
)


import boto3
session = boto3.session.Session()
print("Region:",  session.region_name)
print("Profile:", session.profile_name)
credentials = session.get_credentials()
print("Creds found:", credentials is not None)


============================================================
Region: us-east-1
============================================================
ACTIVE     | anthropic.claude-sonnet-4-20250514-v1:0
ACTIVE     | anthropic.claude-opus-4-20250514-v1:0
ACTIVE     | anthropic.claude-3-haiku-20240307-v1:0
LEGACY     | anthropic.claude-3-5-sonnet-20241022-v2:0
...

============================================================
Region: ap-south-1
============================================================
Error: ... (likely no Bedrock support here)
