import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": "Say hello"}
    ]
})

try:
    response = client.invoke_model(
        modelId="anthropic.claude-sonnet-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )
    result = json.loads(response["body"].read())
    print(result["content"][0]["text"])

except Exception as e:
    print(type(e).__name__, ":", e)
