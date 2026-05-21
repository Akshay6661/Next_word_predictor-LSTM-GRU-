# Call Claude 3.5 Sonnet on Bedrock
def invoke_claude(prompt, max_tokens=1000):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = client.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps(body)
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# Test
print(invoke_claude("Summarize what pharmacovigilance means in 2 sentences."))
