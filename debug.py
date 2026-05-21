import boto3

# Try each region where Bedrock is available
for region in ["us-east-1", "us-west-2", "ap-south-1"]:
    print(f"\n{'='*60}")
    print(f"Region: {region}")
    print('='*60)
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        models = bedrock.list_foundation_models(byProvider="Anthropic")
        for m in models["modelSummaries"]:
            status = m.get("modelLifecycle", {}).get("status", "unknown")
            print(f"{status:10s} | {m['modelId']}")
    except Exception as e:
        print(f"Error: {e}")
