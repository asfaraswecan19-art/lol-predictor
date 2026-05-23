import requests
import json

key = input("Paste your API key: ").strip()

response = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    },
    json={
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello"}]
    },
    timeout=30
)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
