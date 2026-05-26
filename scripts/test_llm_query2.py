#!/usr/bin/env python3
"""Debug: check full raw response."""
import os, requests, json

API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-5bN2WCRK0PEMBRHfirSvNQTxAR0GQwhFouumdzcVBCKnqkf5TCyy1dyAOuzx2Ppk")
API_BASE = "https://opencode.ai/zen/go/v1"
MODEL = "deepseek-v4-flash"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Try with larger max_tokens and no system prompt
payload1 = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Answer ONLY with a single number between 0.0 and 1.0: what is your confidence that gene A activates gene B? Just the number."}
    ],
    "temperature": 0.0,
    "max_tokens": 200,
}
resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload1, timeout=30)
print(f"=== Test A: Large max_tokens ===")
print(f"Status: {resp.status_code}")
data = resp.json()
msg = data["choices"][0]["message"]
print(f"All keys in message: {list(msg.keys())}")
print(f"Content: '{msg.get('content', 'MISSING')}'")
print(f"Reasoning: {msg.get('reasoning_content', 'MISSING')}")
print(f"Full message: {json.dumps(msg, indent=2)[:500]}")
print(f"Usage: {json.dumps(data.get('usage', {}))}")

# Try with thinking parameter
print(f"\n=== Test B: With thinking param ===")
payload2 = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Output ONLY a number 0.0 to 1.0 for: confidence X causes Y. Just the number."}
    ],
    "temperature": 0.0,
    "max_tokens": 200,
    "thinking": {"type": "disabled"}
}
resp2 = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload2, timeout=30)
print(f"Status: {resp2.status_code}")
data2 = resp2.json()
msg2 = data2["choices"][0]["message"]
print(f"Content: '{msg2.get('content', 'MISSING')}'")
if resp2.status_code != 200:
    print(f"Error: {resp2.text[:500]}")
