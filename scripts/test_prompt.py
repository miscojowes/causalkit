#!/usr/bin/env python3
"""Test different prompt styles to get meaningful causal priors."""
import os, requests, json

API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-5bN2WCRK0PEMBRHfirSvNQTxAR0GQwhFouumdzcVBCKnqkf5TCyy1dyAOuzx2Ppk")
API_BASE = "https://opencode.ai/zen/go/v1"
MODEL = "deepseek-v4-flash"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

tests = [
    # Style 1: With thinking, single number
    {
        "name": "With thinking, single number",
        "payload": {
            "model": MODEL,
            "messages": [{"role": "user", "content": "In a biological signaling pathway where X0 is a transcription factor and X1 is a kinase it regulates: rate confidence (0.0-1.0) that X0 causes X1. Output ONLY the number."}],
            "temperature": 0.0,
            "max_tokens": 300,
        }
    },
    # Style 2: Without thinking, single number  
    {
        "name": "No thinking, single number",
        "payload": {
            "model": MODEL,
            "messages": [{"role": "user", "content": "In a biological signaling pathway where X0 is a transcription factor and X1 is a kinase it regulates: rate confidence (0.0-1.0) that X0 causes X1. Output ONLY the number."}],
            "temperature": 0.0,
            "max_tokens": 300,
            "thinking": {"type": "disabled"}
        }
    },
    # Style 3: Multiple edges, no thinking
    {
        "name": "Multiple edges, no thinking",
        "payload": {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": """Domain: Biological signaling pathway.
X0 = TranscriptionFactorA (upstream regulator)
X1 = KinaseB (activated by X0)
X2 = PhosphataseC (activated by X0)
X3 = ResponseProteinD (integrates X1 and X2)
X4 = EffectorE (downstream of X3)
X5 = OutputF (downstream of X4)

Rate confidence (0.0 to 1.0) for each causal relationship.
Output ONLY numbers, one per line, in order:

X0→X1:
X0→X2:
X1→X3:
X2→X3:
X3→X4:
X4→X5:
X0→X5:
"""
            }],
            "temperature": 0.0,
            "max_tokens": 300,
            "thinking": {"type": "disabled"}
        }
    },
]

for test in tests:
    print(f"\n=== {test['name']} ===")
    resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=test["payload"], timeout=30)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    print(f"Content: '{content}'")
    if reasoning:
        print(f"Reasoning (first 100): {reasoning[:100]}")
    print(f"Usage: {data['usage']['completion_tokens']} total, {data['usage'].get('completion_tokens_details', {}).get('reasoning_tokens', 0)} reasoning")
