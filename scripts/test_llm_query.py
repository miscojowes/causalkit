#!/usr/bin/env python3
"""Debug: test what the LLM actually returns."""
import os, requests, json

API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-5bN2WCRK0PEMBRHfirSvNQTxAR0GQwhFouumdzcVBCKnqkf5TCyy1dyAOuzx2Ppk")
API_BASE = "https://opencode.ai/zen/go/v1"
MODEL = "deepseek-v4-flash"

# Test: ask about a single edge with forced JSON output
prompts = [
    # Test 1: Simple direct question
    "Is there a causal relationship from X0 to X1 in a biological signaling pathway? "
    "Answer with a single number from 0.0 to 1.0 representing your confidence. "
    "Only output the number, nothing else.",
    
    # Test 2: With forced format
    "Rate your confidence (0.0 to 1.0) that X3 causes X4 in a signaling cascade. "
    "Respond ONLY with the number.",

    # Test 3: JSON
    '{"task": "rate causal confidence", "cause": "TranscriptionFactor", "effect": "KinaseA", "confidence": ',
]

for i, prompt in enumerate(prompts):
    print(f"\n=== Test {i+1} ===")
    print(f"Prompt: {prompt[:80]}...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a causal inference expert. Answer with extreme conciseness."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    
    resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=30)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"Response: '{content}'")
        print(f"Token usage: {data.get('usage', {})}")
    else:
        print(f"Error: {resp.text[:200]}")
