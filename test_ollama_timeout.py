#!/usr/bin/env python3
"""
Test script to verify OLLAMA timeout and streaming fixes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.ollama_client import OllamaClient
import time
import logging

# Set up logging to see detailed output
logging.basicConfig(level=logging.INFO)

def test_ollama_timeout():
    """Test OLLAMA timeout and streaming functionality"""
    print("=== TESTING OLLAMA TIMEOUT AND STREAMING ===")
    
    client = OllamaClient()
    
    print(f"Timeout setting: {client.timeout} seconds")
    print(f"Streaming enabled: {client.use_streaming}")
    print(f"Max retries: {client.max_retries}")
    
    # Test connection first
    print("\n1. Testing connection...")
    if client.check_connection():
        print("   ✅ OLLAMA connection successful")
    else:
        print("   ❌ OLLAMA connection failed")
        return
    
    # Test simple prompt with streaming
    print("\n2. Testing simple prompt with streaming...")
    start_time = time.time()
    
    simple_prompt = "What is 2+2? Please answer in one sentence."
    response = client.generate_response(simple_prompt, use_stream=True)
    
    elapsed = time.time() - start_time
    
    if response:
        print(f"   ✅ Streaming response received in {elapsed:.2f} seconds")
        print(f"   Response: '{response[:100]}...'")
    else:
        print(f"   ❌ No response received after {elapsed:.2f} seconds")
    
    # Test simple prompt with blocking
    print("\n3. Testing simple prompt with blocking...")
    start_time = time.time()
    
    response = client.generate_response(simple_prompt, use_stream=False)
    
    elapsed = time.time() - start_time
    
    if response:
        print(f"   ✅ Blocking response received in {elapsed:.2f} seconds")
        print(f"   Response: '{response[:100]}...'")
    else:
        print(f"   ❌ No response received after {elapsed:.2f} seconds")
    
    # Test longer prompt that might timeout
    print("\n4. Testing complex financial analysis prompt...")
    start_time = time.time()
    
    complex_prompt = """
    You are a financial analyst. Provide a detailed analysis of Apple Inc (AAPL) considering:
    1. Current market conditions
    2. Technical indicators
    3. Fundamental analysis
    4. Risk factors
    5. Investment recommendation
    
    Please provide a comprehensive 5-paragraph analysis with specific details.
    """
    
    response = client.generate_response(complex_prompt)
    
    elapsed = time.time() - start_time
    
    if response:
        print(f"   ✅ Complex analysis received in {elapsed:.2f} seconds")
        print(f"   Response length: {len(response)} characters")
        print(f"   First 200 chars: '{response[:200]}...'")
        
        if elapsed > 30:
            print(f"   🎯 SUCCESS: Response took {elapsed:.2f}s, longer than old 30s timeout!")
        else:
            print(f"   ℹ️ Response completed quickly in {elapsed:.2f}s")
    else:
        print(f"   ❌ No response received after {elapsed:.2f} seconds")
        if elapsed >= 120:
            print("   ⚠️ Hit new 2-minute timeout limit")
        else:
            print("   ⚠️ Failed before timeout limit")

if __name__ == "__main__":
    test_ollama_timeout()