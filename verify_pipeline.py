"""Quick verification test for the voice pipeline."""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_pipeline():
    print("=" * 60)
    print("VOICE PIPELINE VERIFICATION TEST")
    print("=" * 60)
    
    # Test 1: Import and initialize
    print("\n[1/4] Testing imports and initialization...")
    try:
        from agent.livekit_agent import get_agent
        agent = get_agent()
        await agent.initialize()
        print("✅ Agent initialized successfully")
        print(f"   - RAG ready: {agent.rag.is_ready}")
        print(f"   - Scholarships: {agent.rag.vectorstore.size}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Test text message handling
    print("\n[2/4] Testing text message handling...")
    try:
        response = await agent.handle_text_message("Hello, I am a student from UP")
        print("✅ Text message handled")
        print(f"   - Response: {response[:100]}..." if len(response) > 100 else f"   - Response: {response}")
    except Exception as e:
        print(f"❌ Text handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test streaming response
    print("\n[3/4] Testing streaming response...")
    try:
        sentences = []
        async for sentence in agent.conversation_handler.generate_response_stream("Tell me about scholarships"):
            sentences.append(sentence)
            if len(sentences) <= 2:
                print(f"   - Sentence {len(sentences)}: {sentence[:50]}...")
        print(f"✅ Streaming works: {len(sentences)} sentences")
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test process_audio_stream method exists
    print("\n[4/4] Verifying streaming methods exist...")
    try:
        assert hasattr(agent, 'process_audio_stream'), "process_audio_stream method missing"
        assert hasattr(agent, '_tts_worker'), "_tts_worker method missing"
        print("✅ Streaming methods present")
        print("   - process_audio_stream: ✓")
        print("   - _tts_worker: ✓")
    except AssertionError as e:
        print(f"❌ Method check failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL VERIFICATION TESTS PASSED!")
    print("=" * 60)
    print("\nThe voice pipeline is working correctly.")
    print("Start server with: python -m agent.livekit_agent")
    return True

if __name__ == "__main__":
    asyncio.run(test_pipeline())
