"""Quick TTS test"""
import asyncio
import httpx
import base64

async def test():
    api_key = input("Enter your GOOGLE_API_KEY: ").strip()
    
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    
    payload = {
        "input": {"text": "Namaste! Main Vidya hoon."},
        "voice": {"languageCode": "hi-IN", "name": "hi-IN-Wavenet-A"},
        "audioConfig": {"audioEncoding": "MP3"}
    }
    
    async with httpx.AsyncClient() as client:
        print("Testing Google TTS API...")
        response = await client.post(url, json=payload)
        
        if response.status_code == 200:
            audio_base64 = response.json().get("audioContent")
            if audio_base64:
                audio_bytes = base64.b64decode(audio_base64)
                with open("test_audio.mp3", "wb") as f:
                    f.write(audio_bytes)
                print(f"✅ SUCCESS! Audio saved to test_audio.mp3 ({len(audio_bytes)} bytes)")
                print("Play the file to hear the TTS output!")
            else:
                print("❌ No audio content returned")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

asyncio.run(test())
