
import os
import sys
from pathlib import Path

# Add backend to path to import config
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config
from twilio.rest import Client

def make_call(to_number):
    config = get_config()
    
    if not config.twilio.is_configured():
        print("❌ Twilio is not configured in .env")
        return

    client = Client(config.twilio.account_sid, config.twilio.auth_token)
    
    print(f"📞 Initiating call to {to_number}...")
    print(f"   From: {config.twilio.phone_number}")
    print(f"   Webhook: https://subsplenial-hurtless-waltraud.ngrok-free.dev/twilio/voice")
    
    try:
        call = client.calls.create(
            url='https://subsplenial-hurtless-waltraud.ngrok-free.dev/twilio/voice',
            to=to_number,
            from_=config.twilio.phone_number
        )
        print(f"✅ Call initiated! SID: {call.sid}")
    except Exception as e:
        print(f"❌ Failed to make call: {e}")

if __name__ == "__main__":
    to_number = "+918950678427"
    make_call(to_number)
