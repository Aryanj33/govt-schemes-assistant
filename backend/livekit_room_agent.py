"""
Standalone LiveKit Room Agent (Python 3.9 compatible)
=====================================================
Joins the LiveKit room as an AI participant.
Listens to user audio → STT (Groq) → LLM (Groq/Gemini) → TTS (EdgeTTS) → publishes back audio.

Uses the base `livekit` SDK only (no livekit-agents), which works on Python 3.9.

Run alongside main.py:
    python livekit_room_agent.py
"""

import asyncio
import io
import struct
import sys
import wave
import logging
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging, get_logger
from utils.config import get_config

config = get_config()
setup_logging(level=config.log_level)
logger = get_logger()

# ── LiveKit SDK ──────────────────────────────────────────────────────────────
try:
    from livekit import rtc, api as lk_api
except ImportError:
    print("❌ livekit package not installed. Run: pip install livekit")
    sys.exit(1)


# ── VAD constants ───────────────────────────────────────────────────────────
SAMPLE_RATE       = 48000        # LiveKit default (Hz)
CHANNELS          = 1
SPEECH_RMS_THRESH = 800          # 16-bit PCM RMS threshold for speech (0-32767 range)
SILENCE_THRESH_S  = 0.6          # seconds of silence = end of utterance
FRAME_MS          = 10           # LiveKit audio frame ~10ms
SILENCE_FRAMES    = int(SILENCE_THRESH_S * 1000 / FRAME_MS)   # 60 frames
MIN_SPEECH_S      = 0.4          # minimum utterance length to bother processing
MIN_SPEECH_BYTES  = int(SAMPLE_RATE * 2 * MIN_SPEECH_S)       # bytes


def pcm_rms(pcm: bytes) -> float:
    """Compute RMS energy of 16-bit signed mono PCM."""
    if len(pcm) < 2:
        return 0.0
    # Unpack every 4th sample to save CPU (still accurate enough for VAD)
    step = 4
    num_samples = len(pcm) // 2
    indices = range(0, num_samples, step)
    samples = struct.unpack_from(f'{len(indices)}h',
                                 b''.join(pcm[i*2:i*2+2] for i in indices))
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> bytes:
    """Wrap raw 16-bit PCM bytes into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def publish_audio(local_participant: rtc.LocalParticipant, audio_bytes: bytes):
    """Publish MP3/PCM bytes back to the room as a LocalAudioTrack."""
    try:
        import pydub
        seg = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_channels(1).set_sample_width(2).set_frame_rate(48000)
        pcm_data = seg.raw_data
    except Exception:
        # Fallback: assume it's already 48kHz 16-bit mono PCM
        pcm_data = audio_bytes

    source = rtc.AudioSource(48000, 1)
    track  = rtc.LocalAudioTrack.create_audio_track("vidya-voice", source)

    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    pub  = await local_participant.publish_track(track, opts)
    logger.info(f"📤 Publishing audio track: {pub.sid}")

    chunk_size = 48000 * 2 // 50  # 20ms chunks
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk.ljust(chunk_size, b"\x00")
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=chunk_size // 2,
        )
        await source.capture_frame(frame)
        await asyncio.sleep(0.018)  # ~20ms between frames

    await local_participant.unpublish_track(pub.sid)
    logger.info("✅ Audio published and track released")


async def handle_participant_audio(
    participant: rtc.RemoteParticipant,
    track: rtc.Track,
    local_participant: rtc.LocalParticipant,
    agent,
):
    """Stream audio from a participant, detect speech end, and respond."""
    logger.info(f"🎤 Listening to {participant.identity}")

    audio_stream  = rtc.AudioStream(track)
    buffer        = bytearray()
    is_speaking   = False
    silence_count = 0
    frame_count   = 0

    async for event in audio_stream:
        frame: rtc.AudioFrame = event.frame
        pcm = bytes(frame.data)

        # Correctly decode 16-bit signed PCM for RMS
        rms = pcm_rms(pcm)
        frame_count += 1

        # Log RMS every 200 frames (~2s) so we can tune threshold
        if frame_count % 200 == 0:
            logger.debug(f"[VAD] rms={rms:.0f} speaking={is_speaking} buf={len(buffer)//1024}KB")

        if rms > SPEECH_RMS_THRESH:
            if not is_speaking:
                logger.info(f"🗣  Speech start (rms={rms:.0f})")
            is_speaking   = True
            silence_count = 0
            buffer.extend(pcm)

        elif is_speaking:
            silence_count += 1
            buffer.extend(pcm)   # keep trailing silence for natural end

            if silence_count >= SILENCE_FRAMES:
                if len(buffer) >= MIN_SPEECH_BYTES:
                    logger.info(f"⏹  Speech end — {len(buffer)/1024:.1f} KB → pipeline")
                    wav_data = pcm_to_wav(bytes(buffer))
                    try:
                        response_audio = await agent.process_audio(wav_data)
                        if response_audio:
                            await publish_audio(local_participant, response_audio)
                        else:
                            logger.warning("⚠️  No audio response from pipeline")
                    except Exception as e:
                        logger.error(f"❌ Pipeline error: {e}", exc_info=True)
                else:
                    logger.debug(f"[VAD] utterance too short ({len(buffer)} bytes), discarding")

                buffer        = bytearray()
                is_speaking   = False
                silence_count = 0

        else:
            # Complete silence — not accumulating
            pass


async def run_room_agent():
    """Main agent loop — joins the LiveKit room and waits for participants."""
    from agent.livekit_agent import get_agent

    logger.info("🚀 Vidya LiveKit Room Agent starting…")

    agent = get_agent()
    await agent.initialize()
    logger.info("✅ AI pipeline initialised")

    # Generate an agent token
    token = (
        lk_api.AccessToken(config.livekit.api_key, config.livekit.api_secret)
        .with_identity("vidya-agent")
        .with_name("Vidya AI")
        .with_grants(lk_api.VideoGrants(
            room_join=True,
            room=config.livekit.room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        .to_jwt()
    )

    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity == "vidya-agent":
            return          # ignore our own audio
        asyncio.ensure_future(
            handle_participant_audio(participant, track, room.local_participant, agent)
        )

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"👤 Participant joined: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"👤 Participant left: {participant.identity}")
        agent.reset_session()

    # Connect to room
    lk_url = config.livekit.url
    print(f"[agent] Connecting to LiveKit at {lk_url}, room={config.livekit.room_name}")
    logger.info(f"🌐 Connecting to LiveKit at {lk_url}, room={config.livekit.room_name}")

    await room.connect(lk_url, token)
    print(f"[agent] ✅ Joined room: {room.name}")
    logger.info(f"✅ Agent joined room: {room.name}")

    # Stay alive until room is active
    try:
        while True:
            await asyncio.sleep(5)
            if room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
                break
    except asyncio.CancelledError:
        pass
    finally:
        await room.disconnect()
        logger.info("🔌 Agent disconnected")


if __name__ == "__main__":
    try:
        asyncio.run(run_room_agent())
    except KeyboardInterrupt:
        logger.info("👋 Agent stopped")
