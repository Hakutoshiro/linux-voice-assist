import asyncio
import edge_tts
from pydub import AudioSegment
import tempfile
import subprocess

VOICE = "en-GB-SoniaNeural"

async def generate_audio(text: str, file_path: str) -> None:
    """Generate audio file for the given text."""
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(file_path)

def text_to_speech(text: str) -> None:
    """Generate speech from text and play it in a separate process."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        asyncio.run(generate_audio(text, temp_audio_file.name))
        # Play the file using a subprocess, isolating playback
        subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_audio_file.name], check=True)

# Example usage
if __name__ == "__main__":
    text_to_speech("Hello World!")