import os
import sys
import logging
import json
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import sounddevice as sd
from scipy.io import wavfile
from faster_whisper import WhisperModel
from pydub import AudioSegment


class VoiceRecognitionThread(QThread):
    status_update = pyqtSignal(str)
    command_received = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        print("DEBUGGING :: ")
        

    def run(self):
        self.status_update.emit("Listening")
        logging.info("Listening for audio input")

        try:
            # Record audio
            duration = 8  # seconds
            fs = 44100  # Sample rate (Vosk models typically expect 16kHz)
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
            sd.wait()
            # recording = recording.flatten()

            # Apply audio processing
            processed_audio = recording
            final_path = "audio.mp3"
            wav_file = "temp_audio.wav"
            wavfile.write(wav_file, fs, processed_audio)
            audio = AudioSegment.from_wav(wav_file)
            audio.export(final_path, format="mp3")
            model_size = "large-v3"
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments ,info = model.transcribe("./audio.mp3", beam_size=1 , language= "en")
            command = ""
            for segment in segments:
                print("%s" % (segment.text))
                command += segment.text
            os.remove(wav_file)
            os.remove(final_path)
            self.command_received.emit(command)
            

        except Exception as e:
            error_message = f"Error in voice recognition: {str(e)}"
            self.status_update.emit(error_message)
            logging.error(error_message)
