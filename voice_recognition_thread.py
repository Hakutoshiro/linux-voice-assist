import os
import sys
import logging
import json
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import sounddevice as sd
from scipy.io import wavfile
from vosk import Model, KaldiRecognizer
from audio_processing import process_audio
from config import VOSK_MODEL_PATH



class VoiceRecognitionThread(QThread):
    status_update = pyqtSignal(str)
    command_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        print("DEBUGGING :: ")
        print(VOSK_MODEL_PATH)
        # Initialize Vosk model
        if not os.path.exists(VOSK_MODEL_PATH):
            logging.error(f"Please download a model from https://alphacephei.com/vosk/models and unpack as {VOSK_MODEL_PATH}")
            sys.exit(1)
        self.model = Model(VOSK_MODEL_PATH)
        self.rec = KaldiRecognizer(self.model, 16000)

    def run(self):
        self.status_update.emit("Listening")
        logging.info("Listening for audio input")

        try:
            # clear the buffer
            sd.default.device = None
            self.rec = KaldiRecognizer(self.model, 16000)
            # Record audio
            duration = 4  # seconds
            fs = 16000  # Sample rate (Vosk models typically expect 16kHz)
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            recording = recording.flatten()

            # Apply audio processing
            # processed_audio = process_audio(recording, fs)
            processed_audio = recording
            # Convert float audio to int16
            audio_int16 = (processed_audio * 32767).astype(np.int16)
            
            # Perform recognition with Vosk
            if self.rec.AcceptWaveform(audio_int16.tobytes()):
                result = json.loads(self.rec.Result())
                command = result['text']
                if command:
                    logging.info(f"Recognized command using Vosk: {command}")
                    self.command_received.emit(command)
                else:
                    self.status_update.emit("No speech detected")
                    logging.warning("No speech detected")
            else:
                # self.status_update.emit("Could not understand audio")
                # logging.error("Vosk could not understand audio")
                partial = json.loads(self.rec.PartialResult())
                #print(f"Vosk partial result: {partial}")
                command = partial["partial"]
                if command:
                    logging.info(f"Recognized command using Vosk: {command}")
                    self.command_received.emit(command)
                else:
                    self.status_update.emit("No speech detected")
                    logging.warning("No speech detected")

        except Exception as e:
            error_message = f"Error in voice recognition: {str(e)}"
            self.status_update.emit(error_message)
            logging.error(error_message)
