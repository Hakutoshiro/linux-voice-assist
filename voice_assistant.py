import os
import subprocess
import logging
import json
import tempfile
import pyttsx3
# from gtts import gTTS
# from playsound import playsound
# from pydub import AudioSegment
# from pydub.generators import Sine
# import simpleaudio as sa
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QProgressBar, QApplication
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor\

from llms.gemini import GeminiService
import numpy as np
from llms.groq import GroqService

from config import GEMINI_API_KEY

# import google.generativeai as genai
from voice_recognition_thread import VoiceRecognitionThread
from config import GEMINI_API_KEY, GROQ_API_KEY

from rag_service import RAGService

from create_embeddings import EmbeddingProvider

import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Gemini API
# genai.configure(api_key=GEMINI_API_KEY)

class TTSThread(QThread):
    def __init__(self, text):
        super().__init__()
        self.text = text
        

    def run(self):
        engine = pyttsx3.init()
        engine.say(self.text)
        engine.runAndWait()
        engine.stop()


class VoiceAssistant(QMainWindow):
    def __init__(self, llm_service='gemini', embeddings_dir='embeddings'):
        super().__init__()
        self.setWindowTitle("Smart Voice Assistant")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C3E50;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 16px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QProgressBar {
                border: 2px solid #3498DB;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_layout = QHBoxLayout()
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.status_label)

        self.listen_button = QPushButton("Start Listening")
        self.listen_button.clicked.connect(self.start_listening)
        header_layout.addWidget(self.listen_button)

        main_layout.addLayout(header_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status constants
        self.STATUS_IDLE = "Idle"
        self.STATUS_LISTENING = "Listening for voice input..."
        self.STATUS_PROCESSING = "Processing command..."
        self.STATUS_GENERATING = "Generating script..."
        self.STATUS_VALIDATING = "Validating script..."
        self.STATUS_EXECUTING = "Executing command..."
        self.STATUS_INTERPRETING = "Interpreting results..."
        self.STATUS_SPEAKING = "Speaking response..."
        self.STATUS_ERROR = "Error occurred"

        # Initial status
        self.current_status = self.STATUS_IDLE
        self.status_label.setText(self.current_status)

        # Terminal
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: #000000; /* Black background */
                color: #FFFFFF; /* White text */
                border: none;
                font-family: 'DejaVu Sans Mono', monospace; /* Standard Linux font */
                font-size: 16px; /* Increased font size */
                padding: 10px;
            }

        """)
        main_layout.addWidget(self.terminal)

        # Set up the terminal prompt
        self.terminal.append("$ ")
        self.terminal.moveCursor(QTextCursor.End)

        # Voice recognition thread
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.status_update.connect(self.update_status)
        self.voice_thread.command_received.connect(self.process_command)

        # Initialize the LLM service
        if llm_service == 'gemini':
            self.llm = GeminiService(GEMINI_API_KEY)
        elif llm_service == 'groq':
            self.llm = GroqService(GROQ_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM service: {llm_service}")

        self.llm.initialize()

        try:
            self.rag = RAGService(embeddings_dir)
            self.rag.load_index()
        except FileNotFoundError:
            logging.error(
                "Embeddings not found. Please run create_embeddings.py first with your context file."
            )
            raise

        self.embedding_provider = EmbeddingProvider(llm_service)


        # Timer for progress bar
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)

        self.allowed_commands = set([
            'ls', 'echo', 'cat', 'grep', 'awk', 'sed', 'cut', 'sort', 'uniq',
            'wc', 'head', 'tail', 'find', 'date', 'pwd', 'whoami', 'uname', 'mkdir', 
            'rmdir', 'touch', 'cp', 'mv', 'less', 'nano', 'vim', 'more', 'diff', 
            'tar', 'gzip', 'gunzip', 'zip', 'unzip', 'ping', 'curl', 'wget', 'apt', 
            'dpkg', 'python', 'python3', 'g++', 'gcc', 'node', 'javac', 'java', 
            'ruby', 'df', 'du', 'free', 'top', 'htop', 'uptime', 'ps', 'id', 
            'hostname', 'cal', 'man', 'bc', 'time', 'xargs', 'tr', 'chmod', 'chown', 
            'tee', 'split', 'dmesg', 'iostat', 'vmstat', 'sar', 'lsof', 'who', 
            'last', 'mount', 'umount', 'blkid', 'fdisk', 'mkfs', 'ifconfig', 'netstat', 
            'ss', 'iptables', 'traceroute', 'nmap', 'which', 'locate', 'bc',
            'alias', 'unalias', 'factor', 'yes', 'shutdown', 'reboot', 'kill', 'killall',
            'history', 'ip', 'clear', 'exit', 'logout', 'su', 'sudo', 'passwd', 'useradd',
        ])
        # self.tts_engine = pyttsx3.init()
        # for text-to-speech
        self.speech_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name


        logging.info("Voice Assistant initialized")

    def start_listening(self):
        self.progress_bar.setValue(0)
        self.timer.start(40)
        self.voice_thread.start()
        self.current_status = self.STATUS_LISTENING
        self.status_label.setText(self.current_status)
        self.terminal_print("Listening for command...")
        self.listen_button.setEnabled(False)
        self.listen_button.setText("Listening...")


    def update_progress(self):
        value = self.progress_bar.value() + 1
        if value > 100:
            self.timer.stop()
            self.progress_bar.setValue(0)
            self.listen_button.setEnabled(True)
            self.listen_button.setText("Start Listening")
        else:
            self.progress_bar.setValue(value)


    def update_status(self, status):
        self.status_label.setText(status)
        self.terminal_print(f"Status: {status}")


    def terminal_print(self, text):
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(text + "\n")
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.ensureCursorVisible()
        self.terminal.append("$ ")
        self.terminal.moveCursor(QTextCursor.End)

    def validate_script(self, script_content):
        # Step 1: Check against whitelist
        for line in script_content.split('\n'):
            command = line.strip().split()[0] if line.strip() else ''
            if command and command not in self.allowed_commands:
                self.terminal_print(f"Validation failed: '{command}' is not in the whitelist of allowed commands.")
                return False

        # Step 2: Use shellcheck for static analysis
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(['shellcheck', '-s', 'bash', temp_file_path], capture_output=True, text=True)
            if result.returncode != 0:
                self.terminal_print("ShellCheck found issues in the script:")
                self.terminal_print(result.stdout)
                return False
            else:
                self.terminal_print("ShellCheck validation passed.")
                return True
        finally:
            os.unlink(temp_file_path)

    def execute_in_sandbox(self, script_content):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name

        try:
            # Use firejail to create a sandboxed environment
            result = subprocess.run([
                'sudo',
                'firejail',
                '--noprofile',
                '--quiet',
                '--private',
                '--net=none',
                '--nosound',
                '--no3d',
                'bash',
                temp_file_path
            ], capture_output=True, text=True)

            return result.stdout if result.returncode == 0 else result.stderr
        finally:
            os.unlink(temp_file_path)

    # def interpret_output(self, original_query, command_output):
    #     model = genai.GenerativeModel('gemini-1.5-flash')
    #     prompt = OUTPUT_INTERPRETATION_PROMPT.format(
    #         query=original_query,
    #         output=command_output
    #     )
    #     response = model.generate_content(prompt)
    #     return response.text

    def speak_text(self, text):
        self.current_status = self.STATUS_SPEAKING
        self.status_label.setText(self.current_status)
        self.tts_thread = TTSThread(text)
        self.tts_thread.start()
        self.tts_thread.wait()  # Wait for the thread to finish
        self.current_status = self.STATUS_IDLE
        self.status_label.setText(self.current_status)



    def closeEvent(self, event):
        # Ensure all threads are stopped before closing
        if hasattr(self, 'tts_thread') and self.tts_thread.isRunning():
            self.tts_thread.wait()
        event.accept()


    def process_command(self, command):
        logging.info(f"Processing command: {command}")
        self.current_status = self.STATUS_PROCESSING
        self.status_label.setText(self.current_status)
        self.terminal_print(f"Received command: {command}")

        
        try:
            # Get context from RAG model
            self.current_status = self.STATUS_GENERATING
            self.status_label.setText(self.current_status)

            context = self.rag.get_relevant_context(command)
            # self.terminal_print(f"Context: {context}")
            # Generate bash script using the LLM service
            bash_script, description = self.llm.generate_bash_script(command, context)

            logging.debug(f"Generated bash script: {bash_script}")
            logging.debug(f"Description: {description}")
            
            self.terminal_print(f"Description: {description}")
            self.terminal_print("Validating script...")
            
            # Validate the script
            if self.validate_script(bash_script):
                # Execute in sandbox
                self.current_status = self.STATUS_EXECUTING
                self.status_label.setText(self.current_status)
                self.terminal_print("Executing command in sandbox...")
                command_output = self.execute_in_sandbox(bash_script)
                self.terminal_print(f"Raw output: {command_output}")
                
                # Interpret output
                self.current_status = self.STATUS_INTERPRETING
                self.status_label.setText(self.current_status)
                interpreted_response = self.llm.interpret_output(command, command_output)
                self.terminal_print(f"Interpreted response: {interpreted_response}")
                
                # Speak response
                self.current_status = self.STATUS_SPEAKING
                self.status_label.setText(self.current_status)
                self.speak_text(interpreted_response)
                
            else:
                self.current_status = self.STATUS_ERROR
                self.status_label.setText(self.current_status)
                error_message = "Script execution aborted due to security concerns."
                self.terminal_print(error_message)
                self.speak_text(error_message)

        except Exception as e:
            self.current_status = self.STATUS_ERROR
            self.status_label.setText(self.current_status)
            error_message = f"Error executing command: {str(e)}"
            self.terminal_print(error_message)
            self.speak_text(error_message)
            logging.error(error_message)

        
        self.current_status = self.STATUS_IDLE
        self.status_label.setText(self.current_status)



