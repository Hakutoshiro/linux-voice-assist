import sys
from PyQt5.QtWidgets import QApplication
from voice_assistant import VoiceAssistant
from config import EMBEDDINGS_DIR_PATH
if __name__ == "__main__":
    app = QApplication(sys.argv)
    assistant = VoiceAssistant(llm_service="groq", embeddings_dir=EMBEDDINGS_DIR_PATH)
    assistant.show()
    sys.exit(app.exec_())
