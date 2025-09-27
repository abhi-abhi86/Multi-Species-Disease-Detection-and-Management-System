# DiseaseDetectionApp/ui/chatbot_dialog.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import re

class ChatbotWorker(QObject):
    """
    A simple worker to perform database lookups in the background.
    This is not a real AI, but a fast information retriever.
    """
    response_ready = pyqtSignal(str)

    def __init__(self, message, database):
        super().__init__()
        self.message = message.lower().strip()
        self.database = database

    def run(self):
        """
        Searches the local database for keywords from the user's message.
        """
        # Basic intent recognition
        if "what is" in self.message:
            # Extract the term after "what is"
            search_term = self.message.split("what is", 1)[-1].strip().rstrip('?')
        else:
            # Use the whole message as the search term
            search_term = self.message.rstrip('?')
        
        # Clean the search term
        search_term = re.sub(r'\s+', ' ', search_term).strip()

        if not search_term:
            self.response_ready.emit("Please ask a more specific question, like 'What is Ringworm?'")
            return

        # Search for a matching disease in the database
        best_match = None
        for disease in self.database:
            if search_term.lower() in disease['name'].lower():
                best_match = disease
                break # Found a direct match

        if best_match:
            response = best_match.get('description', 'No description available for this disease.')
        else:
            response = f"I'm sorry, I couldn't find any information about '{search_term}' in my database."

        self.response_ready.emit(response)


class ChatbotDialog(QDialog):
    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.setWindowTitle("Disease Info Chatbot")
        self.setMinimumSize(400, 500)

        self.layout = QVBoxLayout(self)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText("Ask a question about a disease, e.g., 'What is Ringworm?'")
        self.chat_history.setHtml("<p style='color:#888;'><i>Hello! I'm a bot that can look up disease information from the local database. How can I help?</i></p>")

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_user_message)
        self.user_input.returnPressed.connect(self.send_button.click)

        self.layout.addWidget(self.chat_history)
        self.layout.addWidget(self.user_input)
        self.layout.addWidget(self.send_button)

        self.thread = None
        self.worker = None

    def handle_user_message(self):
        user_text = self.user_input.text().strip()
        if not user_text:
            return

        self.chat_history.append(f"<b>You:</b> {user_text}")
        self.user_input.clear()
        self.send_button.setEnabled(False)
        self.user_input.setPlaceholderText("Searching database...")

        # Run lookup logic in a background thread to keep UI responsive
        self.thread = QThread()
        self.worker = ChatbotWorker(user_text, self.database)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.response_ready.connect(self.display_bot_response)
        
        self.worker.response_ready.connect(self.cleanup_thread)

        self.thread.start()

    def display_bot_response(self, bot_text):
        self.chat_history.append(f"<b style='color:#005A9C;'>Bot:</b> {bot_text}<br>")

    def cleanup_thread(self):
        self.send_button.setEnabled(True)
        self.user_input.setPlaceholderText("Type your message here...")
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None
