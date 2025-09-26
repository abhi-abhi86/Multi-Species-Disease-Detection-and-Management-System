#chatbot_dialog.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import requests
import json

# A simple worker to handle API calls in the background
class ChatbotWorker(QObject):
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, message, database):
        super().__init__()
        self.message = message
        self.database = database

    def run(self):
        try:
            # This is a placeholder for a real API endpoint.
            # Using a public echo API to demonstrate the concept.
            # In a real app, you would replace this with your NLP/LLM API endpoint.
            api_url = "https://api.runapi.io/v1/echo"
            
            # Augment the user's query with context from our database
            context = " ".join([f"{d['name']}: {d['description']}" for d in self.database])

            payload = {
                "prompt": self.message,
                "context": context,
                "model": "advanced_chat_model" # Fictional parameter
            }
            
            # Simulate a more complex response for demonstration
            if "what is" in self.message.lower():
                disease_name = self.message.lower().replace("what is", "").strip().title()
                for disease in self.database:
                    if disease['name'].lower() == disease_name.lower():
                        response_text = disease.get('description', 'No description available.')
                        self.response_ready.emit(response_text)
                        return

            # Fallback for a generic echo response if not a direct question
            response = requests.post(api_url, json={"text": f"Query received: '{self.message}'"})
            response.raise_for_status()
            response_data = response.json()
            
            bot_response = response_data.get("text", "Sorry, I couldn't process that.")
            self.response_ready.emit(bot_response)

        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"Network error: {e}")
        except Exception as e:
            self.error_occurred.emit(f"An unexpected error occurred: {e}")


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
        self.chat_history.setHtml("<p style='color:#888;'><i>Hello! I am an intelligent assistant. How can I help you today?</i></p>")

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_user_message)
        self.user_input.returnPressed.connect(self.send_button.click) # Allow pressing Enter

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
        self.user_input.setPlaceholderText("Thinking...")

        # Run chatbot logic in a background thread
        self.thread = QThread()
        self.worker = ChatbotWorker(user_text, self.database)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.response_ready.connect(self.display_bot_response)
        self.worker.error_occurred.connect(self.display_error)
        
        self.thread.start()

    def display_bot_response(self, bot_text):
        self.chat_history.append(f"<b style='color:#005A9C;'>Bot:</b> {bot_text}<br>")
        self.cleanup_thread()

    def display_error(self, error_text):
        self.chat_history.append(f"<b style='color:red;'>Error:</b> {error_text}<br>")
        self.cleanup_thread()

    def cleanup_thread(self):
        self.send_button.setEnabled(True)
        self.user_input.setPlaceholderText("Type your message here...")
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None
