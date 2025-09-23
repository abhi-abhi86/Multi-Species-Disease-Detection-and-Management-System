#chatbot_dialog.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt6.QtCore import Qt

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
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_user_message)
        self.user_input.returnPressed.connect(self.send_button.click) # Allow pressing Enter

        self.layout.addWidget(self.chat_history)
        self.layout.addWidget(self.user_input)
        self.layout.addWidget(self.send_button)

    def handle_user_message(self):
        user_text = self.user_input.text().strip()
        if not user_text:
            return

        self.chat_history.append(f"You: {user_text}")
        self.user_input.clear()

        # Simple keyword-based response logic
        bot_response = self.generate_bot_response(user_text)
        self.chat_history.append(f"Bot: {bot_response}\n")

    def generate_bot_response(self, user_text):
        user_text = user_text.lower()
        
        # Check if the user is asking about a specific disease
        for disease in self.database:
            if disease['name'].lower() in user_text:
                if 'solution' in user_text or 'cure' in user_text:
                    return f"The recommended solution for {disease['name']} is: {disease.get('solution', 'N/A')}"
                if 'symptoms' in user_text or 'stages' in user_text:
                    stages = "\n".join([f"• {k}: {v}" for k, v in disease.get('stages', {}).items()])
                    return f"The stages/symptoms for {disease['name']} are:\n{stages}"
                return disease.get('description', 'No description available.')

        if "hello" in user_text or "hi" in user_text:
            return "Hello! How can I help you today? You can ask me about diseases in the database."
        if "bye" in user_text:
            return "Goodbye! Stay healthy."
            
        return "I'm sorry, I don't understand that. Please ask about a specific disease."
