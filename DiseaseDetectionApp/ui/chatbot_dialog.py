# DiseaseDetectionApp/ui/chatbot_dialog.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import re
try:
    from fuzzywuzzy import process
except ImportError:
    process = None # Handle case where library is not installed

class ChatbotWorker(QObject):
    """
    An improved worker that performs more intelligent database lookups.
    It uses FuzzyWuzzy for robust matching, handles spelling mistakes, 
    understands basic intents, and responds to simple greetings.
    """
    response_ready = pyqtSignal(str)

    def __init__(self, message, database):
        super().__init__()
        self.message = message.lower().strip()
        self.database = database

    def run(self):
        """
        Processes the user's message to provide a helpful response.
        """
        try:
            if process is None:
                self.response_ready.emit("Chatbot functionality is limited. Please install the 'fuzzywuzzy' and 'python-Levenshtein' libraries by running: pip install fuzzywuzzy python-Levenshtein")
                return
            
            # 1. Handle simple greetings and conversational filler
            greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
            if any(greet in self.message for greet in greetings):
                self.response_ready.emit("Hello! How can I help you today? You can ask me about diseases in the database, for example: 'What are the symptoms of Ringworm?'")
                return

            if "how are you" in self.message:
                self.response_ready.emit("I'm a bot, so I'm always running efficiently! What can I help you with?")
                return
            
            if "help" in self.message or "what can you do" in self.message:
                self.response_ready.emit("I can provide information about diseases stored in the local database. You can ask me for a disease's description, symptoms, causes, or solution. For example: 'Tell me about Powdery Mildew'.")
                return

            # 2. Extract the main topic from the user's query
            disease_names = [d['name'] for d in self.database]
            search_term = self.message
            
            # Remove common question phrases to isolate the search term
            question_phrases = ["what is", "what are", "tell me about", "describe", "what's", "whats", "show me"]
            for phrase in question_phrases:
                if self.message.startswith(phrase):
                    search_term = self.message[len(phrase):].strip()
                    break
            
            # Clean up the search term
            search_term = re.sub(r'[^\w\s]', '', search_term)

            if not search_term:
                self.response_ready.emit("I'm not sure what you mean. Please try asking a question like 'What is Ringworm?'")
                return

            # 3. Find the best matching disease using FuzzyWuzzy
            # process.extractOne returns (best_match, score)
            best_match, score = process.extractOne(search_term, disease_names)
            
            # We set a confidence threshold of 50. If the score is below this,
            # it's likely not a good match.
            if score < 50:
                self.response_ready.emit(f"I'm sorry, I couldn't find any information about '{search_term}' in my database. Please check the spelling or try another disease name.")
                return

            # Retrieve the full disease data object
            matched_disease = next((d for d in self.database if d['name'] == best_match), None)

            # Ensure a match was actually found before proceeding
            if not matched_disease:
                self.response_ready.emit(f"An unexpected error occurred while looking up '{best_match}'.")
                return

            # 4. Determine the specific information the user wants (intent detection)
            response = ""
            if any(word in self.message for word in ['symptom', 'sign', 'stage']):
                stages = matched_disease.get("stages", {})
                if stages:
                    stages_text = "\n".join([f"• <b>{k}:</b> {v}" for k, v in stages.items()])
                    response = f"Here are the stages/symptoms for <b>{matched_disease['name']}</b>:<br>{stages_text}"
                else:
                    response = f"I don't have specific symptom or stage information for {matched_disease['name']}."
            elif any(word in self.message for word in ['cause', 'reason']):
                response = matched_disease.get('causes', f"I don't have information on the causes of {matched_disease['name']}.")
            elif any(word in self.message for word in ['prevent', 'avoid']):
                response = matched_disease.get('preventive_measures', f"I don't have preventive measure information for {matched_disease['name']}.")
            elif any(word in self.message for word in ['solution', 'cure', 'treat', 'fix']):
                response = matched_disease.get('solution', f"I don't have solution/cure information for {matched_disease['name']}.")
            elif any(word in self.message for word in ['risk', 'factor']):
                response = matched_disease.get('risk_factors', f"I don't have risk factor information for {matched_disease['name']}.")
            else:
                # Default to the general description if no specific intent is found
                response = matched_disease.get('description', 'No description available for this disease.')

            self.response_ready.emit(response)
        
        except Exception as e:
            print(f"Chatbot Error: {e}")
            self.response_ready.emit("I'm sorry, something went wrong. Please try your question again.")


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
        self.user_input.setPlaceholderText("Thinking...")

        # Run lookup logic in a background thread to keep UI responsive
        self.thread = QThread()
        self.worker = ChatbotWorker(user_text, self.database)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.response_ready.connect(self.display_bot_response)
        
        self.worker.response_ready.connect(self.cleanup_thread)

        self.thread.start()

    def display_bot_response(self, bot_text):
        # Using HTML for richer text formatting in the response
        self.chat_history.append(f"<b style='color:#005A9C;'>Bot:</b> {bot_text.replace(r'n', '<br>')}<br>")

    def cleanup_thread(self):
        self.send_button.setEnabled(True)
        self.user_input.setPlaceholderText("Type your message here...")
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None

