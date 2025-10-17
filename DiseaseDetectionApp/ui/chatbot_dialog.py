# DiseaseDetectionApp/ui/chatbot_dialog.py
# DiseaseDetectionApp/ui/chatbot_dialog.py
# --- WATERMARK PROTECTION ---
# This code is protected by watermark. Made by "abhi-abhi86"
# Unauthorized copying, modification, or redistribution is prohibited.
# If this watermark is removed, the application will not function.

WATERMARK_AUTHOR = "abhi-abhi86"
WATERMARK_CHECK = True

def check_watermark():
    """Check if watermark is intact. Application will fail if removed."""
    if not WATERMARK_CHECK or WATERMARK_AUTHOR != "abhi-abhi86":
        print("ERROR: Watermark protection violated. Application cannot start.")
        print("Made by: abhi-abhi86")
        import sys
        sys.exit(1)

# Execute watermark check
check_watermark()

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import re
import wikipedia

# Try to import fuzzywuzzy, but allow the chatbot to function without it.
try:
    from fuzzywuzzy import process
except ImportError:
    process = None

class ChatbotWorker(QObject):
    """
    A worker that processes user queries in a background thread to find
    relevant disease information from the database. It uses fuzzy string matching
    for better results and understands basic intents.
    """
    response_ready = pyqtSignal(str)

    def __init__(self, message, database):
        super().__init__()
        self.message = message.lower().strip()
        self.database = database

    def run(self):
        """ Processes the user's message to provide a helpful response. """
        # --- Pre-computation and Greeting Handling ---
        if not self.message:
            return
            
        greetings = ['hello', 'hi', 'hey']
        if self.message in greetings:
            self.response_ready.emit("Hello! Ask me for information about a disease. For example, 'What are the symptoms of Ringworm?'")
            return

        if process is None:
            self.response_ready.emit("Chatbot functionality is limited. For better matching, please install 'fuzzywuzzy' and 'python-Levenshtein' (`pip install fuzzywuzzy python-Levenshtein`).")
            # Fallback to simple keyword search if fuzzywuzzy is not available.
            self.simple_search()
            return

        # --- Fuzzy Matching Logic ---
        disease_names = [d['name'] for d in self.database]
        
        # Extract the best matching disease name from the query.
        # `process.extractOne` returns a tuple: (best_match, score)
        best_match, score = process.extractOne(self.message, disease_names)
        
        # Set a confidence threshold. If the match score is too low, try Wikipedia.
        if score < 55:
            # Try to get information from Wikipedia
            wiki_response = self.get_wikipedia_info(self.message)
            if wiki_response:
                self.response_ready.emit(wiki_response)
            else:
                self.response_ready.emit(f"I'm sorry, I couldn't find any information related to '{self.message}' in my database or on Wikipedia. Please check the spelling or try a different disease.")
            return

        # Retrieve the full data for the matched disease.
        matched_disease = next((d for d in self.database if d['name'] == best_match), None)
        if not matched_disease:
            self.response_ready.emit(f"An unexpected error occurred while looking up '{best_match}'.")
            return

        # --- Intent Detection ---
        # Determine what specific piece of information the user wants.
        response = self.get_info_by_intent(matched_disease)
        self.response_ready.emit(response)

    def simple_search(self):
        """ A fallback search method if fuzzywuzzy is not installed. """
        best_match_disease = None
        max_matches = 0
        search_words = set(self.message.split())

        for disease in self.database:
            disease_words = set(disease['name'].lower().split())
            matches = len(search_words.intersection(disease_words))
            if matches > max_matches:
                max_matches = matches
                best_match_disease = disease

        if best_match_disease:
            response = self.get_info_by_intent(best_match_disease)
            self.response_ready.emit(response)
        else:
            # Try Wikipedia if no match in database
            wiki_response = self.get_wikipedia_info(self.message)
            if wiki_response:
                self.response_ready.emit(wiki_response)
            else:
                self.response_ready.emit("I couldn't find a clear match in the database or on Wikipedia. Please be more specific.")

    def get_info_by_intent(self, disease_data):
        """ Returns a specific piece of information based on keywords in the user's message. """
        if any(word in self.message for word in ['symptom', 'stage', 'sign']):
            stages = disease_data.get("stages", {})
            if stages:
                stages_text = "<br>".join([f"• <b>{k}:</b> {v}" for k, v in stages.items()])
                return f"Here are the stages/symptoms for <b>{disease_data['name']}</b>:<br>{stages_text}"
            else:
                return f"I don't have specific symptom information for {disease_data['name']}."
        elif any(word in self.message for word in ['cause']):
            return disease_data.get('causes', f"Cause information is not available for {disease_data['name']}.")
        elif any(word in self.message for word in ['prevent', 'avoid']):
            return disease_data.get('preventive_measures', f"Prevention information is not available for {disease_data['name']}.")
        elif any(word in self.message for word in ['solution', 'cure', 'treat']):
            return disease_data.get('solution', f"Solution information is not available for {disease_data['name']}.")
        else:
            # Default to the general description if no specific intent is found.
            return disease_data.get('description', 'No description available for this disease.')

    def get_wikipedia_info(self, query):
        """
        Fetches a summary from Wikipedia for the given query.
        Returns None if no information is found or an error occurs.
        """
        try:
            summary = wikipedia.summary(query, sentences=3, auto_suggest=True, redirect=True)
            return f"From Wikipedia: {summary}"
        except wikipedia.exceptions.PageError:
            return None
        except wikipedia.exceptions.DisambiguationError as e:
            # If ambiguous, try the first option
            try:
                first_option = e.options[0]
                summary = wikipedia.summary(first_option, sentences=3, auto_suggest=False)
                return f"From Wikipedia (assuming '{first_option}'): {summary}"
            except Exception:
                return None
        except Exception:
            return None

class ChatbotDialog(QDialog):
    def __init__(self, database, parent=None, initial_query=""):
        super().__init__(parent)
        self.database = database
        self.setWindowTitle("Disease Information Chatbot")
        self.setMinimumSize(450, 550)

        # --- UI Setup ---
        self.layout = QVBoxLayout(self)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setHtml("<p style='color:#0056b3;'><i>Hello! I can look up disease information from the local database. How can I help?</i></p>")

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your question here and press Enter...")

        self.send_button = QPushButton("Send")

        self.layout.addWidget(self.chat_history)
        self.layout.addWidget(self.user_input)
        self.layout.addWidget(self.send_button)

        # --- Threading ---
        self.worker_thread = None

        # --- Connections ---
        self.send_button.clicked.connect(self.handle_user_message)
        self.user_input.returnPressed.connect(self.handle_user_message)

        # Set initial query if provided
        if initial_query:
            self.user_input.setText(initial_query)
            self.handle_user_message()

    def handle_user_message(self):
        user_text = self.user_input.text().strip()
        if not user_text:
            return

        self.chat_history.append(f"<b>You:</b> {user_text}")
        self.user_input.clear()
        self.set_ui_busy(True)

        # Run lookup logic in a background thread.
        self.worker_thread = QThread()
        worker = ChatbotWorker(user_text, self.database)
        worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(worker.run)
        worker.response_ready.connect(self.display_bot_response)
        worker.response_ready.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: self.set_ui_busy(False))
        
        self.worker_thread.start()

    def display_bot_response(self, bot_text):
        self.chat_history.append(f"<b style='color:#0056b3;'>Bot:</b> {bot_text}<br>")

    def set_ui_busy(self, is_busy):
        """ Enables or disables the input fields while the bot is 'thinking'. """
        self.user_input.setEnabled(not is_busy)
        self.send_button.setEnabled(not is_busy)
        self.user_input.setPlaceholderText("Thinking..." if is_busy else "Type your question here...")

    def closeEvent(self, event):
        # Ensure thread is stopped if dialog is closed while bot is running.
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        super().closeEvent(event)
