                                          
                                          
                              
                                                            
                                                                      
                                                                  

WATERMARK_AUTHOR = "abhi-abhi86"
WATERMARK_CHECK = True

def check_watermark():
    """Check if watermark is intact. Application will fail if removed."""
    if not WATERMARK_CHECK or WATERMARK_AUTHOR != "abhi-abhi86":
        print("ERROR: Watermark protection violated. Application cannot start.")
        print("Made by: abhi-abhi86")
        import sys
        sys.exit(1)

                         
check_watermark()

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import re
import wikipedia

                                                                         
try:
    from fuzzywuzzy import process
except ImportError:
    process = None

                                                     
try:
    from ..core.llm_integrator import LLMIntegrator
    llm_available = True
except ImportError:
    llm_available = False
    LLMIntegrator = None

class WikipediaWorker(QObject):
    """
    A worker that fetches Wikipedia information for a disease in a background thread.
    """
    response_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, disease_name):
        super().__init__()
        self.disease_name = disease_name

    def run(self):
        """Fetch Wikipedia summary for the disease."""
        try:
            summary = wikipedia.summary(self.disease_name, sentences=3, auto_suggest=True, redirect=True)
            self.response_ready.emit(f"From Wikipedia: {summary}")
        except wikipedia.exceptions.PageError:
            self.response_ready.emit(None)
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                first_option = e.options[0]
                summary = wikipedia.summary(first_option, sentences=3, auto_suggest=False)
                self.response_ready.emit(f"From Wikipedia (assuming '{first_option}'): {summary}")
            except Exception:
                self.response_ready.emit(None)
        except Exception:
            self.response_ready.emit(None)
        finally:
            self.finished.emit()


class ChatbotWorker(QObject):
    """
    A worker that processes user queries in a background thread to find
    relevant disease information from the database. It uses fuzzy string matching
    for better results, understands basic intents, and integrates LLM for enhanced responses.
    """
    response_ready = pyqtSignal(str)

    def __init__(self, message, database, llm_integrator=None):
        super().__init__()
        self.message = message.lower().strip()
        self.database = database
        self.llm_integrator = llm_integrator

    def run(self):
        """ Processes the user's message to provide a helpful response. """

        if not self.message:
            return
            
        greetings = ['hello', 'hi', 'hey']
        if self.message in greetings:
            self.response_ready.emit("Hello! Ask me for information about a disease. For example, 'What are the symptoms of Ringworm?'")
            return

        if process is None:
            self.response_ready.emit("Chatbot functionality is limited. For better matching, please install 'fuzzywuzzy' and 'python-Levenshtein' (`pip install fuzzywuzzy python-Levenshtein`).")
                                                                               
            self.simple_search()
            return

                                      
        disease_names = [d['name'] for d in self.database]

                                                                
                                                                   
        best_match, score = process.extractOne(self.message, disease_names)

                                                                                          
        if score < 55:
                                                        
            if self.llm_integrator:
                llm_response = self.llm_integrator.generate_response(self.message)
                self.response_ready.emit(llm_response)
            else:
                                       
                wiki_response = self.get_wikipedia_info(self.message)
                if wiki_response:
                    self.response_ready.emit(wiki_response)
                else:
                    self.response_ready.emit(f"I'm sorry, I couldn't find any information related to '{self.message}' in my database or on Wikipedia. Please check the spelling or try a different disease.")
            return

                                                         
        matched_disease = next((d for d in self.database if d['name'] == best_match), None)
        if not matched_disease:
            self.response_ready.emit(f"An unexpected error occurred while looking up '{best_match}'.")
            return

                                                      
                                                                      
        response = self.get_info_by_intent(matched_disease)
                                                                               
        if self.llm_integrator and len(response) < 200:                                
                                                               
            enhanced_response = self.llm_integrator.generate_response(
                f"Provide a brief, detailed explanation for: {response}",
                {"disease": matched_disease['name'], "query": self.message}
            )
            if enhanced_response and not enhanced_response.startswith("Sorry"):
                response = enhanced_response
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
                                                
            try:
                first_option = e.options[0]
                summary = wikipedia.summary(first_option, sentences=3, auto_suggest=False)
                return f"From Wikipedia (assuming '{first_option}'): {summary}"
            except Exception:
                return None
        except Exception:
            return None

class ChatbotDialog(QDialog):
    def __init__(self, database, parent=None, initial_query="", llm_integrator=None, disease_from_image=None):
        super().__init__(parent)
        self.database = database
        self.llm_integrator = llm_integrator
        self.disease_from_image = disease_from_image
        self.setWindowTitle("AI-Powered Disease Information Chatbot")
        self.setMinimumSize(500, 600)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 
                border-radius: 10px;
            }
            QTextEdit {
                background: white;
                border: 1px solid 
                border-radius: 5px;
                padding: 5px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 12px;
            }
            QLineEdit {
                background: white;
                border: 2px solid 
                border-radius: 20px;
                padding: 8px 15px;
                font-size: 14px;
                font-family: 'Segoe UI', sans-serif;
            }
            QLineEdit:focus {
                border-color: 
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 
            }
            QPushButton:pressed {
                background: 
            }
        """)

                          
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)

                     
        title_label = QLabel("🤖 AI Disease Assistant")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0078d4; margin-bottom: 10px;")
        self.layout.addWidget(title_label)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("border-radius: 8px;")

        greeting = "Hello! I can look up disease information from the local database"
        if self.llm_integrator:
            greeting += " and provide AI-enhanced responses."
        else:
            greeting += "."
        greeting += " How can I help?"
        self.chat_history.setHtml(f"<p style='color: #0078d4; font-weight: bold;'>{greeting}</p>")

                      
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Ask about diseases or type symptoms...")
        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(80)

        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)

        self.layout.addWidget(self.chat_history)
        self.layout.addLayout(input_layout)

                           
        self.worker_thread = None

                             
        self.send_button.clicked.connect(self.handle_user_message)
        self.user_input.returnPressed.connect(self.handle_user_message)

                                       
        if initial_query:
            self.user_input.setText(initial_query)
            self.handle_user_message()
        elif self.disease_from_image:
                                                                              
            self.fetch_wikipedia_for_disease(self.disease_from_image)

    def fetch_wikipedia_for_disease(self, disease_name):
        """Fetch Wikipedia information for a disease detected from image."""
        self.chat_history.append(f"<p style='color: #0078d4;'>Fetching Wikipedia information for '{disease_name}'...</p>")
        self.chat_history.append("<p style='color: #666;'>Please wait...</p>")

                                                  
        self.worker_thread = QThread()
        worker = WikipediaWorker(disease_name)
        worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(worker.run)
        worker.response_ready.connect(self.display_wikipedia_response)
        worker.finished.connect(self.worker_thread.quit)
        worker.finished.connect(worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def display_wikipedia_response(self, response):
        """Display the Wikipedia response in the chat history."""
        if response:
            self.chat_history.append(f"<div style='background: #e8f5e8; border-left: 4px solid #28a745; padding: 10px; margin: 5px 0; border-radius: 5px;'>{response}</div>")
        else:
            self.chat_history.append("<p style='color: #dc3545;'>No Wikipedia information found for this disease.</p>")

    def handle_user_message(self):
        user_text = self.user_input.text().strip()
        if not user_text:
            return

        self.chat_history.append(f"<b>You:</b> {user_text}")
        self.user_input.clear()
        self.set_ui_busy(True)

                                                  
        self.worker_thread = QThread()
        worker = ChatbotWorker(user_text, self.database, self.llm_integrator)
        worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(worker.run)
        worker.response_ready.connect(self.display_bot_response)
        worker.response_ready.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: self.set_ui_busy(False))
        
        self.worker_thread.start()

    def display_bot_response(self, bot_text):
        self.chat_history.append(f"<b style='color: #0078d4;'>Bot:</b> {bot_text}")

    def set_ui_busy(self, is_busy):
        """ Enables or disables the input fields while the bot is 'thinking'. """
        self.user_input.setEnabled(not is_busy)
        self.send_button.setEnabled(not is_busy)
        self.user_input.setPlaceholderText("Thinking..." if is_busy else "Type your question here...")

    def closeEvent(self, event):
                                                                            
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        super().closeEvent(event)
