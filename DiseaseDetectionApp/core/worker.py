                                    
                                    
                              
                                                            
                                                                      
                                                                  

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

import os
import sys

                  
                                                            
                                                                
                                    
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtCore import QObject, pyqtSignal
from core.ml_processor import predict_from_symptoms
from core.ncbi_integration import get_pubmed_summary, generate_ncbi_report

class DiagnosisWorker(QObject):
    """
    Worker object to run the diagnosis in a separate thread, preventing the UI from freezing.
    It now includes caching for PubMed results to reduce redundant network calls.
    """
                                                                                      
    pubmed_cache = {}
                                   
    wiki_cache = {}

                                                
                                                                                                                 
    finished = pyqtSignal(dict, float, str, str, str, str)

                                        
                                          
    error = pyqtSignal(str, str)

                                                
    progress = pyqtSignal(str)

    def __init__(self, ml_processor, image_path, symptoms, domain, database):
        super().__init__()
        self.ml_processor = ml_processor
        self.image_path = image_path
        self.symptoms = symptoms
        self.domain = domain
        self.database = database
        self.is_running = True                                                  

    def run(self):
        """
        Executes the diagnosis process. It chooses between image or symptom analysis,
        fetches external data, and emits the results.
        """
        try:
            result, confidence, wiki, stage = None, 0.0, "", ""
            
                                                                         
                                                                             
            if self.image_path:
                self.progress.emit("Analyzing image with AI model... Please wait.")
                result, confidence, wiki, stage = self.ml_processor.predict_from_image(
                    self.image_path, self.domain, self.database
                )
            elif self.symptoms:
                self.progress.emit("Analyzing symptoms...")
                result, confidence, wiki, stage = predict_from_symptoms(
                    self.symptoms, self.domain, self.database
                )
            else:
                                                                                              
                self.error.emit("No input (image or symptoms) was provided to the worker.", self.domain)
                return

                                                                                 
            if self.is_running and result:
                disease_name = result.get('name', 'Unknown Disease')
                pubmed_summary = ""
                wiki_summary = wiki

                                                                                  
                if disease_name in self.wiki_cache:
                    self.progress.emit("Fetching Wikipedia data from cache...")
                    wiki_summary = self.wiki_cache[disease_name]
                else:
                                                                                                      
                    if wiki and wiki != "N/A":
                        self.wiki_cache[disease_name] = wiki

                                                                               
                if disease_name in self.pubmed_cache:
                    self.progress.emit("Fetching research data from cache...")
                    pubmed_summary = self.pubmed_cache[disease_name]
                else:
                    self.progress.emit("Fetching recent research from PubMed...")
                    try:
                                                      
                        pubmed_summary = get_pubmed_summary(disease_name, domain=self.domain)
                                                                      
                        self.pubmed_cache[disease_name] = pubmed_summary
                    except Exception as e:
                                                             
                        print(f"Network error while fetching from PubMed: {e}")
                        pubmed_summary = "Could not retrieve online research data. Please check your internet connection."

                                                                                         
                if self.is_running:
                    self.finished.emit(result, confidence, wiki_summary, stage, pubmed_summary, self.domain)
            
                                                                  
            elif self.is_running:
                                                                                                  
                error_message = wiki or "No diagnosis could be made. The input did not match any known diseases."
                self.error.emit(error_message, self.domain)

        except Exception as e:
                                                                  
            print(f"An unexpected error occurred in the diagnosis worker: {e}")
            self.error.emit(f"An unexpected error occurred: {e}", self.domain)

    def stop(self):
        """Allows the main thread to signal this worker to stop."""
        self.is_running = False
