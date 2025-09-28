# DiseaseDetectionApp/core/worker.py
from PyQt6.QtCore import QObject, pyqtSignal
from core.ml_processor import predict_from_symptoms
from core.ncbi_integration import get_pubmed_summary

class DiagnosisWorker(QObject):
    """
    Worker object to run the diagnosis in a separate thread, preventing the UI from freezing.
    It now includes caching for PubMed results to reduce redundant network calls.
    """
    # A class-level cache to store PubMed results for the duration of the app session.
    pubmed_cache = {}
    
    # Signal to emit when diagnosis is complete.
    # Emits: result_dict, confidence_float, wiki_summary_str, predicted_stage_str, pubmed_summary_str, domain_str
    finished = pyqtSignal(dict, float, str, str, str, str)
    
    # Signal to emit if an error occurs.
    # Emits: error_message_str, domain_str
    error = pyqtSignal(str, str)
    
    # Signal to emit progress updates to the UI.
    progress = pyqtSignal(str)

    def __init__(self, ml_processor, image_path, symptoms, domain, database):
        super().__init__()
        self.ml_processor = ml_processor
        self.image_path = image_path
        self.symptoms = symptoms
        self.domain = domain
        self.database = database
        self.is_running = True # A flag to allow stopping the worker gracefully.

    def run(self):
        """
        Executes the diagnosis process. It chooses between image or symptom analysis,
        fetches external data, and emits the results.
        """
        try:
            result, confidence, wiki, stage = None, 0.0, "", ""
            
            # Determine whether to use image or symptom-based prediction.
            # The UI logic should ensure that one of these is always present.
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
                # This should not happen if the UI logic is correct, but it's a safe fallback.
                self.error.emit("No input (image or symptoms) was provided to the worker.", self.domain)
                return

            # If the diagnosis was successful and the worker hasn't been stopped.
            if self.is_running and result:
                disease_name = result.get('name', 'Unknown Disease')
                pubmed_summary = ""

                # Check the cache for PubMed data first to avoid network calls.
                if disease_name in self.pubmed_cache:
                    self.progress.emit("Fetching research data from cache...")
                    pubmed_summary = self.pubmed_cache[disease_name]
                else:
                    self.progress.emit("Fetching recent research from PubMed...")
                    try:
                        # This is the actual network call.
                        pubmed_summary = get_pubmed_summary(disease_name)
                        # Store the result in the cache for next time.
                        self.pubmed_cache[disease_name] = pubmed_summary
                    except Exception as e:
                        # Gracefully handle network failures.
                        print(f"Network error while fetching from PubMed: {e}")
                        pubmed_summary = "Could not retrieve online research data. Please check your internet connection."
                
                # Check again in case the user closed the app during the network request.
                if self.is_running:
                    self.finished.emit(result, confidence, wiki, stage, pubmed_summary, self.domain)
            
            # Handle cases where diagnosis ran but found no match.
            elif self.is_running:
                # The 'wiki' variable often contains the reason for failure from the ml_processor.
                error_message = wiki or "No diagnosis could be made. The input did not match any known diseases."
                self.error.emit(error_message, self.domain)

        except Exception as e:
            # A general catch-all for any other unexpected errors.
            print(f"An unexpected error occurred in the diagnosis worker: {e}")
            self.error.emit(f"An unexpected error occurred: {e}", self.domain)

    def stop(self):
        """Allows the main thread to signal this worker to stop."""
        self.is_running = False
