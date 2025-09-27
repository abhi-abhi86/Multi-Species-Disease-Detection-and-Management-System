# DiseaseDetectionApp/core/worker.py
from PyQt6.QtCore import QObject, pyqtSignal
from core.ml_processor import MLProcessor, predict_from_symptoms
from core.ncbi_integration import get_pubmed_summary # Import the new function

class DiagnosisWorker(QObject):
    """
    Worker object to run the diagnosis in a separate thread.
    Now includes caching for PubMed results and more robust network error handling.
    """
    # Class-level cache to store PubMed results across different worker instances
    pubmed_cache = {}
    
    # Signal to emit when diagnosis is complete
    # Emits: result_dict, confidence_float, wiki_summary_str, predicted_stage_str, pubmed_summary_str, domain_str
    finished = pyqtSignal(dict, float, str, str, str, str)
    # Signal to emit if an error occurs
    # Emits: error_message_str, domain_str
    error = pyqtSignal(str, str)
    # Signal to emit progress updates
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
        Starts the diagnosis process, fetches research data upon completion,
        and utilizes a cache to avoid redundant network requests.
        """
        try:
            result, confidence, wiki, stage = None, 0, "", ""
            if self.symptoms:
                self.progress.emit("Analyzing symptoms...")
                result, confidence, wiki, stage = predict_from_symptoms(
                    self.symptoms, self.domain, self.database
                )
            elif self.image_path:
                self.progress.emit("Analyzing image... Please wait.")
                result, confidence, wiki, stage = self.ml_processor.predict_from_image(
                    self.image_path, self.domain, self.database
                )
            else:
                self.error.emit("No input provided (image or symptoms).", self.domain)
                return

            if self.is_running and result:
                disease_name = result['name']
                pubmed_summary = ""

                # Check cache first
                if disease_name in self.pubmed_cache:
                    self.progress.emit("Fetching research from cache...")
                    pubmed_summary = self.pubmed_cache[disease_name]
                else:
                    # If not in cache, fetch from network with error handling
                    self.progress.emit("Fetching recent research from PubMed...")
                    try:
                        # This is the actual network call
                        pubmed_summary = get_pubmed_summary(disease_name)
                        # Store result in cache for next time
                        self.pubmed_cache[disease_name] = pubmed_summary
                    except Exception as e:
                        # If network fails, provide a graceful message instead of crashing
                        print(f"Network error while fetching from PubMed: {e}")
                        pubmed_summary = "Could not retrieve online research data. Please check your internet connection."
                
                if self.is_running: # Check again in case the process was stopped
                    self.finished.emit(result, confidence, wiki, stage, pubmed_summary, self.domain)
            elif self.is_running:
                # If no result was found from the initial diagnosis
                self.error.emit("No diagnosis could be made. The AI model could not identify a matching disease.", self.domain)

        except Exception as e:
            self.error.emit(f"An unexpected error occurred during the diagnosis process: {e}", self.domain)

    def stop(self):
        self.is_running = False

