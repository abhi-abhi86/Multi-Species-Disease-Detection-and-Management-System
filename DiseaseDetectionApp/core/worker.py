# DiseaseDetectionApp/core/worker.py
from PyQt6.QtCore import QObject, pyqtSignal
from core.ml_processor import MLProcessor, predict_from_symptoms

class DiagnosisWorker(QObject):
    """
    Worker object to run the diagnosis in a separate thread.
    """
    # Signal to emit when diagnosis is complete
    # Emits: result_dict, confidence_float, wiki_summary_str, predicted_stage_str, domain_str
    finished = pyqtSignal(dict, float, str, str, str)
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
        Starts the diagnosis process.
        """
        try:
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
                # This case should ideally be caught in the UI, but we handle it here too.
                self.error.emit("No input provided (image or symptoms).", self.domain)
                return

            if self.is_running:
                if result:
                    self.finished.emit(result, confidence, wiki, stage, self.domain)
                else:
                    self.error.emit("No diagnosis could be made. The AI model could not identify a matching disease.", self.domain)

        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {e}", self.domain)

    def stop(self):
        self.is_running = False

