






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
from core.wikipedia_integration import get_wikipedia_summary
import json





sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtCore import QObject, Signal
from core.ml_processor import predict_from_symptoms
from core.ncbi_integration import get_pubmed_summary, generate_ncbi_report

class DiagnosisWorker(QObject):
    """
    Worker object to run the diagnosis in a separate thread, preventing the UI from freezing.
    It now includes caching for PubMed results to reduce redundant network calls.
    """
    _in_memory_cache = {}
    _cache_file = os.path.join(os.path.dirname(__file__), '..', 'cache.json')
    _persistent_cache = {}

    finished = Signal(dict, float, str, str, str, str)




    error = Signal(str, str)


    progress = Signal(str, str)  # message, domain


    def __init__(self, ml_processor, image_path, symptoms, domain, database):
        super().__init__()
        self.ml_processor = ml_processor
        self.image_path = image_path
        self.symptoms = symptoms
        self.domain = domain
        self.database = database
        self.is_running = True
        self._load_persistent_cache()

    @classmethod
    def _load_persistent_cache(cls):
        if not cls._persistent_cache:
            try:
                if os.path.exists(cls._cache_file):
                    with open(cls._cache_file, 'r', encoding='utf-8') as f:
                        cls._persistent_cache = json.load(f)
                        print("Loaded persistent cache with", len(cls._persistent_cache), "items.")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load persistent cache file: {e}")
                cls._persistent_cache = {}

    def run(self):
        """
        Executes the diagnosis process. It chooses between image or symptom analysis,
        fetches external data, and emits the results.
        """
        try:
            result, confidence, wiki, stage = None, 0.0, "", ""



            if self.image_path:
                self.progress.emit("Analyzing image with AI model... Please wait.", self.domain)
                result, confidence, wiki, stage = self.ml_processor.predict_from_image(
                    self.image_path, self.domain, self.database
                )
            elif self.symptoms:
                self.progress.emit("Analyzing symptoms...", self.domain)
                result, confidence, wiki, stage = predict_from_symptoms(
                    self.symptoms, self.domain, self.database
                )
            else:

                self.error.emit("No input (image or symptoms) was provided to the worker.", self.domain)
                return


            if self.is_running and result:
                disease_name = result.get('name', 'Unknown Disease')

                # Centralize all network calls here to leverage caching
                cache_key = f"wiki_{disease_name}"
                if cache_key in self._in_memory_cache:
                    self.progress.emit("Fetching Wikipedia data from cache...", self.domain)
                    wiki_summary = self._in_memory_cache[cache_key]
                elif cache_key in self._persistent_cache:
                    self.progress.emit("Fetching Wikipedia data from persistent cache...", self.domain)
                    wiki_summary = self._persistent_cache[cache_key]
                    self._in_memory_cache[cache_key] = wiki_summary
                else:
                    self.progress.emit("Fetching summary from Wikipedia...", self.domain)
                    wiki_summary = get_wikipedia_summary(disease_name)
                    if wiki_summary:
                        self._add_to_cache(cache_key, wiki_summary)

                cache_key = f"pubmed_{disease_name}"
                if cache_key in self._in_memory_cache:
                    self.progress.emit("Fetching research data from cache...", self.domain)
                    pubmed_summary = self._in_memory_cache[cache_key]
                elif cache_key in self._persistent_cache:
                    self.progress.emit("Fetching research data from persistent cache...", self.domain)
                    pubmed_summary = self._persistent_cache[cache_key]
                    self._in_memory_cache[cache_key] = pubmed_summary
                else:
                    self.progress.emit("Fetching recent research from PubMed...", self.domain)
                    try:
                        pubmed_summary = get_pubmed_summary(disease_name, domain=self.domain)
                        self._add_to_cache(cache_key, pubmed_summary)
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

    @classmethod
    def _add_to_cache(cls, key, value):
        """Adds an item to both in-memory and persistent caches."""
        cls._in_memory_cache[key] = value
        cls._persistent_cache[key] = value
        try:
            with open(cls._cache_file, 'w', encoding='utf-8') as f:
                json.dump(cls._persistent_cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not write to persistent cache file: {e}")
