import os
import sys
from rank_bm25 import BM25Okapi

# Ensure the parent directory is in the system path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_handler import load_database

class SearchEngine:
    """
    An in-memory search engine using the BM25 algorithm for fast and relevant text search
    across the entire disease database.
    """

    def __init__(self):
        """
        Initializes the search engine by loading all disease data, creating a corpus,
        and building the BM25 index.
        """
        print("Initializing search engine...")
        self.disease_data = load_database()
        self.corpus = []
        self.doc_map = []

        if not self.disease_data:
            print("Warning: No disease data found for search engine.")
            self.bm25 = None
            return

        for disease in self.disease_data:
            # Combine relevant text fields into a single document for indexing
            name = disease.get('name', '')
            description = disease.get('description', '')
            causes = disease.get('causes', '')
            solution = disease.get('solution', '')
            
            # Handle list-based preventive measures
            preventive_measures = disease.get('preventive_measures', [])
            preventive_str = ' '.join(preventive_measures) if isinstance(preventive_measures, list) else str(preventive_measures)

            full_text = ' '.join([name, description, causes, solution, preventive_str])
            self.corpus.append(full_text)
            self.doc_map.append(disease)

        # Tokenize the corpus (split strings into lists of words)
        tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]

        # Create and train the BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Search engine initialized with {len(self.disease_data)} documents.")

    def search(self, query: str, top_n: int = 10):
        """
        Searches the indexed data for a given query.
        
        Args:
            query (str): The search query string.
            top_n (int): The maximum number of results to return.

        Returns:
            list: A ranked list of the original disease data dictionaries.
        """
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get the top N indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]
        
        # Return the corresponding original disease objects
        return [self.doc_map[i] for i in top_indices if doc_scores[i] > 0]