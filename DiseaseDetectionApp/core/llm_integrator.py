                                            
                              
                                                            
                                                                      
                                                                  

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
import openai
from typing import Optional, Dict, Any

class LLMIntegrator:
    """
    Integrates OpenAI GPT for enhanced AI responses in the chatbot and diagnosis explanations.
    Provides natural language generation for personalized treatment plans and context-aware queries.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM integrator with OpenAI API key.
        If no key provided, LLM features will be disabled and fallback to database search.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.llm_available = bool(self.api_key)

        if self.llm_available:
            openai.api_key = self.api_key
            self.model = "gpt-3.5-turbo"                                                            
        else:
            print("Warning: OpenAI API key not found. LLM features disabled. Set OPENAI_API_KEY environment variable for enhanced AI responses.")

        self.conversation_history = []                             
        self.cache = {}                                     

    def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response using GPT based on the query and optional context.
        Context can include disease data, user history, etc.
        Falls back to database search if LLM unavailable.
        """
        if not self.llm_available:
            return "LLM features are disabled. Please set OPENAI_API_KEY environment variable for enhanced AI responses. Falling back to database search."

                                                
        cache_key = (query, str(context) if context else None)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            messages = self._build_messages(query, context)
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
                timeout=10                                            
            )
            bot_response = response.choices[0].message['content'].strip()
                                                    
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": bot_response})
                                                       
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

                                               
            self.cache[cache_key] = bot_response
                                                       
            if len(self.cache) > 50:
                                                     
                oldest_keys = list(self.cache.keys())[:10]
                for key in oldest_keys:
                    del self.cache[key]

            return bot_response
        except openai.error.Timeout as e:
            return "Response timed out. Please try again or check your internet connection."
        except openai.error.RateLimitError as e:
            return "API rate limit reached. Please wait a moment and try again."
        except openai.error.AuthenticationError as e:
            return "Authentication failed. Please check your OpenAI API key."
        except Exception as e:
            return f"Sorry, I encountered an error generating a response: {str(e)}. Falling back to database search."

    def generate_treatment_plan(self, disease_data: Dict[str, Any], user_symptoms: str) -> str:
        """
        Generate a personalized treatment plan using LLM based on disease data and user symptoms.
        Falls back to database info if LLM unavailable.
        """
        if not self.llm_available:
            return f"Treatment information from database: {disease_data.get('solution', 'Consult a professional for treatment options.')}"
        prompt = f"""
        Based on the following disease information and user symptoms, generate a personalized treatment plan.
        Be empathetic, informative, and advise consulting a professional.

        Disease: {disease_data.get('name', 'Unknown')}
        Description: {disease_data.get('description', 'N/A')}
        Stages/Symptoms: {disease_data.get('stages', 'N/A')}
        Solution/Cure: {disease_data.get('solution', 'N/A')}
        Preventive Measures: {disease_data.get('preventive_measures', 'N/A')}

        User Symptoms: {user_symptoms}

        Provide a step-by-step treatment plan, including home remedies, when to seek medical help, and prevention tips.
        """
        return self.generate_response(prompt, {"type": "treatment_plan"})

    def explain_diagnosis(self, diagnosis_result: Dict[str, Any]) -> str:
        """
        Provide a detailed explanation of the diagnosis in natural language.
        Falls back to simple explanation if LLM unavailable.
        """
        if not self.llm_available:
            return f"Diagnosis: {diagnosis_result.get('name', 'Unknown')} with {diagnosis_result.get('confidence', 0)}% confidence. {diagnosis_result.get('description', 'Please consult a professional for detailed information.')}"
        prompt = f"""
        Explain the following diagnosis result in simple, understandable language.
        Make it patient-friendly and include what this means, potential causes, and next steps.

        Diagnosis: {diagnosis_result.get('name', 'Unknown')}
        Confidence: {diagnosis_result.get('confidence', 0)}%
        Description: {diagnosis_result.get('description', 'N/A')}
        Stages: {diagnosis_result.get('stages', 'N/A')}
        Solution: {diagnosis_result.get('solution', 'N/A')}
        """
        return self.generate_response(prompt, {"type": "diagnosis_explanation"})

    def _build_messages(self, query: str, context: Optional[Dict[str, Any]] = None) -> list:
        """
        Build the message list for OpenAI API, including system prompt and history.
        """
        system_prompt = """
        You are an AI assistant for a disease detection app. Provide accurate, helpful information about diseases in plants, humans, and animals.
        Always advise consulting professionals for medical advice. Be empathetic and informative.
        If information is from the database, prioritize it; otherwise, use general knowledge.
        """
        messages = [{"role": "system", "content": system_prompt}]

                                  
        messages.extend(self.conversation_history[-10:])                    

                                 
        if context:
            context_str = f"Context: {context}"
            messages.append({"role": "user", "content": context_str})

        messages.append({"role": "user", "content": query})
        return messages

    def reset_memory(self):
        """Reset conversation history for a new session."""
        self.conversation_history = []
        self.cache = {}                             
