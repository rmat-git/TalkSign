import os
import threading
import google.generativeai as genai

class GeminiHandler:
    def __init__(self, api_key, log_callback=print):
        self.log = log_callback
        self.model = None
        self.api_key = api_key
        self.is_ready = False
        
        # Initialize in a separate thread to prevent app startup lag
        threading.Thread(target=self._initialize, daemon=True).start()

    def _initialize(self):
        try:
            os.environ["GOOGLE_API_KEY"] = self.api_key
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.is_ready = True
            self.log("System: Gemini 2.5 Flash Connected (Background Init).")
        except Exception as e:
            self.log(f"Gemini Init Error: {e}")
            self.is_ready = False

    def translate(self, raw_text):
        """
        Sends raw tokens to Gemini and returns the corrected sentence.
        Returns the original text if the API fails or isn't ready.
        """
        if not self.is_ready or not self.model:
            self.log("Gemini Not Ready: Skipping translation.")
            return raw_text

        try:
            self.log(f"Gemini: Refining '{raw_text}'...")
            prompt = f"Translate ASL Gloss to English: \"{raw_text}\". Output ONLY the sentence."
            
            # Use generate_content directly (blocking call, meant to be run in a thread)
            response = self.model.generate_content(prompt)
            
            if response.text:
                cleaned_text = response.text.strip()
                self.log(f"Gemini Result: {cleaned_text}")
                return cleaned_text
            
        except Exception as e:
            self.log(f"Gemini Request Failed: {e}")
        
        return raw_text