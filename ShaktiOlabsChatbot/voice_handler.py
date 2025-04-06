import speech_recognition as sr
import pyttsx3
import pygame.mixer
import tempfile
import keyboard
import os
import time
import re
import logging
from difflib import SequenceMatcher

pygame.mixer.init()

class VoiceHandler:
    def __init__(self, supported_languages, set_language_callback):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.current_language = "en"
        self.supported_languages = supported_languages
        self.set_language_callback = set_language_callback
        self.engine = pyttsx3.init()  # Use pyttsx3 for offline TTS
        self.engine.setProperty('rate', 150)  # Increase speech rate for faster output
        self.command_aliases = {
            "show": "show menu", "menu": "show menu", "quiz": "quiz me", "guide": "guide me",
            "weak": "weak area roadmap", "roadmap": "weak area roadmap", "repeat": "replay last response",
            "exit": "exit", "language": "change language", "mode": "change mode", "next": "next quiz",
            "end": "end section", "help": "help"
        }
        logging.info("VoiceHandler initialized with pyttsx3.")

    def set_language(self, lang_code):
        if lang_code in self.supported_languages:
            self.current_language = lang_code
            # Map to pyttsx3 language codes (approximate mapping)
            lang_map = {"en": "en", "hi": "hi", "ta": "ta", "te": "te", "ml": "ml", "kn": "kn",
                        "bn": "bn", "mr": "mr", "gu": "gu", "pa": "pa"}
            self.engine.setProperty('voice', lang_map.get(lang_code, "en"))
            logging.info(f"VoiceHandler language set to: {lang_code}")
        else:
            logging.warning(f"Language {lang_code} not supported in VoiceHandler. Keeping {self.current_language}.")

    def output_response(self, response, translate_text_func, format_markdown_func):
        translated_response = translate_text_func(response)
        formatted_response = format_markdown_func(translated_response)
        print(formatted_response)
        self.speak_response(translated_response)

    def speak_response(self, text):
        try:
            clean_text = re.sub(r'[*\[\]#]+', '', text)
            logging.info(f"Speaking translated text: {clean_text} in {self.current_language}")
            self.engine.say(clean_text)
            self.engine.runAndWait()
            if keyboard.is_pressed("q"):
                self.engine.stop()
        except Exception as e:
            logging.error(f"TTS Error: {e}")
            print("TTS failed. Switching to text mode.")
            self.engine.stop()

    def recognize_speech(self, translate_text_func, is_language_change=False):
        max_attempts = 2  # Reduced to speed up
        for attempt in range(max_attempts):
            with self.microphone as source:
                print(f"Listening... (Press 'q' to stop) [Attempt {attempt + 1}/{max_attempts}]")
                self.speak_response(translate_text_func("I’m listening—go ahead!"))
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Faster noise adjustment
                try:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)  # Reduced timeouts
                except sr.WaitTimeoutError:
                    self.speak_response(translate_text_func("I didn’t hear anything—try again!"))
                    continue

            lang_codes = ["en-IN", "hi-IN", "ta-IN", "te-IN", "ml-IN", "kn-IN", "bn-IN", "mr-IN", "gu-IN", "pa-IN"]
            for lang in lang_codes:
                try:
                    text = self.recognizer.recognize_google(audio, language=lang).lower()
                    detected_lang = lang.split('-')[0]
                    # Only update language if explicitly changing
                    if is_language_change and detected_lang in self.supported_languages and detected_lang != self.current_language:
                        self.current_language = detected_lang
                        self.set_language_callback(detected_lang)
                        self.speak_response(translate_text_func(f"Language updated to {self.supported_languages[detected_lang]}!"))
                    best_match = max(self.command_aliases.keys(), key=lambda x: SequenceMatcher(None, x, text).ratio()) if text else ""
                    matched_command = self.command_aliases.get(best_match, text) if SequenceMatcher(None, best_match, text).ratio() > 0.6 else text
                    logging.info(f"Recognized speech in {detected_lang}: {text} (Matched to: {matched_command})")
                    print(f"You said ({self.current_language}): {matched_command}")
                    self.speak_response(translate_text_func(f"You said: {matched_command}"))
                    return matched_command.strip()
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    logging.error(f"Speech recognition failed: {e}")
                    self.speak_response(translate_text_func("Speech recognition failed—try again!"))
                    return None
            if attempt < max_attempts - 1:
                self.speak_response(translate_text_func("Oops, I didn’t catch that—try again!"))
        return translate_text_func("Sorry, I couldn’t understand—switch to text mode or try later!")
