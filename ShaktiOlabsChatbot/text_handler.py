# text_handler.py
import logging

class TextHandler:
    def __init__(self, get_language_callback):
        self.current_language = "en"  # Will be updated by the callback
        self.get_language_callback = get_language_callback

    @property
    def current_language(self):
        """Get the current language from the chatbot."""
        return self.get_language_callback()

    @current_language.setter
    def current_language(self, value):
        """Setter for current_language (updated by chatbot)."""
        self._current_language = value

    def get_input(self, translate_text_func):
        """Get text input from the user."""
        return input(translate_text_func("Your Query: ")).strip()

    def output_response(self, response, translate_text_func, format_markdown_func):
        """Display the response to the user."""
        formatted_response = format_markdown_func(translate_text_func(response))
        print(formatted_response)