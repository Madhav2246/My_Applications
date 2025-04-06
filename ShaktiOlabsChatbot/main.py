import json
import os
import re
import logging
from quiz_generator import QuizGenerator
from googletrans import Translator
from voice_handler import VoiceHandler
from text_handler import TextHandler
from difflib import SequenceMatcher  # Added for fuzzy matching
import pyttsx3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OLabsChatbot:
    def __init__(self):
        logging.info("Initializing OLabsChatbot...")
        self.translator = Translator()
        self.quiz_gen = QuizGenerator()

        self.supported_languages = {
            "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
            "kn": "Kannada", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi"
        }
        logging.info(f"Supported languages: {self.supported_languages}")

        self.voice_handler = VoiceHandler(self.supported_languages, self.set_language)
        self.text_handler = TextHandler(self.get_language)

        self.voice_enabled = False
        self.current_language = "en"
        self.current_subject = None
        self.current_topic = None
        self.quiz_completed = False
        self.last_quiz_weak_areas = []
        self.conversation_history = []
        self.last_intent = None
        self.in_session = False

        self.crawler_data = {"content_texts": [], "topics_map": {}}
        self.subjects = self.load_subjects()
        self.content_texts = self.crawler_data["content_texts"]
        self.topics_map = self.crawler_data["topics_map"]
        self.load_data()
        logging.info("OLabsChatbot initialization complete.")

    def set_language(self, lang_code):
        if lang_code not in self.supported_languages:
            logging.warning(f"Language code '{lang_code}' not supported. Defaulting to 'en'.")
            lang_code = "en"
        self.current_language = lang_code
        self.quiz_gen.set_language(lang_code)
        self.text_handler.current_language = self.current_language
        if hasattr(self.voice_handler, 'set_language'):
            self.voice_handler.set_language(lang_code)
        logging.info(f"Language set to: {self.current_language}")

    def get_language(self):
        return self.current_language

    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(script_dir, "olabs_data.json")
        try:
            logging.info(f"Attempting to load: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.crawler_data["content_texts"] = data.get("content_texts", [])
                if "topics_map" not in data or not data["topics_map"]:
                    logging.info("topics_map not found in olabs_data.json. Generating from content_texts.")
                    topics_map = {}
                    for text in self.crawler_data["content_texts"]:
                        if any(phrase in text.lower() for phrase in ["olabs", "developed by", "loading", "for teachers", "for students", "for schools", "contact us", "system requirements", "frequently asked questions", "main page", "workshops", "in the news"]):
                            continue
                        topic = text.strip()
                        if topic:
                            url_key = topic.lower().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")
                            topics_map[topic] = f"https://www.olabs.edu.in/{url_key}"
                    self.crawler_data["topics_map"] = topics_map
                else:
                    self.crawler_data["topics_map"] = data["topics_map"]
                self.content_texts = list(self.crawler_data["topics_map"].keys())
                self.topics_map = self.crawler_data["topics_map"]
                logging.info(f"Loaded {len(self.content_texts)} texts from {cache_file}")
                logging.info(f"Topics map: {self.topics_map}")
        except FileNotFoundError:
            logging.error(f"{cache_file} not found in {script_dir}. Initializing with empty data.")
            self.content_texts = []
            self.crawler_data["content_texts"] = []
            self.topics_map = {}
            self.crawler_data["topics_map"] = {}
        except json.JSONDecodeError:
            logging.error(f"{cache_file} is corrupted or invalid JSON.")
            self.content_texts = []
            self.crawler_data["content_texts"] = []
            self.topics_map = {}
            self.crawler_data["topics_map"] = {}
        except Exception as e:
            logging.error(f"Unexpected error while loading {cache_file}: {e}")
            self.content_texts = []
            self.crawler_data["content_texts"] = []
            self.topics_map = {}
            self.crawler_data["topics_map"] = {}

    def load_subjects(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(script_dir, "olabs_data.json")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                topics_map = data.get("topics_map", {})
                if not topics_map:
                    topics_map = self.crawler_data["topics_map"]
                topics = list(topics_map.keys())
                subjects = {"Physics": [], "Chemistry": [], "Biology": [], "Mathematics": [], "English": []}
                logging.info(f"Total topics to categorize: {len(topics)}")
                logging.info(f"Sample topics: {topics[:5]}")
                for topic in topics:
                    topic_lower = topic.lower()
                    if any(kw in topic_lower for kw in ["ohm", "resistance", "zener", "diode", "galvanometer", "metre bridge", "potentiometer", "sonometer", "mirror", "lens", "refraction", "refractive", "emf", "electrostatic", "magnetic", "calipers", "plane", "vectors", "newton", "resonance", "modulus", "gauge", "pendulum", "spherometer", "balance", "friction", "spring", "tension", "viscosity", "boyle", "heat", "motion", "bernoulli", "beats", "waves", "capillary", "energy", "pendulums", "resistance", "reflection", "field lines", "induction", "hooke", "archimedes", "bell jar", "pulse", "temperature", "conductivity", "pressure", "dropper", "velocity"]):
                        subjects["Physics"].append(topic)
                    elif any(kw in topic_lower for kw in ["lyophilic", "lyophobic", "emulsions", "inorganic", "kmno₄", "organic", "kinetics", "chromatography", "proteins", "functional groups", "thermochemistry", "emf", "carbohydrates", "oils", "fats", "aniline", "enthalpy", "anions", "laboratory", "melting", "estimation", "crystallization", "boiling", "equilibrium", "ph", "lassaigne", "cations", "oxalic", "filtration", "acids", "bases", "acetic", "reactivity", "indicator", "reaction", "saponification", "cleaning", "naoh", "bleaching", "sulphate", "oxidation", "esterification", "foaming", "sulphur", "temperature", "electrolytes", "metals", "soda", "mixtures", "conservation", "sublimate", "colloidal", "evaporation", "compressible", "rutherford", "exothermic", "endothermic", "periodic table"]):
                        subjects["Chemistry"].append(topic)
                    elif any(kw in topic_lower for kw in ["pollen", "turbidity", "soil", "amylase", "mitosis", "plant population", "pollutants", "gametogenesis", "blastula", "pedigree", "mendel", "hybridization", "xerophytes", "hydrophytes", "meiosis", "assortment", "staining", "disease", "gametophyte", "transpiration", "stomata", "osmosis", "dicot", "monocot", "plasmolysis", "respiration", "carbohydrates", "proteins", "urea", "sugar", "albumin", "bile", "root", "stem", "inflorescences", "animals", "microscope", "bacteria", "fungi", "leaf", "joints", "imbibition", "photosynthesis", "homology", "analogy", "reproduction", "embryo", "carbon dioxide", "phototropism", "geotropism", "fermentation", "propagation", "starch", "adulterant", "cells", "tissues", "mosquito", "plants", "parasite", "herbarium", "diseases", "pond", "micro-organisms"]):
                        subjects["Biology"].append(topic)
                    elif any(kw in topic_lower for kw in ["queue", "stack", "plot", "scalar", "quadrant", "origin", "projection", "interface", "constructors", "printers", "setters", "getters", "class", "phishing", "emis", "cubes", "largest", "random", "recursive", "taylor", "search", "binary", "math", "name", "min-max", "merge", "fibonacci", "palindrome", "factorial", "gcd", "quick", "bubble", "insertion", "selection", "interest", "swap", "lcm", "largest", "armstrong", "prime", "compute", "numbers"]):
                        subjects["Mathematics"].append(topic)
                    elif any(kw in topic_lower for kw in ["prepositions", "correction", "omission", "direction", "pronunciation", "passive", "active", "tense", "comprehension", "speech", "agreement", "singular", "plural"]):
                        subjects["English"].append(topic)
                    else:
                        logging.debug(f"Topic '{topic}' not categorized under any subject")
                logging.info(f"Loaded topics: {subjects}")
                has_topics = any(len(topics) > 0 for topics in subjects.values())
                if not has_topics:
                    logging.warning("No topics loaded from olabs_data.json. Using default topics.")
                    return {
                        "Physics": ["Motion", "Force and Laws of Motion", "Gravitation", "Work and Energy", "Sound", "Light - Reflection and Refraction", "Electricity", "Magnetism"],
                        "Chemistry": ["Properties of Acids and Bases"],
                        "Biology": [],
                        "Mathematics": ["Pythagoras Theorem", "Right Circular Cylinder"],
                        "English": ["Tense Conversion", "Direct and Indirect Speech"]
                    }
                return subjects
        except FileNotFoundError:
            logging.warning("olabs_data.json not found. Using default subjects.")
            return {
                "Physics": ["Motion", "Force and Laws of Motion", "Gravitation", "Work and Energy", "Sound", "Light - Reflection and Refraction", "Electricity", "Magnetism"],
                "Chemistry": ["Properties of Acids and Bases"],
                "Biology": [],
                "Mathematics": ["Pythagoras Theorem", "Right Circular Cylinder"],
                "English": ["Tense Conversion", "Direct and Indirect Speech"]
            }

    def format_markdown(self, text):
        # Preserve headings but remove extra spaces after #
        text = re.sub(r'^(#+)\s*(.*)$', r'\1 \2', text, flags=re.MULTILINE)
        
        # Remove Markdown asterisks for bold (**text**) and italic (*text*)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
        
        # Simplify numbered lists: Remove asterisks but keep the numbering
        text = re.sub(r'(\d+\.\s+)(.*?)(?=\n|$)', r'\1\2', text, flags=re.MULTILINE)
        
        # Simplify key-value pairs (e.g., "Current Language: English")
        text = re.sub(r'(\w+):(\s+.*)', r'\1: \2', text, flags=re.MULTILINE)
        
        # Remove asterisks from bullet points but keep the structure
        text = re.sub(r'-\s+\*(.*?)\*(?=\n|$)', r'- \1', text, flags=re.MULTILINE)
        
        return text.strip()

    def translate_text(self, text):
        if self.current_language == "en":
            return text
        try:
            translated = self.translator.translate(text, dest=self.current_language).text
            return translated
        except Exception as e:
            logging.error(f"Translation Error: {e}")
            return text

    def get_menu(self):
        menu = "# Welcome to Your Learning Adventure!\n\nHey there! I’m Shakti, your buddy for exploring science and beyond. What’s sparking your interest today?\n\n"
        menu += f"Current Language: {self.supported_languages.get(self.current_language, self.current_language).capitalize()}\n"
        menu += f"Current Subject: {self.current_subject or 'None'}\n"
        menu += f"Current Topic: {self.current_topic or 'None'}\n\n"
        menu += "Options:\n- Q: Jump into a Quiz\n- G: Get Guidance\n- W: Roadmap for Weak Spots\n- M: See This Menu\n- R: Replay Last Response\n- E: Wrap Up Section\n- X: Exit\n- L: Change Language\n- C: Change Mode\n"
        if self.quiz_completed and hasattr(self, 'in_session') and self.in_session:
            menu += "- N: Next Quiz Challenge\n"
        return self.format_markdown(self.translate_text(menu))

    def select_subject(self, prompt="Let’s start with a subject!"):
        response = f"# Pick a Subject\n\n{prompt}:\n\n"
        subjects = list(self.subjects.keys())
        for i, subject in enumerate(subjects, 1):
            response += f"{i}. {subject}\n"
        response += "\nSay the number, name, or ask for help!"
        if self.voice_enabled:
            self.voice_handler.output_response(response, self.translate_text, self.format_markdown)
            choice = self.voice_handler.recognize_speech(self.translate_text)
        else:
            self.text_handler.output_response(response, self.translate_text, self.format_markdown)
            choice = self.text_handler.get_input(self.translate_text).lower()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(subjects):
                return subjects[idx]
        except ValueError:
            choice = choice.strip()
            if choice in [s.lower() for s in subjects]:
                return next(s for s in subjects if s.lower() == choice)
            elif "help" in choice or "list" in choice:
                return self.list_available_subjects()
        return None

    def list_available_subjects(self):
        response = "# Available Subjects\n\nHere are the subjects you can explore:\n\n"
        for i, subject in enumerate(self.subjects.keys(), 1):
            response += f"{i}. {subject}\n"
        response += "\nPlease pick one by number or name!"
        if self.voice_enabled:
            self.voice_handler.output_response(response, self.translate_text, self.format_markdown)
            choice = self.voice_handler.recognize_speech(self.translate_text)
        else:
            self.text_handler.output_response(response, self.translate_text, self.format_markdown)
            choice = self.text_handler.get_input(self.translate_text).lower()
        try:
            idx = int(choice) - 1
            subjects = list(self.subjects.keys())
            if 0 <= idx < len(subjects):
                return subjects[idx]
        except ValueError:
            if choice in [s.lower() for s in self.subjects.keys()]:
                return next(s for s in self.subjects.keys() if s.lower() == choice)
        return None

    def list_topics(self, subject=None):
        subject = subject or self.current_subject
        if not subject:
            return self.translate_text("# Oops!\n\nWe need a subject first—try selecting a subject!"), False
        topics = self.subjects.get(subject, [])
        if not topics:
            response = f"# Topics for {subject}\n\nHmm, it seems I don’t have topics for {subject} yet. Let’s try another subject!"
            suggestions = [s for s in self.subjects.keys() if s != subject and self.subjects[s]]
            if suggestions:
                response += f"\nSuggestions: {', '.join(suggestions)}"
        else:
            response = f"# Topics for {subject}\n\nReady to explore? Here’s what’s available:\n\n"
            for i, topic in enumerate(topics, 1):
                url = self.topics_map.get(topic, "https://www.olabs.edu.in")
                response += f"{i}. {topic} - [Dive In]({url})\n"
        response += "\nPick one by number or name, or say 'help' for more options!"
        return self.format_markdown(self.translate_text(response))

    def select_topic(self, subject=None):
        subject = subject or self.current_subject
        if not subject:
            return None
        topics = self.subjects.get(subject, [])
        if not topics:
            return None
        response = self.list_topics(subject)
        if self.voice_enabled:
            self.voice_handler.output_response(response, self.translate_text, self.format_markdown)
            choice = self.voice_handler.recognize_speech(self.translate_text)
        else:
            self.text_handler.output_response(response, self.translate_text, self.format_markdown)
            choice = self.text_handler.get_input(self.translate_text).lower()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(topics):
                return topics[idx]
        except ValueError:
            if choice in [t.lower() for t in topics]:
                return next(t for t in topics if t.lower() == choice)
        return None

    def run_quiz_with_voice(self, topic):
        text = f"This is a description of {topic}. It involves understanding the core concepts and applications related to {topic}."
        self.quiz_gen.set_language(self.current_language)
        user_score, weak_concepts = self.quiz_gen.run_quiz(text)

        if user_score is None:
            return 0, ["Error in quiz generation"]

        if not hasattr(self, 'in_session') or not self.in_session:
            self.in_session = True
            self.session_topic = topic
            logging.info(f"Started quiz session for topic: {topic}")

        self.quiz_completed = True
        self.last_quiz_weak_areas = weak_concepts
        response = f"# Quiz Results Are In!\n\nTopic: {topic}\nScore: {user_score}/10\n"
        if weak_concepts:
            response += f"Boost These: {', '.join(weak_concepts)}\n"

        if user_score >= 7:
            response += f"\nFantastic job! You’ve mastered {topic} with a score of {user_score}/10!\n"
            response += "Let’s end this session and explore something new.\n- Say Q for a new Quiz\n- Say G for Guidance\n- Or say X to exit!"
            self.in_session = False
            self.quiz_completed = False
            self.current_subject = None
            self.current_topic = None
        else:
            response += "\nGreat effort! Let’s keep improving:\n- Say W for a Weak Area Roadmap\n- Say G for Guidance based on your results\n- Say N for another Quiz on this topic\n- Or choose another option!"

        if self.voice_enabled:
            self.voice_handler.output_response(response, self.translate_text, self.format_markdown)
        else:
            self.text_handler.output_response(response, self.translate_text, self.format_markdown)
        return user_score, weak_concepts

    def generate_roadmap(self, topic):
        if not self.last_quiz_weak_areas:
            response = f"# Roadmap Ready!\n\nNo weak areas identified yet. Take a quiz on '{topic}' with 'Q' to get started!"
        else:
            response = f"# Weak Spot Mastery Plan for {topic}!\n\nLet’s turn these into strengths:\n\n"
            for area in self.last_quiz_weak_areas:
                url = self.topics_map.get(area, "https://www.olabs.edu.in")
                response += f"### {area}\n- Explore: [Link]({url})\n- Power Up: Focus on understanding {area} with OLabs simulations and practice problems.\n"
            response += "\nYou’re unstoppable—where to next?"
        return self.format_markdown(self.translate_text(response))

    def provide_guidance(self, topic=None):
        if not topic:
            subject = self.select_subject("Let’s choose a subject for guidance!")
            if not subject:
                return self.translate_text("# Whoops!\n\nThat subject didn’t work—try again or say 'help'!"), False
            self.current_subject = subject
            topic = self.select_topic(subject)
            if not topic:
                return self.translate_text("# Oops!\n\nPlease pick a topic for guidance!"), False
            self.current_topic = topic

        response = f"# Your Guide to {topic}!\n\nBuckle up—we’re diving into {topic}!\n\n"
        if self.last_quiz_weak_areas and topic in self.last_quiz_weak_areas:
            response += f"Focus Area: You struggled with {topic} in your last quiz. Let’s strengthen it!\n"
        response += f"Overview: {topic} is an exciting topic that explores key concepts in your subject area.\n"
        response += "Key Concepts:\n- Core Idea: Understand the basics of {topic}.\n- Application: Apply {topic} in real-world scenarios.\n"
        response += f"Practical Applications: Use {topic} to solve problems or conduct experiments.\n"
        response += f"Explore More: [OLabs {topic}]({self.topics_map.get(topic, 'https://www.olabs.edu.in')})\n\n"
        response += "What’s your next move—quiz, roadmap, or something fresh?"
        return self.format_markdown(self.translate_text(response))

    def handle_intent(self, query):
        query = query.lower().strip() if query else ""
        self.last_intent = query

        if not query:
            return self.translate_text("# Hmm...\n\nI didn’t catch that—try again or say 'help'!"), False

        if query in ["x", "exit"]:
            response = "# See Ya!\n\nThanks for the journey—keep shining! Come back anytime!"
            self.in_session = False
            return response, True

        if any(kw in query for kw in ["q", "start quiz", "quiz me", "take quiz"]):
            if not all([self.current_subject, self.current_topic]):
                subject = self.select_subject("Let’s start with a subject for your quiz!")
                if not subject:
                    return self.translate_text("# Whoops!\n\nThat subject didn’t work—try again or say 'help'!"), False
                self.current_subject = subject
                topic = self.select_topic(subject)
                if not topic:
                    return self.translate_text("# Oops!\n\nPlease pick a topic for your quiz!"), False
                self.current_topic = topic
            score, weak_concepts = self.run_quiz_with_voice(self.current_topic)
            self.last_quiz_weak_areas = weak_concepts
            self.quiz_completed = True
            return "", False

        if any(kw in query for kw in ["g", "guidance", "guide me", "what is", "learn about"]):
            topic_match = re.search(r'(?:guide me on|what is|learn about)\s+(.+)', query)
            topic = topic_match.group(1) if topic_match else self.current_topic
            if not topic:
                subject = self.select_subject("Let’s choose a subject for guidance!")
                if not subject:
                    return self.translate_text("# Whoops!\n\nThat subject didn’t work—try again or say 'help'!"), False
                self.current_subject = subject
                topic = self.select_topic(subject)
                if not topic:
                    return self.translate_text("# Oops!\n\nPlease pick a topic for guidance!"), False
                self.current_topic = topic
            response = self.provide_guidance(topic)
            return response, False

        if query in ["w", "weak area roadmap", "improve"]:
            if not self.quiz_completed:
                return self.translate_text("# Quiz First!\n\nTake a quiz with 'Q' to spot your weak areas!"), False
            if not hasattr(self, 'in_session') or not self.in_session:
                return self.translate_text("# Session Ended!\n\nPlease start a new quiz session with 'Q' to continue!"), False
            subject = self.current_subject or self.select_subject("Let’s choose a subject for your roadmap!")
            if not subject:
                return self.translate_text("# Whoops!\n\nThat subject didn’t work—try again or say 'help'!"), False
            self.current_subject = subject
            topic = self.current_topic or self.select_topic(subject)
            if not topic:
                return self.translate_text("# Oops!\n\nPlease pick a topic for your roadmap!"), False
            self.current_topic = topic
            return self.generate_roadmap(topic), False

        if query in ["l", "language", "change language"] or "change language to" in query:
            response = "# Change Language\n\nChoose a language:\n"
            for i, (code, name) in enumerate(self.supported_languages.items(), 1):
                response += f"{i}. {name} ({code})\n"
            response += "\nSay the number, code (e.g., 'te' for Telugu), or full name like 'Telugu'!"
            if self.voice_enabled:
                self.voice_handler.output_response(response, self.translate_text, self.format_markdown)
                choice = self.voice_handler.recognize_speech(self.translate_text, is_language_change=True)
                if choice and "change language to" in choice:
                    lang_match = re.search(r'change language to\s+(.+)', choice.lower())
                    if lang_match:
                        choice = lang_match.group(1).strip()
            else:
                self.text_handler.output_response(response, self.translate_text, self.format_markdown)
                choice = self.text_handler.get_input(self.translate_text).lower()
                if "change language to" in query:
                    lang_match = re.search(r'change language to\s+(.+)', query.lower())
                    if lang_match:
                        choice = lang_match.group(1).strip()

            lang_code = None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.supported_languages):
                    lang_code = list(self.supported_languages.keys())[idx]
            except (ValueError, TypeError):
                choice = choice.strip().lower() if choice else ""
                lang_code = next((code for code, name in self.supported_languages.items()
                                if code == choice or name.lower() == choice or SequenceMatcher(None, name.lower(), choice).ratio() > 0.8), None)

            if lang_code:
                self.set_language(lang_code)
                self.voice_handler.set_language(lang_code)
                return f"# Language Updated!\n\nNow using {self.supported_languages[lang_code]} ({lang_code}). What’s next?", False
            return self.translate_text("# Oops!\n\nInvalid language choice—try a number, code like 'te', or full name like 'Telugu'!"), False

        if query in ["c", "change mode", "mode"]:
            response = "# Change Mode\n\nChoose your mode:\n1. Text\n2. Voice\n\nSay the number or mode name!"
            if self.voice_enabled:
                self.voice_handler.output_response(response, self.translate_text, self.format_markdown)
                choice = self.voice_handler.recognize_speech(self.translate_text)
            else:
                self.text_handler.output_response(response, self.translate_text, self.format_markdown)
                choice = self.text_handler.get_input(self.translate_text).lower()

            if choice in ["1", "text"]:
                self.voice_enabled = False
                return self.translate_text("# Mode Updated!\n\nNow in Text mode. What’s next?"), False
            elif choice in ["2", "voice"]:
                self.voice_enabled = True
                return self.translate_text("# Mode Updated!\n\nNow in Voice mode. What’s next?"), False
            return self.translate_text("# Oops!\n\nInvalid mode choice—try 1 for Text or 2 for Voice!"), False

        if query in ["m", "menu", "show menu"]:
            return self.get_menu(), False

        if query in ["r", "read last output", "repeat"]:
            if self.conversation_history:
                return self.conversation_history[-1], False
            return self.translate_text("# Hmm...\n\nNo previous response to replay! Let’s start fresh!"), False

        if query in ["n", "next quiz", "another quiz"]:
            if not hasattr(self, 'in_session') or not self.in_session:
                return self.translate_text("# Session Ended!\n\nPlease start a new quiz session with 'Q' to continue!"), False
            if not self.quiz_completed:
                return self.translate_text("# Quiz Check!\n\nFinish a quiz with 'Q' first!"), False
            score, weak_concepts = self.run_quiz_with_voice(self.current_topic)
            self.last_quiz_weak_areas = weak_concepts
            return "", False

        if query in ["e", "end section", "finish"]:
            self.quiz_completed = False
            self.current_subject = None
            self.current_topic = None
            self.in_session = False
            return self.translate_text("# Section Complete!\n\nAwesome work! What’s our next adventure?"), False

        if query in ["help"]:
            response = "# Help\n\nI’m Shakti, your OLabs assistant! Here’s how to use me:\n- Say 'Q' or 'quiz me' to start a quiz.\n- Say 'G' or 'guide me' for guidance.\n- Say 'W' for a weak area roadmap after a quiz.\n- Say 'M' or 'show menu' to see options.\n- Say 'L' or 'change language' to switch languages.\n- Say 'C' or 'change mode' to toggle text/voice.\n- Say 'X' or 'exit' to quit.\nWhat would you like to do?"
            return self.format_markdown(self.translate_text(response)), False

        if any(kw in query for kw in ["what is", "tell me about", "explain"]):
            topic = re.search(r'(?:what is|tell me about|explain)\s+(.+)', query)
            if topic:
                topic = topic.group(1)
                self.current_topic = topic
                response = self.provide_guidance(topic)
                return response, False
            return self.translate_text("# Hmm...\n\nPlease specify a topic, e.g., 'what is sound?'"), False

        response = self.translate_text(f"# Let’s Get Creative!\n\nI didn’t quite get that, {self.current_subject or 'friend'}! Try commands like 'quiz me', 'guide me', or 'show menu'. Need help? Say 'help'!")
        return response, False

    def run(self):
        welcome_message = "Hi, I am your OLabs assistant, Shakti, created by Team VirtueVerse. Let’s embark on a learning adventure together!"
        print("Shakti by Team VirtueVerse - OLabs Assistant | Date: March 16, 2025")
        self.voice_handler.speak_response(self.translate_text(welcome_message))

        prompt = "Hey, explorer! I’m Shakti, your learning sidekick. Text or voice—how do we roll?"
        if self.voice_enabled:
            self.voice_handler.output_response(prompt, self.translate_text, self.format_markdown)
        else:
            self.text_handler.output_response(prompt, self.translate_text, self.format_markdown)

        while True:
            if self.voice_enabled:
                mode = self.voice_handler.recognize_speech(self.translate_text)
                mode = mode.lower() if mode else ""
            else:
                mode = self.text_handler.get_input(self.translate_text).lower()
            if mode in ["text", "voice"]:
                if mode == "text":
                    self.voice_enabled = False
                else:
                    self.voice_enabled = True
                break
            if self.voice_enabled:
                self.voice_handler.output_response("Please choose 'text' or 'voice'!", self.translate_text, self.format_markdown)
            else:
                self.text_handler.output_response("Please choose 'text' or 'voice'!", self.translate_text, self.format_markdown)

        lang_prompt = "Great! Now, please choose a language (e.g., 'te' for Telugu, 'en' for English, or say the full name like 'Telugu'):"
        if self.voice_enabled:
            self.voice_handler.output_response(lang_prompt, self.translate_text, self.format_markdown)
            lang = self.voice_handler.recognize_speech(self.translate_text, is_language_change=True)
            lang = lang.lower() if lang else ""
        else:
            self.text_handler.output_response(lang_prompt, self.translate_text, self.format_markdown)
            lang = self.text_handler.get_input(self.translate_text).lower()

        if "change language to" in lang:
            lang_match = re.search(r'change language to\s+(.+)', lang)
            if lang_match:
                lang = lang_match.group(1).strip()

        lang_code = next((code for code, name in self.supported_languages.items() if name.lower() == lang or code == lang), None)
        if lang_code:
            self.set_language(lang_code)
        else:
            try:
                idx = int(lang) - 1
                if 0 <= idx < len(self.supported_languages):
                    lang_code = list(self.supported_languages.keys())[idx]
                    self.set_language(lang_code)
                else:
                    self.set_language("en")
                    logging.warning(f"Invalid language index '{lang}'. Defaulting to 'en'.")
            except ValueError:
                self.set_language("en")
                logging.warning(f"Invalid language input '{lang}'. Defaulting to 'en'.")

        menu = self.get_menu()
        if self.voice_enabled:
            self.voice_handler.output_response(menu, self.translate_text, self.format_markdown)
        else:
            self.text_handler.output_response(menu, self.translate_text, self.format_markdown)

        while True:
            if self.voice_enabled:
                query = self.voice_handler.recognize_speech(self.translate_text, is_language_change=("change language" in self.last_intent))
            else:
                query = self.text_handler.get_input(self.translate_text)
            response, should_exit = self.handle_intent(query)
            self.conversation_history.append(response)
            if self.voice_enabled:
                self.voice_handler.output_response(response, self.translate_text, self.format_markdown)
            else:
                self.text_handler.output_response(response, self.translate_text, self.format_markdown)
            if should_exit:
                break

if __name__ == "__main__":
    try:
        chatbot = OLabsChatbot()
        chatbot.run()
    except Exception as e:
        logging.error(f"Chatbot failed: {e}")
        print("An unexpected error occurred. Check the logs for details.")