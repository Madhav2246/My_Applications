import random
import numpy as np
from collections import defaultdict
import google.generativeai as genai
from googletrans import Translator, LANGUAGES  # Updated import

# RL parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.5

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
NUM_DIFFICULTIES = len(DIFFICULTY_LEVELS)
NUM_PERFORMANCE_BUCKETS = 3
NUM_MASTERY_LEVELS = 3
STATE_SPACE_SIZE = NUM_DIFFICULTIES * NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS
NUM_ACTIONS = 3

Q = np.zeros((STATE_SPACE_SIZE, NUM_ACTIONS))
for difficulty in range(NUM_DIFFICULTIES):
    for performance in range(NUM_PERFORMANCE_BUCKETS):
        for mastery in range(NUM_MASTERY_LEVELS):
            state_idx = difficulty * (NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS) + performance * NUM_MASTERY_LEVELS + mastery
            if performance == 0:
                Q[state_idx][0] = 0.5
            elif performance == 2:
                Q[state_idx][2] = 0.5

concept_mastery = defaultdict(list)

API_KEY = "AIzaSyCzmpRdo3Qo7VT_eb5R9GJgey_HnO5xAc0"  # Replace with your actual API key
genai.configure(api_key=API_KEY)
translator = Translator()

class QuizGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.current_difficulty = 1
        self.previous_score = 5
        self.performance_bucket = self.get_performance_bucket(self.previous_score)
        self.mastery_level = 1
        self.epsilon = EPSILON
        self.current_language = "en"

    def set_language(self, lang):
        self.current_language = lang if lang in LANGUAGES else "en"  # Updated to use LANGUAGES directly

    def get_state_index(self, difficulty, performance, mastery):
        return difficulty * (NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS) + performance * NUM_MASTERY_LEVELS + mastery

    def get_performance_bucket(self, score):
        if score <= 3:
            return 0
        elif score <= 7:
            return 1
        else:
            return 2

    def get_mastery_level(self, concepts):
        if not concepts:
            return 1
        total_correct = sum(sum(concept_mastery.get(concept, [])) for concept in concepts)
        total_questions = sum(len(concept_mastery.get(concept, [])) for concept in concepts)
        if total_questions == 0:
            return 1
        accuracy = total_correct / total_questions
        return 0 if accuracy < 0.4 else 1 if accuracy < 0.7 else 2

    def choose_action(self, state_index, score):
        if score < 5:
            return 0
        elif score >= 8:
            return 2
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(NUM_ACTIONS))
        return np.argmax(Q[state_index])

    def update_q_table(self, state_index, action, reward, next_state_index):
        old_value = Q[state_index][action]
        next_max = np.max(Q[next_state_index])
        Q[state_index][action] = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)

    def calculate_reward(self, score, previous_score, difficulty):
        base_reward = 2 if score >= 8 else 1 if score >= 5 else 0 if score >= 3 else -2
        improvement_reward = 1 if score > previous_score else 0 if score == previous_score else -1
        difficulty_bonus = difficulty * 0.5 if score >= 5 else -difficulty * 0.5
        return base_reward + improvement_reward + difficulty_bonus

    def adjust_difficulty(self, action):
        if action == 0:
            return max(0, self.current_difficulty - 1)
        elif action == 1:
            return self.current_difficulty
        elif action == 2:
            return min(NUM_DIFFICULTIES - 1, self.current_difficulty + 1)

    def translate_text(self, text):
        if self.current_language == "en":
            return text
        try:
            return translator.translate(text, dest=self.current_language).text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def generate_questions(self, difficulty, text):
        difficulty_desc = {"easy": "simple facts", "medium": "processes", "hard": "complex applications"}
        prompt = f"""
        Generate 10 unique questions strictly based on the following text at {DIFFICULTY_LEVELS[difficulty]} difficulty - {difficulty_desc[DIFFICULTY_LEVELS[difficulty]]}:
        - 5 open-ended questions requiring text-based responses
        - 5 multiple-choice questions (MCQs) with 4 options (A, B, C, D)
        All questions must be directly relevant to the text content.
        For each:
        1. Include a concept label
        2. For open-ended: Provide a concise sample answer
        3. For MCQs: Provide 4 options and the correct answer
        Format exactly as:
        Q1. [Concept: concept_name] Open-ended question?
        Sample Answer: [sample_answer]
        Q6. [Concept: concept_name] MCQ question?
        A. [option_a]
        B. [option_b]
        C. [option_c]
        D. [option_d]
        Correct: [correct_letter]
        Text: {text}
        """
        try:
            response = self.model.generate_content(prompt)
            questions, sample_answers_or_options, concepts, is_mcq_flags = self.parse_quiz_response(response.text)
            return questions[:10], sample_answers_or_options[:10], concepts[:10], is_mcq_flags[:10]
        except Exception as e:
            print(f"Error generating questions: {e}")
            return ["Backup question"] * 10, ["Backup answer"] * 5 + [({"A": "A", "B": "B", "C": "C", "D": "D"}, "A")] * 5, ["general"] * 10, [False] * 5 + [True] * 5

    def parse_quiz_response(self, response_text):
        questions, sample_answers_or_options, concepts, is_mcq_flags = [], [], [], []
        question_blocks = []
        current_block = []
        for line in response_text.split('\n'):
            if line.strip() and (line.strip().startswith('Q') and any(char.isdigit() for char in line[:3])):
                if current_block:
                    question_blocks.append('\n'.join(current_block))
                    current_block = []
            if line.strip():
                current_block.append(line)
        if current_block:
            question_blocks.append('\n'.join(current_block))
        
        for block in question_blocks:
            if not block.strip():
                continue
            lines = block.strip().split('\n')
            q_line = lines[0]
            concept_start = q_line.find('[Concept:')
            concept_end = q_line.find(']', concept_start) if concept_start != -1 else -1
            concept = q_line[concept_start+9:concept_end].strip() if concept_start != -1 and concept_end != -1 else "general"
            question = q_line[concept_end+1:].strip() if concept_start != -1 and concept_end != -1 else q_line.split('.', 1)[1].strip() if '.' in q_line else q_line
            
            if "Sample Answer:" in block:
                sample_answer = next((line.split('Sample Answer:')[1].strip() for line in lines[1:] if 'Sample Answer:' in line), "No answer provided")
                questions.append(self.translate_text(question))
                sample_answers_or_options.append(self.translate_text(sample_answer))
                concepts.append(concept)
                is_mcq_flags.append(False)
            elif "Correct:" in block:
                options = {}
                correct_answer = None
                for line in lines[1:]:
                    if line.startswith(('A.', 'B.', 'C.', 'D.')):
                        opt_letter, opt_text = line[0], line[2:].strip()
                        options[opt_letter] = self.translate_text(opt_text)
                    elif 'Correct:' in line:
                        correct_answer = line.split('Correct:')[1].strip()
                questions.append(self.translate_text(question))
                sample_answers_or_options.append((options, correct_answer))
                concepts.append(concept)
                is_mcq_flags.append(True)
        
        while len(questions) < 10:
            questions.append(self.translate_text(f"Backup question {len(questions)+1}"))
            sample_answers_or_options.append(self.translate_text("Backup answer") if len(questions) <= 5 else ({"A": "A", "B": "B", "C": "C", "D": "D"}, "A"))
            concepts.append("general")
            is_mcq_flags.append(False if len(questions) <= 5 else True)
        return questions[:10], sample_answers_or_options[:10], concepts[:10], is_mcq_flags[:10]

    def evaluate_text_answer(self, question, user_answer, sample_answer, concept):
        # Remove language detection to avoid async issues
        # try:
        #     detected_lang = translator.detect(user_answer).lang
        #     if detected_lang != 'en':
        #         user_answer = translator.translate(user_answer, dest='en').text
        # except Exception as e:
        #     print(f"Translation error: {e}")

        prompt = f"""
        Evaluate this user answer for similarity to the sample answer, based on the concept '{concept}'.
        Score as:
        - 1 mark if fully similar (nearly identical meaning)
        - 0.5 marks if partially similar (some overlap in meaning)
        - 0 marks if not similar (no meaningful match)
        Provide brief feedback.

        Question: {question}
        User Answer: {user_answer}
        Sample Answer: {sample_answer}

        Format:
        Score: [number]
        Feedback: [text]
        """
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            score_line = next((line for line in response_text.split('\n') if line.startswith('Score:')), 'Score: 0')
            feedback_line = next((line for line in response_text.split('\n') if line.startswith('Feedback:')), 'Feedback: No feedback provided')
            score = float(score_line.split(':')[1].strip())
            feedback = self.translate_text(feedback_line.split(':')[1].strip())
            return score, feedback
        except Exception as e:
            print(f"Error evaluating text answer: {e}")
            return 0, self.translate_text("Evaluation failed")

    def evaluate_mcq_answer(self, user_answer, correct_answer):
        score = 1 if user_answer.upper() == correct_answer.upper() else 0
        return score

    def take_quiz(self, questions, sample_answers_or_options, concepts, is_mcq_flags):
        print(self.translate_text("\n--- Quiz ---"))
        print(self.translate_text("Answer in your language for text questions; A/B/C/D for MCQs:"))
        user_answers = []
        quiz_data = []
        
        for i, (question, answer_data, concept, is_mcq) in enumerate(zip(questions, sample_answers_or_options, concepts, is_mcq_flags)):
            print(f"\nQ{i+1}. [{concept}] {question}")
            if is_mcq:
                options, _ = answer_data
                for opt, text in options.items():
                    print(f"{opt}. {text}")
                user_answer = input(self.translate_text("Your answer (A/B/C/D): ")).strip().upper()
                while user_answer not in ['A', 'B', 'C', 'D']:
                    print(self.translate_text("Invalid input, enter A, B, C, or D:"))
                    user_answer = input(self.translate_text("Your answer (A/B/C/D): ")).strip().upper()
            else:
                user_answer = input(self.translate_text("Your answer: ")).strip()
            user_answers.append(user_answer)
            quiz_data.append((question, user_answer, answer_data, concept, is_mcq))
        
        total_score = 0
        concept_results = defaultdict(list)
        weak_areas = defaultdict(list)
        
        for question, user_answer, answer_data, concept, is_mcq in quiz_data:
            if is_mcq:
                _, correct_answer = answer_data
                score = self.evaluate_mcq_answer(user_answer, correct_answer)
                feedback = self.translate_text("Correct" if score == 1 else "Incorrect")
            else:
                sample_answer = answer_data
                score, feedback = self.evaluate_text_answer(question, user_answer, sample_answer, concept)
            concept_results[concept].append(score)
            total_score += score
            if score < (1 if is_mcq else 0.5):
                weak_areas[concept].append(feedback)
        
        for concept, results in concept_results.items():
            concept_mastery[concept].extend(results)
        
        feedback_para = self.translate_text("Review: ")
        if weak_areas:
            feedback_para += self.translate_text("You need improvement in ") + ", ".join([f"'{concept}'" for concept in weak_areas.keys()]) + "."
        else:
            feedback_para += self.translate_text("You performed well across all concepts!")
        
        print("\n" + feedback_para)
        print(self.translate_text(f"Score: {total_score}/10"))
        return total_score, list(weak_areas.keys())

    def run_quiz(self, text):
        if not text.strip():
            print(self.translate_text("Text cannot be empty."))
            return 0, []
        
        current_state_index = self.get_state_index(self.current_difficulty, self.performance_bucket, self.mastery_level)
        questions, sample_answers_or_options, concepts, is_mcq_flags = self.generate_questions(self.current_difficulty, text)
        
        if not questions:
            print(self.translate_text("Failed to generate questions."))
            return 0, []
        
        print(self.translate_text(f"\nDifficulty: {DIFFICULTY_LEVELS[self.current_difficulty]}"))
        user_score, weak_concepts = self.take_quiz(questions, sample_answers_or_options, concepts, is_mcq_flags)
        
        if user_score is None:
            print(self.translate_text("Error occurred during quiz. Score set to 0."))
            user_score = 0
        
        reward = self.calculate_reward(user_score, self.previous_score, self.current_difficulty)
        action = self.choose_action(current_state_index, user_score)
        next_difficulty = self.adjust_difficulty(action)
        
        next_state_index = self.get_state_index(next_difficulty, self.get_performance_bucket(user_score), self.get_mastery_level(concepts))
        self.update_q_table(current_state_index, action, reward, next_state_index)
        
        self.current_difficulty = next_difficulty
        self.previous_score = user_score
        self.performance_bucket = self.get_performance_bucket(user_score)
        self.mastery_level = self.get_mastery_level(concepts)
        
        return user_score, weak_concepts