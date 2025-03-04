import json
import torch
import warnings
import speech_recognition as sr
import pyttsx3
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings('ignore')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load pre-trained GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the fine-tuned BioBART model and tokenizer
extraction_dir = '../../bertV2'
model = AutoModelForSeq2SeqLM.from_pretrained(extraction_dir)
tokenizer = AutoTokenizer.from_pretrained(extraction_dir)
print("LOG: Model Loaded...")

# Load the medical knowledge base from JSON
def load_medical_knowledge_base(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

medical_knowledge_base = load_medical_knowledge_base('knowledgeBase.json')

# Function to generate chatbot response using GPT-2
def generate_response(prompt, max_length=150):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=gpt2_tokenizer.eos_token_id, 
        temperature=0.7, 
        top_p=0.9, 
        do_sample=True, 
    )
    return gpt2_tokenizer.decode(outputs[0][0], skip_special_tokens=True)

# Function to classify symptoms using BioBART
def classify_symptoms(inputs):
    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=150, 
            do_sample=True,   
            temperature=0.9, 
            top_p=0.95,
            num_return_sequences=1  
        )
        
        return [tokenizer.decode(output, skip_special_tokens=True).strip().lower() for output in outputs]

# Function to provide medical advice
def medical_advice(symptoms):
    advice = ""
    for symptom in symptoms:
        if symptom in medical_knowledge_base:
            advice += f"Symptom: {symptom}\nAdvice: {medical_knowledge_base[symptom]}\n\n"
        else:
            advice += f"Symptom: {symptom}\nAdvice: No specific advice found. Please consult a doctor.\n\n"
    return advice

# Function to take voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"\nYou (Voice Input): {text}")
            return text
        except sr.UnknownValueError:
            print("\nSorry, I couldn't understand that. Please try again.")
            return None
        except sr.RequestError:
            print("\nError with the speech recognition service. Try again later.")
            return None

# Function to speak output
def speak(text):
    print(f"Chatbot: {text}")
    engine.say(text)
    engine.runAndWait()

# Main chatbot loop
def chatbot():
    speak("Welcome to the Biomedical Chatbot! How can I assist you today?")
    speak("Note: This chatbot provides general medical advice. Please consult a healthcare professional for accurate diagnosis.")

    while True:
        print("\nSay something (or type if you prefer). Say 'exit' to quit.")
        user_input = get_voice_input()

        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            speak("Thank you for using the chatbot. Stay healthy!")
            break
        
        symptoms = classify_symptoms(user_input)
        speak("Detected Symptoms: " + ", ".join(symptoms))

        if symptoms and any(symptom in medical_knowledge_base for symptom in symptoms):
            advice = medical_advice(symptoms)
            speak(advice)
        # else:
        #     response = generate_response(user_input)
        #     speak(response)

# Run the chatbot
if __name__ == "__main__":
    chatbot()
