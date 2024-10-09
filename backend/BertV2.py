import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
# Load pre-trained GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the fine-tuned BioBART model and tokenizer
extraction_dir = './bertV2'
model = AutoModelForSeq2SeqLM.from_pretrained(extraction_dir)
tokenizer = AutoTokenizer.from_pretrained(extraction_dir)
print("LOG: Model Loaded...")

# Load the medical knowledge base from JSON
def load_medical_knowledge_base(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load medical knowledge base
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
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to classify symptoms using BioBERT
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
        
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return [text.strip().lower() for text in generated_texts]  

# Function to provide medical advice based on symptoms
def medical_advice(symptoms):
    advice = ""
    for symptom in symptoms:
        if symptom in medical_knowledge_base:
            advice += f"Symptom: {symptom}\nAdvice: {medical_knowledge_base[symptom]}\n\n"
        else:
            advice += f"Symptom: {symptom}\nAdvice: No specific advice found. Please consult a doctor.\n\n"
    return advice

# Main chatbot loop
def chatbot():
    print("Welcome to the Biomedical Chatbot! How can I assist you today?")
    print("Note: This chatbot provides general medical advice. Please consult a healthcare professional for accurate diagnosis.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Thank you for using the chatbot. Stay healthy!")
            break
        
        symptoms = classify_symptoms(user_input)
        print("Symptom: " + " ".join(symptoms))
        
        if symptoms and any(symptom in medical_knowledge_base for symptom in symptoms):
            advice = medical_advice(symptoms)
            print(f"Chatbot: \n{advice}")
        else:
            response = generate_response(user_input)
            print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
