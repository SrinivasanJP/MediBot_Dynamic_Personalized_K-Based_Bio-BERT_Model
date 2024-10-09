from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model = BertForMaskedLM.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Function to generate chatbot response
def generate_response(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Note: Bert doesn't have EOS token by default
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # Decode the generated tokens to a string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Medical knowledge base (simple version for demo)
medical_knowledge_base = {
    "headache": "Headaches are often caused by stress, dehydration, or tension. It’s important to rest and drink plenty of fluids.",
    "fever": "Fever can be a sign of infection or inflammation. If the fever is high or persists for more than a few days, consult a doctor.",
    "tiredness": "Tiredness can be caused by lack of sleep, anemia, or low energy levels. Try improving your sleep habits and diet."
}

# Function to provide medical advice based on symptoms
def medical_advice(symptoms):
    """
    Provides basic medical advice based on the symptoms detected.

    Args:
        symptoms (list): List of symptoms detected in user input.

    Returns:
        str: Basic advice or response.
    """
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
        # User input
        user_input = input("\nYou: ")
        
        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Thank you for using the chatbot. Stay healthy!")
            break
        
        # Simple symptom detection from the input
        symptoms = [symptom for symptom in medical_knowledge_base if symptom in user_input.lower()]
        
        # If symptoms are detected, provide specific medical advice
        if symptoms:
            advice = medical_advice(symptoms)
            print(f"Chatbot: \n{advice}")
        else:
            # Generate a response using BioBERT model (this would require some adaptation for MLM task)
            response = generate_response(user_input)
            print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
