from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import pandas as pd

# Load dataset from CSV
df = pd.read_csv('patient_diagnosis_data.csv')

# Define label mapping
label_mapping = {
    "Diabetes Management": 0,
    "Hypertension Management": 1,
    "Spinal Surgery": 2,
    "Liver Function": 3,
    "COPD Management": 4,
    "Anemia Management": 5,
    "Hyperlipidemia": 6,
    "Kidney Function": 7,
    "Diabetes Monitoring": 8,
    "Arthritis Management": 9
}

# Reverse label mapping for printing statement
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Map labels in the DataFrame
df['label'] = df['label'].map(label_mapping)

# Convert to Dataset object
dataset = Dataset.from_pandas(df)

# Tokenizer and tokenization function
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def tokenize_function(example):
    return tokenizer(
        example['text'], 
        padding='max_length', truncation=True, max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", num_labels=len(label_mapping)
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Testing the model on new inputs
def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    
    # Map predicted label to the actual statement
    return reverse_label_mapping[predicted_class_id]

# Example prediction
test_text = "Patient diagnosed with Type 2 Diabetes. Prescribed Metformin."
prediction = predict(test_text)

print(f"Input: {test_text}")
print(f"Predicted label: {prediction}")
