import pandas as pd                         # For handling CSV files and dataframes
from sklearn.model_selection import train_test_split  # For splitting data into training and evaluation sets
# from datasets import Dataset                # For converting dataframes to Hugging Face datasets
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  # Hugging Face tools

# # Load the dataset from a CSV file
# df = pd.read_csv('patient_diagnosis_data.csv')  # Replace with your CSV file containing 'text' and 'label' columns
# print("Loading done...")
# # Split the dataset into training and evaluation sets (80% train, 20% eval)
# train_df, eval_df = train_test_split(df, test_size=0.2)

# # Convert the pandas dataframes to Hugging Face Dataset objects
# train_dataset = Dataset.from_pandas(train_df)
# eval_dataset = Dataset.from_pandas(eval_df)

# # Load the pre-trained BioBERT tokenizer
# tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# # Define a function to tokenize the text data
# def tokenize_function(examples):
#     # Tokenize the text and pad/truncate to a fixed length
#     return tokenizer(examples['text'], padding='max_length', truncation=True)

# # Apply the tokenize function to the datasets
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# # Set the format of the datasets to PyTorch tensors (input_ids, attention_mask, label)
# train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# # Load the pre-trained BioBERT model for sequence classification
# model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# # Set training arguments, including output directory, number of epochs, and batch sizes
# training_args = TrainingArguments(
#     output_dir='./results',                  # Directory to save model checkpoints
#     evaluation_strategy="epoch",             # Evaluate the model at the end of each epoch
#     learning_rate=2e-5,                      # Learning rate for the optimizer
#     per_device_train_batch_size=16,          # Batch size for training
#     per_device_eval_batch_size=16,           # Batch size for evaluation
#     num_train_epochs=3                       # Number of training epochs
# )

# # Initialize the Trainer with the model, training arguments, and datasets
# trainer = Trainer(
#     model=model,                             # The model to be trained
#     args=training_args,                      # Training arguments (defined above)
#     train_dataset=train_dataset,             # Training dataset
#     eval_dataset=eval_dataset                # Evaluation dataset
# )

# # Start the training process
# trainer.train()
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Step 1: Load your data
data = {
    'text': [
        "Patient diagnosed with Type 2 Diabetes. Prescribed Metformin.",
        "Blood pressure is elevated. Recommend increasing dosage of Lisinopril.",
        "MRI shows a herniated disc at L4-L5. Consider surgical consultation.",
        "Follow-up required for abnormal liver enzyme levels.",
        "Patient reports increased shortness of breath. Check for possible COPD exacerbation.",
        "Hemoglobin levels are low. Suggest iron supplements.",
        "Cholesterol levels are high. Initiate statin therapy.",
        "Signs of early-stage kidney disease. Advise on low-protein diet.",
        "Recommend regular blood sugar monitoring.",
        "Patient experiencing joint pain. Consider Rheumatology referral."
    ],
    'label': [
        "Diabetes Management", "Hypertension Management", "Spinal Surgery",
        "Liver Function", "COPD Management", "Anemia Management",
        "Hyperlipidemia", "Kidney Function", "Diabetes Monitoring", "Arthritis Management"
    ]
}

# Step 2: Define a label mapping
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

# Step 3: Apply the label mapping to the DataFrame
df = pd.DataFrame(data)
df['label'] = df['label'].map(label_mapping)

# Step 4: Convert pandas DataFrame to Dataset object
dataset = Dataset.from_pandas(df)

# Step 5: Load the BioBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Step 6: Tokenize the dataset with padding
def tokenize_function(example):
    return tokenizer(
        example['text'], 
        padding='max_length',        # Pad all sentences to the model's max length
        truncation=True,             # Truncate longer sequences
        max_length=128               # Set max length to avoid excessive padding
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 7: Split the dataset into training and evaluation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Step 8: Load the BioBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", num_labels=len(label_mapping)
)

# Step 9: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Step 10: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Step 11: Train the model
trainer.train()
