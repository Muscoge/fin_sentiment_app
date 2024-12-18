import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the Dataset
data_path = 'data/data.csv'  # Path to your dataset
df = pd.read_csv(data_path)

# Step 2: Preprocessing the Text Data
# Clean and preprocess text data (lowercasing, removing stopwords, etc.)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    # You can add more preprocessing steps such as removing special characters, etc.
    return text

df['Sentence'] = df['Sentence'].apply(preprocess_text)

# Step 3: Encode Labels (Sentiment)
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Step 4: Split Data into Train and Test Sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Sentence'].tolist(),
    df['Sentiment'].tolist(),
    test_size=0.2,
    random_state=42
)

# Step 5: Tokenization with BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512)
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=512)

# Step 6: Convert Data to Hugging Face Dataset Format
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# Step 7: Load Pre-trained BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Step 8: Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory for model predictions and checkpoints
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate after every epoch
)

# Step 9: Initialize MLflow
mlflow.start_run()  # Start an MLflow run to track this experiment

# Log training parameters in MLflow
mlflow.log_param('batch_size', training_args.per_device_train_batch_size)
mlflow.log_param('num_epochs', training_args.num_train_epochs)
mlflow.log_param('learning_rate', training_args.learning_rate)

# Step 10: Trainer for Fine-Tuning
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    tokenizer=tokenizer,                 # tokenizer used for processing text
)

# Step 11: Train the Model and Log Metrics
trainer.train()

# Log metrics in MLflow
train_results = trainer.evaluate()
mlflow.log_metric('train_loss', train_results['eval_loss'])
mlflow.log_metric('eval_accuracy', train_results['eval_accuracy'])

# Step 12: Save the Model and Tokenizer
model.save_pretrained('./fin_sentiment_model')
tokenizer.save_pretrained('./fin_sentiment_model')

# Log model in MLflow
mlflow.pytorch.log_model(model, "model")
mlflow.log_artifact('./fin_sentiment_model', "model")

# Step 13: Evaluate the Model (Optional)
eval_results = trainer.evaluate()
mlflow.log_metric('eval_loss', eval_results['eval_loss'])
mlflow.log_metric('eval_accuracy', eval_results['eval_accuracy'])

# Display evaluation results
print(f"Evaluation results: {eval_results}")

# End the MLflow run
mlflow.end_run()
