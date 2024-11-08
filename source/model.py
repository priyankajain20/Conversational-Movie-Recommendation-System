import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

torch.manual_seed(23)
np.random.seed(23)

# Initialize BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lemmatizer = WordNetLemmatizer()
# Function to clean the text
def clean_text(text):
    plot_words = nltk.word_tokenize(text)
    punctutation_free_words = [word.lower() for word in plot_words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    stop_word_free_words = [word for word in punctutation_free_words if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stop_word_free_words]
    return ' '.join(lemmatized_words)

# Load data
df1 = pd.read_csv('./data/DataForModel.csv')
df1['Movie_Plot'] = df1['Movie_Plot'].apply(clean_text)
df = df1[['Movie_Plot', 'Movie_Revenue_Category']]

# Encode labels and tokenize plots
label_encoder = LabelEncoder()
df['Encoded_Labels'] = label_encoder.fit_transform(df['Movie_Revenue_Category'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define dataset class
class MovieDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        plot = str(self.data.iloc[index]['Movie_Plot'])
        label = self.data.iloc[index]['Encoded_Labels']
        inputs = self.tokenizer.encode_plus(
            plot,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create DataLoader for training and testing
train_dataset = MovieDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = MovieDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Define the neural network architecture
class RevenuePredictionModel(nn.Module):
    def __init__(self, bert_model, num_classes, dropout_prob=0.1):
        super(RevenuePredictionModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return nn.functional.softmax(logits, dim=1)

# Initialize the model and other training parameters
num_classes = len(label_encoder.classes_)
model = RevenuePredictionModel(bert_model, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 5

# Check if checkpoint exists
if os.path.exists('model.pt'):
    print("Loading model checkpoint...")
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
else:
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

    # Save checkpoint
    torch.save(model.state_dict(), 'model.pt')

true_labels=[]
predicted_labels=[]
# Evaluation loop
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
        logits = model(input_ids, attention_mask)
        _, predicted = torch.max(logits, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

predicted_df = pd.DataFrame({'Predicted_Labels': predicted_labels})
predicted_df['Predicted_Labels'] = label_encoder.inverse_transform(predicted_df['Predicted_Labels'])
# Generate test labels
test_df['Actual_Labels'] = label_encoder.inverse_transform(test_df['Encoded_Labels'])

# Save predicted and test dataframes as CSV
predicted_df.to_csv('../output/predicted.csv', index=False)
test_df['Actual_Labels'].to_csv('../output/test.csv', index=False)