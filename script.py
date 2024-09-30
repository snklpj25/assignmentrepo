import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
with open('rt-polarity.txt', 'r', encoding='ISO-8859-1') as f_pos, open('rt-polarityneg.txt', 'r', encoding='ISO-8859-1') as f_neg:
    positive_sentences = f_pos.readlines()
    negative_sentences = f_neg.readlines()

# Ensure equal lengths of positive and negative sets
positive_sentences = positive_sentences[:5331]
negative_sentences = negative_sentences[:5331]

# Create DataFrames for positive and negative data
df_pos = pd.DataFrame(positive_sentences, columns=['text'])
df_pos['label'] = 1  # Positive label

df_neg = pd.DataFrame(negative_sentences, columns=['text'])
df_neg['label'] = 0  # Negative label

# Combine the data
df = pd.concat([df_pos, df_neg]).reset_index(drop=True)

# Shuffle the data to ensure randomness
df = df.sample(frac=1).reset_index(drop=True)

# Split the dataset into train, validation, and test sets
train_df = pd.concat([df[df['label'] == 1][:4000], df[df['label'] == 0][:4000]])
val_df = pd.concat([df[df['label'] == 1][4000:4500], df[df['label'] == 0][4000:4500]])
test_df = pd.concat([df[df['label'] == 1][4500:], df[df['label'] == 0][4500:]])

# Vectorize the text data using TfidfVectorizer (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.75, min_df=5)
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Labels
y_train = train_df['label']
y_val = val_df['label']
y_test = test_df['label']

# Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

# Predictions
y_train_pred = classifier.predict(X_train)
y_val_pred = classifier.predict(X_val)
y_test_pred = classifier.predict(X_test)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()

# Print the evaluation results
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix: Validation Set TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# Final evaluation on test set
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print(f"Confusion Matrix: Test Set TP={tp}, FP={fp}, TN={tn}, FN={fn}")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nTest Set Evaluation:")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")










