import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

def train_models():
    print("Loading data...")
    if not os.path.exists("data/reviews.csv"):
        print("Data not found. Please run generate_data.py first.")
        return
        
    df = pd.read_csv("data/reviews.csv")
    
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC(max_iter=1000)
    }
    
    best_acc = 0
    best_model_name = ""
    best_model = None
    
    print("\nTraining and evaluating models:")
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with {(best_acc*100):.2f}% Accuracy")
    
    # Evaluate best model
    y_pred = best_model.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save vectorizer and model
    print("Saving model and vectorizer...")
    joblib.dump(vectorizer, 'static/vectorizer.pkl')
    joblib.dump(best_model, 'static/model.pkl')
    
    # Visualizations
    print("Generating visualizations...")
    os.makedirs('static/images', exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'neutral', 'positive'], 
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title(f'Confusion Matrix ({best_model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('static/images/confusion_matrix.png')
    plt.close()

    # 2. ROC Curve
    # Binarize the output
    classes = ['negative', 'neutral', 'positive']
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    if hasattr(best_model, "decision_function"):
        y_score = best_model.decision_function(X_test_tfidf)
    else:
        y_score = best_model.predict_proba(X_test_tfidf)
        
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {classes[i]} (area = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('static/images/roc_curve.png')
    plt.close()
    
    # 3. Feature Importance (if applicable)
    if hasattr(best_model, 'coef_'):
        plt.figure(figsize=(10, 6))
        feature_names = vectorizer.get_feature_names_out()
        
        # Taking the first class (e.g., negative)
        coefs = best_model.coef_[0]
        top_positive_coefficients = np.argsort(coefs)[-10:]
        top_negative_coefficients = np.argsort(coefs)[:10]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        
        colors = ['red' if c < 0 else 'blue' for c in coefs[top_coefficients]]
        plt.bar(np.arange(20), coefs[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(20), feature_names[top_coefficients], rotation=45, ha='right')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('static/images/feature_importance.png')
        plt.close()
    else:
        print("Feature importance not supported for this model.")
        
    print("Pipeline complete!")

if __name__ == "__main__":
    train_models()
