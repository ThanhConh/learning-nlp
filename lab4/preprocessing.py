import re
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================
# 1. TEXT CLEANING & PREPROCESSING
# ============================================

class TextPreprocessor:
    """
    Xử lý văn bản thô thành tokens sạch
    
    Pipeline:
    Raw text → Lowercase → Remove HTML tags → Remove special chars 
            → Tokenize → Remove stopwords → Lemmatization
    """
    
    def __init__(self):
        """Khởi tạo preprocessor"""
        self.stopwords = self._load_stopwords()
    
    @staticmethod
    def _load_stopwords():
        """Load danh sách stopwords tiếng Anh"""
        english_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for',
            'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or',
            'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they',
            'this', 'to', 'was', 'will', 'with', 'what', 'which', 'who',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'should', 'than', 'too'
        }
        return english_stopwords
    
    def clean_text(self, text):
        """
        Làm sạch văn bản
        
        Args:
            text (str): Văn bản thô
            
        Returns:
            str: Văn bản sạch
        """
        # Lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters nhưng giữ lại spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Chia văn bản thành tokens (từ)
        
        Args:
            text (str): Văn bản đã làm sạch
            
        Returns:
            list: Danh sách tokens
        """
        tokens = text.split()
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Loại bỏ stopwords
        
        Args:
            tokens (list): Danh sách tokens
            
        Returns:
            list: Danh sách tokens sau khi loại bỏ stopwords
        """
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        return filtered_tokens
    
    def preprocess(self, text, remove_stops=True):
        """
        Toàn bộ quy trình preprocessing
        
        Args:
            text (str): Văn bản thô
            remove_stops (bool): Có loại bỏ stopwords không
            
        Returns:
            list: Danh sách tokens đã xử lý
        """
        # Bước 1: Làm sạch
        cleaned = self.clean_text(text)
        
        # Bước 2: Tokenize
        tokens = self.tokenize(cleaned)
        
        # Bước 3: Loại bỏ stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Bước 4: Loại bỏ tokens ngắn (< 2 ký tự)
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens


# ============================================
# 2. TF-IDF VECTORIZATION
# ============================================

class TFIDFVectorizer:
    """
    Chuyển đổi văn bản thành TF-IDF vectors
    
    Formula:
        TF(term) = (Number of times term appears in doc) / (Total terms in doc)
        IDF(term) = log(Total docs / Docs containing term)
        TF-IDF = TF(term) × IDF(term)
    """
    
    def __init__(self, max_features=1000, min_df=2):
        """
        Khởi tạo vectorizer
        
        Args:
            max_features (int): Số features tối đa (vocabulary size)
            min_df (int): Số documents tối thiểu chứa một term
        """
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary = {}
        self.idf_values = {}
        self.fitted = False
    
    def fit(self, documents):
        """
        Fit model trên tập documents
        
        Args:
            documents (list): Danh sách documents (mỗi doc là list tokens)
        """
        # Bước 1: Build vocabulary
        term_doc_count = {}
        
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
        
        # Bước 2: Filter theo min_df
        filtered_terms = {
            term: count for term, count in term_doc_count.items()
            if count >= self.min_df
        }
        
        # Bước 3: Sort by frequency và lấy top max_features
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {
            term: idx for idx, (term, _) in enumerate(sorted_terms[:self.max_features])
        }
        
        # Bước 4: Tính IDF values
        num_docs = len(documents)
        self.idf_values = {}
        
        for term, idx in self.vocabulary.items():
            doc_count = term_doc_count.get(term, 0)
            self.idf_values[idx] = np.log(num_docs / (1 + doc_count))
        
        self.fitted = True
    
    def transform(self, documents):
        """
        Transform documents thành TF-IDF vectors
        
        Args:
            documents (list): Danh sách documents
            
        Returns:
            np.ndarray: Ma trận TF-IDF (n_docs × n_features)
        """
        if not self.fitted:
            raise ValueError("Vectorizer chưa được fit. Gọi fit() trước.")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            # Tính TF
            term_count = {}
            for term in doc:
                term_count[term] = term_count.get(term, 0) + 1
            
            # Tính TF-IDF
            doc_length = len(doc)
            for term, count in term_count.items():
                if term in self.vocabulary:
                    feature_idx = self.vocabulary[term]
                    tf = count / doc_length
                    idf = self.idf_values[feature_idx]
                    tfidf_matrix[doc_idx, feature_idx] = tf * idf
        
        # L2 normalization
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        tfidf_matrix = tfidf_matrix / (norms + 1e-10)
        
        return tfidf_matrix


# ============================================
# 3. LOGISTIC REGRESSION CLASSIFIER
# ============================================

class LogisticRegression:
    """
    Logistic Regression Classifier
    
    Formula:
        z = w^T x + b
        P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
        
    Loss function:
        L = - (y*log(ŷ) + (1-y)*log(1-ŷ))
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=100, lambda_reg=0.001):
        """
        Khởi tạo classifier
        
        Args:
            learning_rate (float): Learning rate cho gradient descent
            num_iterations (int): Số iterations cho training
            lambda_reg (float): Regularization parameter (L2)
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.trained = False
    
    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function
        σ(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        Train model using gradient descent
        
        Args:
            X (np.ndarray): Training features (n_samples × n_features)
            y (np.ndarray): Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Khởi tạo weights và bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.num_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Tính loss (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(predictions + 1e-10) + 
                          (1 - y) * np.log(1 - predictions + 1e-10))
            
            # Backward pass (Gradient computation)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) + \
                 (self.lambda_reg/n_samples) * self.weights
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iterations}, Loss: {loss:.4f}")
        
        self.trained = True
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất
        
        Args:
            X (np.ndarray): Features (n_samples × n_features)
            
        Returns:
            np.ndarray: Xác suất class 1 (n_samples,)
        """
        if not self.trained:
            raise ValueError("Model chưa được train.")
        
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Dự đoán class
        
        Args:
            X (np.ndarray): Features
            threshold (float): Ngưỡng quyết định
            
        Returns:
            np.ndarray: Predicted classes (0 hoặc 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)


# ============================================
# 4. EVALUATION METRICS
# ============================================

class Evaluator:
    """
    Đánh giá model classification
    
    Metrics:
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Tính confusion matrix
        
        Args:
            y_true (np.ndarray): Nhãn thực tế
            y_pred (np.ndarray): Nhãn dự đoán
            
        Returns:
            dict: TP, TN, FP, FN
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Accuracy = (TP + TN) / Total"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        """Precision = TP / (TP + FP)"""
        cm = Evaluator.confusion_matrix(y_true, y_pred)
        return cm['TP'] / (cm['TP'] + cm['FP'] + 1e-10)
    
    @staticmethod
    def recall(y_true, y_pred):
        """Recall = TP / (TP + FN)"""
        cm = Evaluator.confusion_matrix(y_true, y_pred)
        return cm['TP'] / (cm['TP'] + cm['FN'] + 1e-10)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        precision = Evaluator.precision(y_true, y_pred)
        recall = Evaluator.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall + 1e-10)
    
    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Đánh giá toàn diện
        
        Args:
            y_true (np.ndarray): Nhãn thực tế
            y_pred (np.ndarray): Nhãn dự đoán
            
        Returns:
            dict: Tất cả metrics
        """
        cm = Evaluator.confusion_matrix(y_true, y_pred)
        
        return {
            'TP': cm['TP'],
            'TN': cm['TN'],
            'FP': cm['FP'],
            'FN': cm['FN'],
            'Accuracy': Evaluator.accuracy(y_true, y_pred),
            'Precision': Evaluator.precision(y_true, y_pred),
            'Recall': Evaluator.recall(y_true, y_pred),
            'F1': Evaluator.f1_score(y_true, y_pred)
        }


# ============================================
# 5. PIPELINE ORCHESTRATION
# ============================================

class SentimentPipeline:
    """
    Full NLP pipeline cho Sentiment Classification
    
    Pipeline:
    Raw text → Preprocessing → TF-IDF → Logistic Regression → Prediction
    """
    
    def __init__(self, max_features=1000, min_df=2):
        """
        Khởi tạo pipeline
        
        Args:
            max_features (int): Vocabulary size
            min_df (int): Min document frequency
        """
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TFIDFVectorizer(max_features=max_features, min_df=min_df)
        self.classifier = LogisticRegression()
        self.evaluator = Evaluator()
    
    def preprocess_documents(self, documents):
        """
        Xử lý danh sách documents
        
        Args:
            documents (list): Danh sách text strings
            
        Returns:
            list: Danh sách tokenized documents
        """
        return [self.preprocessor.preprocess(doc) for doc in documents]
    
    def fit(self, X_text, y):
        """
        Train pipeline
        
        Args:
            X_text (list): Danh sách documents
            y (np.ndarray): Labels
        """
        # Preprocessing
        X_tokens = self.preprocess_documents(X_text)
        
        # TF-IDF vectorization
        self.vectorizer.fit(X_tokens)
        X_tfidf = self.vectorizer.transform(X_tokens)
        
        # Train classifier
        self.classifier.fit(X_tfidf, y)
    
    def predict(self, X_text):
        """
        Dự đoán nhãn
        
        Args:
            X_text (list): Danh sách documents
            
        Returns:
            np.ndarray: Predicted labels
        """
        # Preprocessing
        X_tokens = self.preprocess_documents(X_text)
        
        # TF-IDF vectorization
        X_tfidf = self.vectorizer.transform(X_tokens)
        
        # Predict
        return self.classifier.predict(X_tfidf)
    
    def predict_proba(self, X_text):
        """
        Dự đoán xác suất
        
        Args:
            X_text (list): Danh sách documents
            
        Returns:
            np.ndarray: Xác suất class 1
        """
        X_tokens = self.preprocess_documents(X_text)
        X_tfidf = self.vectorizer.transform(X_tokens)
        return self.classifier.predict_proba(X_tfidf)
    
    def evaluate(self, X_text, y_true):
        """
        Đánh giá model
        
        Args:
            X_text (list): Danh sách documents
            y_true (np.ndarray): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_text)
        return self.evaluator.evaluate(y_true, y_pred)


if __name__ == "__main__":
    print("="*70)
    print("NLP PIPELINE - SENTIMENT CLASSIFICATION DEMO")
    print("="*70)
    
    # Sample data
    X_train = [
        "this movie is amazing and wonderful",
        "terrible film absolutely bad",
        "great acting very good",
        "waste of time horrible movie",
        "loved it fantastic performance",
        "not good boring movie",
    ]
    
    y_train = np.array([1, 0, 1, 0, 1, 0])
    
    X_test = [
        "this is a great movie",
        "very bad movie",
        "excellent work",
    ]
    
    y_test = np.array([1, 0, 1])
    
    # Train pipeline
    print("\n[1] Training pipeline...")
    pipeline = SentimentPipeline(max_features=100, min_df=1)
    pipeline.fit(X_train, y_train)
    print("✓ Pipeline trained")
    
    # Evaluate on test set
    print("\n[2] Evaluating on test set...")
    metrics = pipeline.evaluate(X_test, y_test)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1:        {metrics['F1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['TP']}, TN: {metrics['TN']}")
    print(f"  FP: {metrics['FP']}, FN: {metrics['FN']}")
    
    # Predict with probability
    print("\n[3] Predicting with confidence scores...")
    for text in X_test:
        prob = pipeline.predict_proba([text])[0]
        sentiment = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        print(f"  Text: '{text}'")
        print(f"    → Sentiment: {sentiment} (confidence: {prob:.2%})\n")
    
    print("="*70)
