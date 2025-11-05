# app/services/ml_service.py

import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from sklearn.preprocessing import LabelEncoder
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

class MLService:
    def __init__(self, tokenizer_path="app/models/tokenizer.pkl", vocab_size=5000, max_len=100):
        """
        Initialize ML service with tokenizer and model configurations.
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.model = None
        self.encoder = LabelEncoder()

    def build_and_train(self, texts, labels, epochs=8, batch_size=32):
        """
        Build, tokenize, encode, and train the multi-class text classification model.
        """
        # --- 1️⃣ Tokenize text data ---
        print("🔤 Tokenizing text data...")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

        # Save tokenizer
        os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)
        with open(self.tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer saved at: {self.tokenizer_path}")

        # --- 2️⃣ Encode labels ---
        print("🏷️ Encoding labels...")
        y_int = self.encoder.fit_transform(labels)
        y = to_categorical(y_int)  # convert to one-hot for multi-class
        num_classes = y.shape[1]

        # --- 3️⃣ Build the model ---
        print("🏗️ Building multi-class LSTM model...")
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=64, input_length=self.max_len),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax")  # Multi-class output
        ])

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.summary()

        # --- 4️⃣ Train the model ---
        print("🚀 Training started...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        print("✅ Training completed successfully.")
        return history

    def predict(self, new_texts):
        """
        Predict the class label(s) for new text(s).
        """
        if not self.tokenizer:
            with open(self.tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)

        sequences = self.tokenizer.texts_to_sequences(new_texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

        preds = self.model.predict(X)
        class_indices = np.argmax(preds, axis=1)
        return self.encoder.inverse_transform(class_indices)