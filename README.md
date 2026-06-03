# 🎬 Filmception

## Overview

Filmception is an AI-based movie analysis system that preprocesses movie summaries, translates them into multiple languages, converts them into audio, and predicts movie genres using machine learning models with SBERT embeddings.

---

## Features

- Text preprocessing and cleaning  
- Movie metadata extraction  
- Multi-language translation of summaries  
- Text-to-speech (audio generation)  
- Multi-label genre prediction  
- Menu-driven user interface  

---

## Data Preprocessing

Movie summaries are cleaned using:

- Lowercasing  
- Tokenization  
- Lemmatization  
- Stopword removal  
- Removal of punctuation, numbers, and special characters  
- Whitespace normalization  

Metadata is extracted from `movie.metadata.tsv`, and genres are converted into a multi-label format.

---

## Translation & Audio

### Translation
- Uses `GoogleTranslator` from `deep_translator`
- Supports multiple languages
- Long texts are split into chunks for processing
- Outputs saved in `translations/`

### Audio
- Uses `gTTS` (Google Text-to-Speech)
- Converts translated text into `.mp3` files
- Handles long audio via chunking and merging
- Outputs saved in `audio/`

---

## Genre Prediction

- Uses SBERT model `paraphrase-mpnet-base-v2`
- Converts summaries into 768-dimensional embeddings
- Two models used:
  - Logistic Regression (One-vs-Rest)
  - MLP Classifier
- Multi-label classification with top 15 genres
- Labels converted using `MultiLabelBinarizer`

---

## Interface

The system includes a simple menu-based interface that allows users to:

- Select movie summaries  
- Translate text  
- Generate audio  
- Predict genres  
- View outputs interactively  

---

## Technologies Used

Python, NLTK, SBERT, Scikit-learn, Pandas, NumPy, deep-translator, gTTS, ffmpeg

---

## Conclusion

Filmception integrates NLP, machine learning, translation, and speech synthesis to analyze movie summaries and predict genres in a multilingual and interactive system.
