# Filmception

## Artificial Intelligence Project



---

# Table of Contents

1. Data Preprocessing and Cleaning
   - a) Summary Extraction and Cleaning
   - b) Metadata Extraction
2. Text Translation and Audio Conversion
   - a) Text Translation
   - b) Audio Conversion
3. Movie Genre Prediction Model
   - a) Feature Extraction: SBERT Embeddings
   - b) Models Trained
   - c) Label Processing
   - d) Evaluation
   - e) Confusion Matrices
4. Interactive User Interface (Menu-Based System)

---

# 1. Data Preprocessing and Cleaning

## a) Summary Extraction and Cleaning

### Removing Special Characters, Stopwords, and Redundant Spaces

- Special characters (e.g., @, #, %, *, etc.) are removed using regular expressions (`re.sub`) as they do not carry semantic value in most NLP tasks.
- Stopwords, which are high-frequency words (like "the", "is", "and") that do not contribute to meaningful content, are removed using NLTK's stopword list to reduce noise and dimensionality.
- Redundant spaces (extra spaces, tabs, etc.) are normalized using regex or Python's string methods (`split() + join()`), ensuring uniform formatting.
- Cleans the text of irrelevant characters and formats it uniformly to prepare for tokenization and feature extraction.

### Lowercasing the Text to Standardize It

- All text is converted to lowercase using `.lower()` to avoid treating the same words in different cases as separate tokens (e.g., Apple ≠ apple).
- Standardization improves the accuracy of word frequency analysis and model generalization.

### Tokenization

- Tokenization is the process of splitting a sentence or document into individual units (tokens), usually words or phrases.
- Performed using NLTK's `word_tokenize()` or Python's `split()` method.
- Example:

```text
"The dog ran fast." → ['the', 'dog', 'ran', 'fast']
```

- Splits text into tokens by whitespace.
- Strips punctuation before splitting.

### Stemming/Lemmatization

- Stemming reduces words to their root by chopping off suffixes (e.g., running → run) but may not yield valid words.
- Instead of rough stemming (which may create non-real words), lemmatization is used for more accurate, dictionary-based root word conversion.
- Lemmatization is preferred for better accuracy, as it uses vocabulary and morphological analysis to return base forms (e.g., was → be, better → good).
- Tool: `WordNetLemmatizer` from NLTK.
- Requires POS tagging for best results.
- Reduces word variation to its base form, improving matching and reducing vocabulary size.

### Removing Non-Relevant Words like Numbers and Punctuations

- Numerical data and punctuation marks are removed using regular expressions and Python's `string.punctuation` list.
- In our code we are removing numbers, punctuations and stop words (like "is", "the", "and").
- These elements rarely contribute meaningful information to tasks like genre classification and add unnecessary complexity to the feature space.
- Removes noise that doesn't help in identifying themes or genres and simplifies the text for model training.

---

## b) Metadata Extraction

In this step, the code reads the `movie.metadata.tsv` file and selects only the necessary columns: the movie ID and the genres.

The genres are stored in a stringified dictionary format, where each movie can have multiple genre labels (multi-label format). These strings are safely converted into Python dictionaries, and the genre names are extracted as lists.

Example:

```python
["Action", "Adventure"]
```

Movies without any genre data are removed to ensure only valid entries remain.

---

# 2. Text Translation and Audio Conversion

## a) Text Translation

### Language Setup

- A dictionary maps human-readable language names (e.g., Arabic, Urdu, Korean) to their corresponding language codes (e.g., `ar`, `ur`, `ko`), needed for API requests.

### Translation Tool

- `GoogleTranslator` from the `deep_translator` library is used to translate summaries from English to the target languages.

### Chunking for Long Texts

- If a movie summary exceeds 4000 characters, it's split into smaller chunks (~3500 characters), ideally at sentence boundaries to maintain coherence.
- Each chunk is translated individually and then concatenated back into a full translation.

### Error Handling

- `try-except` blocks catch exceptions (e.g., rate limits, connection issues) during translation.
- Logs translation errors per language for easier debugging.

### Storage

- Translated summaries are saved as `.txt` files inside the `translations/` subdirectory using a consistent naming pattern.

---

## b) Audio Conversion

### Text-to-Speech Setup

- `gTTS` (Google Text-to-Speech) is used to convert the translated text into `.mp3` audio files.
- Language codes for TTS are reused from the translation setup.

### Chunking for Long Audio

- If the translated text exceeds ~10,000 characters, it's split into smaller chunks (~5000 characters) for TTS processing.
- Each chunk is converted to a temporary `.mp3` file.

### Audio Merging

- If multiple chunks are generated, `ffmpeg` (if available) is used to concatenate them into a single `.mp3` file.
- A manual fallback (binary file write) is used if `ffmpeg` is unavailable.

### Audio File Storage

- Final audio files are saved in the `audio/` subdirectory using a consistent naming convention.

Example:

```text
summaryID_language.mp3
```

---

# 3. Movie Genre Prediction Model

Build a multi-label classification model to predict movie genres based on their summaries using SBERT embeddings and machine learning classifiers.

## a) Feature Extraction: SBERT Embeddings

### Model Used

`paraphrase-mpnet-base-v2`

### Details

- SBERT (Sentence-BERT) is a modification of the BERT architecture that uses a siamese network structure to generate fixed-size sentence embeddings.
- The `paraphrase-mpnet-base-v2` model is one of the most powerful SBERT variants, trained to capture semantic similarity between sentences.
- It outputs a 768-dimensional dense vector for each input text.
- Unlike traditional BERT which outputs token-level embeddings, SBERT produces meaningful sentence-level embeddings that can be directly used for downstream tasks like classification or clustering.
- In this project, the movie summaries were passed through the SBERT model to obtain vector representations, which served as the input features for machine learning models.

---

## b) Models Trained

### 1. Logistic Regression (One-vs-Rest Strategy)

- Trained using the SBERT embeddings as input.
- Handled multi-label output by training a separate binary classifier for each genre.

### 2. Multi-Layer Perceptron (MLPClassifier)

- Feedforward neural network with one or more hidden layers.
- Capable of capturing non-linear patterns between movie summaries and genres.

---

## c) Label Processing

In this project, each movie can belong to multiple genres simultaneously, such as **Action**, **Comedy**, and **Sci-Fi**. This makes it a **multi-label classification problem** (as opposed to multi-class, where each item belongs to only one class).

### Step 1: Selecting Top Genres

- The full dataset may contain dozens of genres, including rare or less meaningful categories.
- To focus the model and reduce noise, only the top 15 most frequent genres were selected based on their overall frequency in the dataset.
- This helps avoid class imbalance caused by underrepresented labels and improves model generalization.

### Step 2: Binarization Using MultiLabelBinarizer

The genre labels for each movie are originally in the form of a list of strings, such as:

```python
["Drama", "Romance", "Thriller"]
```

Machine learning models require numerical input. Therefore, these string labels must be converted to a format the model can work with.

`MultiLabelBinarizer` from `sklearn.preprocessing` was used to convert each genre list into a binary vector, where:

- Each position in the vector corresponds to one of the selected 15 genres.
- A `1` indicates the movie belongs to that genre.
- A `0` means it does not.

Example:

```python
["Drama", "Thriller"] → [1, 0, 0, 1, 0, ..., 0]
```

### Output

- After binarization, the final label for each movie is a vector of length 15 (one for each genre).
- These binary vectors are used as target outputs for both the Logistic Regression and MLP models.

---

## d) Evaluation

### Logistic Regression (Best F1 Score)

| Metric | Value |
|----------|----------|
| Accuracy | 2.59% |
| F1 Score | 0.5509 |
| Hamming Loss | 0.2185 |

### MLP (Neural Network)

| Metric | Value |
|----------|----------|
| Accuracy | 18.17% |
| F1 Score | 0.4879 |
| Hamming Loss | 0.1152 |

### Key Insights

- Accuracy is misleading in multi-label classification (expect low values).
- Logistic Regression had the best F1 score (best balance of precision and recall).
- MLP had the lowest Hamming Loss (fewest incorrect labels).
- Threshold optimization found genre-specific cutoffs ranging from 0.4–0.8.

### Genre-Specific Performance

- Easiest to predict: **Drama** (F1 = 0.75 @ 0.4 threshold)
- Hardest to predict: **Indie** (F1 = 0.31 @ 0.6 threshold)
- High-threshold genres: **Family Film**, **Short Film** (need 80% confidence)

---

## e) Confusion Matrices

### Logistic Regression

> Insert Logistic Regression confusion matrix image here.

Example:

```markdown
![Logistic Regression Confusion Matrix](images/logistic_confusion_matrix.png)
```

### MLP Classifier

> Insert MLP confusion matrix image here.

Example:

```markdown
![MLP Confusion Matrix](images/mlp_confusion_matrix.png)
```

---

# 4. Interactive User Interface (Menu-Based System)

The project includes an interactive menu-based interface that integrates all functionalities into a single workflow.

Users can:

1. Select a movie summary.
2. Translate summaries into multiple languages.
3. Generate audio files from translated summaries.
4. Predict movie genres.
5. View model outputs and predictions.

The interface provides an easy-to-use experience for interacting with all modules of the Filmception system.

---

# Conclusion

Filmception demonstrates the application of Natural Language Processing and Machine Learning techniques for automated movie genre prediction.

By combining:

- Text preprocessing
- Metadata extraction
- Multilingual translation
- Text-to-speech conversion
- SBERT embeddings
- Multi-label classification

the project provides a complete pipeline for movie analysis and genre prediction.

The results show that SBERT embeddings paired with traditional machine learning models can effectively capture semantic information from movie summaries and predict multiple genres simultaneously.
