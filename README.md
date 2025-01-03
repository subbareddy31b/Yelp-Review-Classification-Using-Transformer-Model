# Yelp Review Classification Using Transformer Model 🚀📊

## Project Overview 📝

In this project, we classify Yelp reviews as either positive or negative using a transformer-based deep learning model. The dataset contains over 500,000 rows with Yelp reviews and their corresponding classes (1 for positive 👍 and 0 for negative 👎). We preprocess the text data, create word embeddings using Word2Vec, and train a transformer model to classify the reviews.

## Dataset Description 📋

The dataset contains:
- **Basic Data**: `class` (0 for negative 👎, 1 for positive 👍), `review_text` (the review content).
- **Rows**: Over 500,000 rows with reviews and corresponding labels.

We filter the data to include only the first 50,000 reviews of each class (positive 👍 and negative 👎) to ensure the system runs efficiently and prevents kernel breakage issues. 🖥️

## Preprocessing Steps 🧹

1. **Data Loading**: The dataset is loaded as a DataFrame 📊, and columns are assigned accordingly.
2. **Data Cleaning**: Textual cleaning includes converting text to lowercase 🔤, removing punctuation ❌, and eliminating stopwords 🚫.
3. **Tokenization**: The reviews are tokenized using the `Tokenizer` from Keras 🧑‍💻, and the vocabulary size is determined 🔢.
4. **Word Embedding**: We use the Word2Vec model to generate word embeddings 🌐 from the tokenized text. An embedding matrix is created and converted to PyTorch tensors 🔲 for use in the model.

## Transformer Model Architecture 🔥

The transformer model consists of the following components:

1. **Embedding Layer**: Converts input tokens into vectors of the specified embedding dimension 🔠.
2. **Positional Encoding**: Adds positional encodings 📍 to the embedded vectors to capture the order of tokens.
3. **Encoder**: The encoder layers in the transformer capture contextual information from the input sequence 🧠.
4. **Decoder**: The decoder generates the output sequence based on the encoder’s memory 💡.
5. **Fully Connected Layer**: The output of the decoder is passed through a fully connected layer 🔌 to predict the class (positive 👍 or negative 👎 review).

### Positional Encoding 🌍
We use sinusoidal positional encodings to preserve the order of tokens in the sequence 🔄, which is crucial for transformer-based models.

### Model Variants 🧩
1. **Base Model**: The first model uses the basic transformer architecture with positional encoding, transformer encoder and decoder layers, and a fully connected output layer.
   
2. **Enhanced Model**: The second model builds upon the base model by adding:
   - **Dropout Layer** 🚫: To prevent overfitting.
   - **L2 Regularization** 🔄: To reduce model complexity and prevent overfitting.
   - **Learning Rate Tuning** ⚡: To improve model performance.

### Observation 🔍
After applying the techniques (dropout layer, L2 regularization, and learning rate tuning), the accuracy of the base model, which was 80%, was reduced to **74%** 📉. This suggests that further optimization is required to maintain or improve the model's performance with these enhancements.

## Model Training 🏋️‍♂️
The model is trained using the **Adam optimizer** and the **Cross-Entropy loss function** for binary classification (positive 👍 vs. negative 👎 review).

## Model Performance Metrics 📊
**Base Model**: 
| Metric          | Positive Class (1) | Negative Class (0) | Average |
|-----------------|--------------------|--------------------|---------|
| **Precision**   | 0.81               | 0.79               | 0.80    |
| **Recall**      | 0.78               | 0.81               | 0.80    |
| **F1-Score**    | 0.79               | 0.80               | 0.80    |
| **Support**     | 9964               | 10036              | 20000   |

- **Accuracy**: 0.80

![output](https://github.com/user-attachments/assets/92a5fcf0-bb1c-46d7-91c5-88f094c94349)


**Enhanced Model**: 
| Metric          | Positive Class (1) | Negative Class (0) | Average |
|-----------------|--------------------|--------------------|---------|
| **Precision**   | 0.70               | 0.80               | 0.75    |
| **Recall**      | 0.84               | 0.65               | 0.74    |
| **F1-Score**    | 0.76               | 0.72               | 0.74    |
| **Support**     | 9964               | 10036              | 20000   |

- **Accuracy**: 0.74

![output](https://github.com/user-attachments/assets/5c8918e9-c0ab-4113-873c-6cfa35a6031f)


## Word Cloud Visualization 🌐
We visualize the most frequent words in positive 👍 and negative 👎 reviews using word clouds 💭. The word cloud for positive reviews is displayed on a white background ⚪, and the word cloud for negative reviews is displayed on a black background ⚫.
![output](https://github.com/user-attachments/assets/6c24ea42-b48d-480c-80a1-4b117dcbb2bf)


## Conclusion 🎯

This project demonstrates the use of a transformer model to classify Yelp reviews as positive 👍 or negative 👎. We explore two models: a base model and an enhanced model that incorporates dropout, L2 regularization, and learning rate tuning. Despite the improvements, the enhanced model showed a reduction in accuracy from 80% to 74%, suggesting room for further refinement.
