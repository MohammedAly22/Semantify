# Semantify | An Intelligent Semantic Search
**Semantic search** in Natural Language Processing (NLP) is an advanced approach to **information retrieval** that goes **beyond** the **traditional method of matching keywords**. It involves a profound understanding of the **meanings behind words** and the **contextual nuances** in which they are used.

By leveraging techniques from NLP, semantic search aims to comprehend the **intricacies of human language**. This includes recognizing entities, such as people, places, and organizations, and understanding the relationships between them.

The ultimate goal is to provide **more precise** and **relevant** search results by considering **not just the words** in a query but also **the underlying semantics** and user intent, enhancing the overall search experience.

# Tools Used
The project is implemented using the following Python packages:

| Package | Description |
| --- | --- |
| re | Regular expression library |
| NLTK | Natural Language Toolkit |
| NumPy | Numerical computing library |
| Pandas | Data manipulation library |
| Matplotlib | Data visualization library |
| Sklearn | Machine learning library |
| TensorFlow | Open-source machine learning framework |
| Transformers | Hugging Face package contains state-of-the-art Natural Language Processing models |

# Dataset
The [AG-News-Classification-Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) **due to its substantial size** which is **large enough** to train a quite robust semantic search algorithm. It consists of the following fields: [`Title`, `Description`, and `Class Index`]. The `Class Index` column is an integer ranging from 1 to 4 with these corresponding classes:

| Index | Class |
| --- | --- |
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Science/Technology |

In total, there are **120,000** training samples and **7600** testing samples split into two files.

# Methodology
## Data Preparation
In this phase, I prepared the data before applying and data preprocessing technique, and this phase included:
- Normalizing column names to be lowercase.
- Creating a new `text` column by combining the `title` and `description` columns.
- Selecting relevant features [`text`, `category`].

## Data Preprocessing
After preparing the data, I applied standard text preprocessing techniques on the new `text` columns:
- Normalizing text to be lowercase
- Removing non-alphanumeric characters.
- Removing stopwords as they don't contribute to the semantics of the text.

## Models Selection
### Term-Frequency Inverse-Document-Frequency (TF-IDF):
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical measure used in natural language processing to **evaluate the importance of a word in a document** relative to a collection of documents (corpus). It consists of two components:
1. **Term Frequency (TF):**
   - Measures **how often** a term appears in a document.
   - Calculated as the **ratio** of **the number of occurrences** of a term to **the total number of terms** in the document.

  ```math
\text{TF}(t, d) = \frac{\text{Number of occurrences of term } t \text{ in document } d}{\text{Total number of terms in document } d}
```
   

3. **Inverse Document Frequency (IDF):**
   - Measures the **uniqueness****** of a term across the entire corpus.
   - Calculated as **the logarithm** of the ratio of **the total number of documents** in the corpus to** the number of documents containing the term**.

```math
\text{IDF}(t, D) = \log\left(\frac{\text{Total number of documents in the corpus } N}{\text{Number of documents containing term } t}\right)
```

The TF-IDF score for a term \( t \) in a document \( d \) within a corpus \( D \) is the product of TF and IDF:
```math
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
```

### Doc2Vec:
Doc2Vec, an abbr****eviation for **Document to Vector**, is a notable natural language processing (NLP) technique that extends the principles of Word2Vec to entire documents or sentences.

In contrast to Word2Vec, which represents words as vectors in a continuous vector space, Doc2Vec focuses **on encoding the semantic meaning of entire documents**. The primary implementation of Doc2Vec is known as the **Paragraph Vector model**, where each document in a corpus is associated with a unique vector.

This model employs two training approaches:
- PV-DM (Distributed Memory), akin to Word2Vec's Continuous Bag of Words (CBOW) model, considers **both context words** and the **paragraph vector** for word predictions.
- PV-DBOW (Distributed Bag of Words) relies **solely on the paragraph vector** for predicting target words. The resulting vector representations encapsulate the semantic content of documents, facilitating tasks like document similarity, clustering, and classification.

### Sentence Transformer (MiniLM l6 v2):
**Sentence Transformer** is a state-of-the-art natural language processing (NLP) model designed for **transforming** sentences or phrases into meaningful vector representations in a continuous vector space. Unlike traditional embeddings that capture word meanings, Sentence Transformer focuses on **encoding the semantic content of entire sentences**.

The model is based on **transformer architecture**, a powerful neural network architecture that has shown remarkable success in various NLP tasks. Sentence Transformer is trained on large corpora using unsupervised learning, where it learns to generate dense vectors for sentences. One of the key advantages of Sentence Transformer is its ability to produce **contextualized embeddings**, meaning the **representation of a sentence can vary based on the context** in which it appears.

# Results
For better comparison between several models, I conducted two test cases: one on a random query from the dataset, and the other on an external query wrote it myself to see how robust our models are.

Here is a random query from our dataset along with an external query that I will test the models on:
![0_random_external_queries](https://github.com/MohammedAly22/Semantify/assets/90681796/49890239-79d4-47fe-9585-58644222a5ab)

## TF-IDF Results
### Random Query Results:
Here are the most similar samples to our random query with their similarity scores:
![1_tf_idf_random_query](https://github.com/MohammedAly22/Semantify/assets/90681796/fc4dc204-53c1-46d2-893e-fb7e28b36b0d)


### External Query Results:
Here are the most similar samples to our external query with their similarity scores:
![2_tf_idf_external_query](https://github.com/MohammedAly22/Semantify/assets/90681796/7f332f30-4e63-4928-8a8d-c818e6b50432)



As observed, **despite its simplicity**, this technique **performs quite well** and delivers quick and effective results. With minimal effort, we can obtain the **top similar** results from our dataset for our queries.

Additionally, we notice that the category of these queries is **sports**, and our TF-IDF-based semantic search algorithm aims to **retrieve similar sports-related** results as much as possible.


## Doc2Vec Results
### Random Query Results:
Here are the most similar samples to our random query with their similarity scores:
![3_doc2vec_random_query](https://github.com/MohammedAly22/Semantify/assets/90681796/72b4ec9c-3c8b-46d3-b9f9-0c8481d97043)


### External Query Results:
Here are the most similar samples to our external query with their similarity scores:
![4_doc2vec_external_query](https://github.com/MohammedAly22/Semantify/assets/90681796/cff2620b-b06c-489e-9b13-ce137e6739de)



As observed, the outcomes are **somewhat subpar** when compared to the performance of the TF-IDF based semantic search algorithm. Once more, despite the query **falling under the sports** category, the model yielded results from **different categories** such as **world** and **business**.

## MiniLM l6 v2 Results
### Random Query Results:
Here are the most similar samples to our random query with their similarity scores:
![5_sentence_transformer_random_query](https://github.com/MohammedAly22/Semantify/assets/90681796/3074871b-b93c-4af2-bf41-314ac6a23144)


### External Query Results:
Here are the most similar samples to our external query with their similarity scores:
![6_sentence_transformer_external_query](https://github.com/MohammedAly22/Semantify/assets/90681796/b1487211-4e4d-476f-bb6f-49d06573bceb)



As evident from the results, **the attention mechanisms** play a **crucial role** in providing **contextualized embeddings** for each sample in the dataset. This feature enables us to obtain **the most accurate matching results for our queries**, which specifically discusses a **basketball match** between the **Spurs** and b. The model **successfully retrieves all documents** related to the Spurs and Mavericks, showcasing a commendable similarity score.

# Conclusion
In conclusion, based on the outcomes discussed above, it is evident that fundamental techniques like TF-IDF continue to perform remarkably well even without the use of neural networks. The results obtained with Doc2Vec demonstrate decent performance relying on fixed embeddings. However, the most effective technique appears to be the MiniLM transformer-based model, primarily owing to its utilization of attention mechanisms that can harness contextualized embeddings.
