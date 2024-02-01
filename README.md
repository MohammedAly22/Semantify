# SemanticSearch
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

Here is a random query from our dataset:
| Random Query | Category |
| --- | --- |
| `spurs defeat mavericks 94 80 tim duncan scored 27 points san antonio spurs held dallas 3 17 shooting fourth quarter 94 80 victory mavericks wednesday night` | Sports |

And, here is our external query that we will test the models on:
| External Query | Category |
| --- | --- |
| `Real Madrid beat Barcelona 4-1 in the Spanish Super Cup in Saudi Arabia, as El Clasico delivered drama and brilliant goals once again` | Sports |

## TF-IDF Results
### Random Query Results:
Here are the most similar samples to our random query with their similarity scores:
| Random Query | Category |
| --- | --- |
| `spurs defeat mavericks 94 80 tim duncan scored 27 points san antonio spurs held dallas 3 17 shooting fourth quarter 94 80 victory mavericks wednesday night` | Sports |

| Matched Query | Category | Similarity Score |
| --- | --- | --- |
| `spurs defeat mavericks 94 80 tim duncan scored 27 points san antonio spurs held dallas 3 17 shooting fourth quarter 94 80 victory mavericks wednesday night` | Sports | 1.0000 |
| `spurs beat magic 94 91 ap ap tim duncan 24 points 14 rebounds lead san antonio spurs 94 91 victory wednesday night orlando magic` | Sports | 0.4976 |
| ` nba game summary san antonio dallas dallas tx sports network tim duncan 20 points 13 rebounds five blocks devin brown scored 14 16 points fourth quarter leading san antonio spurs 107 89 victory dallas mavericks american airlines center` | Sports | 0.4658 |
| ` streaking spurs roll past sixers 88 80 ap ap tim duncan scored season high 34 points grabbed 13 rebounds lead san antonio spurs fifth straight victory 88 80 philadelphia 76ers thursday night` | Sports |0.4473 |
| ` spurs past sixers 88 80 san antonio spurs snatched fifth straight victory away game philadelphia 76ers tim duncan led spurs season high 34 points 13 rebounds china radio international reported friday` | Sports | 0.4399 |

### External Query Results:
Here are the most similar samples to our external query with their similarity scores:
| Externa; Query | Category |
| --- | --- |
| `Real Madrid beat Barcelona 4-1 in the Spanish Super Cup in Saudi Arabia, as El Clasico delivered drama and brilliant goals once again.` | Sports |

| Matched Query | Category | Similarity Score |
| --- | --- | --- |
| `barcelona beat real madrid barcelona moved seven points clear top spanish league saturday following three nil victory home second placed real madrid` | Sports | 0.3614   |
| `barcelona shuts rival real madrid madrid spain barcelona moved ahead spanish league beating rival real madrid 3 0 saturday country 39 biggest match` | Sports | 0.3411 |
| `barcelona real madrid post home wins barcelona spain sports network david beckham scored game winner real madrid 39 galacticos 39 barcelona week two spanish premier division` | Sports | 0.3307 |
| `barcelona beats real madrid spanish league barcelona moved ahead spanish league beating rival real madrid 3 0 saturday country 39 biggest match samuel eto 39 giovanni van bronckhorst scored first half ronaldinho` | Sports | 0.2882 |
| `real madrid stays touch leader barcelona spanish lt b gt lt b gt four goals 11 minutes allowed real madrid destroy bernd schuster 39 levante win frustrated fans secured second place spanish league standings` | Sports | 0.2796 |


As observed, **despite its simplicity**, this technique **performs quite well** and delivers quick and effective results. With minimal effort, we can obtain the **top similar** results from our dataset for our queries.

Additionally, we notice that the category of these queries is **sports**, and our TF-IDF-based semantic search algorithm aims to **retrieve similar sports-related** results as much as possible.


## Doc2Vec Results
### Random Query Results:
Here are the most similar samples to our random query with their similarity scores:
| Random Query | Category |
| --- | --- |
| `spurs defeat mavericks 94 80 tim duncan scored 27 points san antonio spurs held dallas 3 17 shooting fourth quarter 94 80 victory mavericks wednesday night` | Sports |

| Matched Query | Category | Similarity Score |
| --- | --- | --- |
| `spurs defeat mavericks 94 80 tim duncan scored 27 points san antonio spurs held dallas 3 17 shooting fourth quarter 94 80 victory mavericks wednesday night` | Sports | 0.9542 |
| `3 bombings resort towns sinai three explosions shook three egyptian sinai resorts popular vacationing israelis killing least 30 people wounding 100` | World | 0.7559 |
| `sec sues 3 former kmart execs washington federal regulators filed civil fraud charges three former kmart executives five current former managers suppliers` | Business | 0.7360 |
| `siebel moves toward self repairing software com october 11 2004 3 34 pm pt fourth priority 39 main focus enterprise directories organizations spawn projects around identity infrastructure` | Science/Technology | 0.7080 |
| `cerberus buy lnr property 3 8 bn new york august 30 new ratings lnr property corporation lnr nys reportedly agreed acquired riley property holdings llc 3` | Business | 0.7076 |

### External Query Results:
Here are the most similar samples to our external query with their similarity scores:
| Externa; Query | Category |
| --- | --- |
| `Real Madrid beat Barcelona 4-1 in the Spanish Super Cup in Saudi Arabia, as El Clasico delivered drama and brilliant goals once again.` | Sports |

| Matched Query | Category | Similarity Score |
| --- | --- | --- |
| `uefa cup champ takes super cup 2 1 victory porto uefa cup holders valencia beat european champion porto 2 1 win super cup monaco 39 stade louis ii friday midfielder vicente laid valencia goals ruben baraja heading
` | Sports | 0.4815 |
| `fa investigate chelsea west ham violence league cup match football association investigate crowd violence marred chelsea 39 1 0 win west ham league cup mateja kezman scored goal wednesday night stamford` | Sports | 0.3916 |
| `marseille 39 european cup winning 39 sorcerer 39 dies belgian 1993 european cup french side marseille time side france captured european club football 39 premier trophy 1978 cup winners cup belgian giants anderlecht` | Sports | 0.3915 |
| `update 1 inter make hard work cup win bologna inter milan came behind beat bologna 3 1 italian cup third round first leg match san siro stadium sunday` | Sports | 0.3821 |
| `golf woods reveals cup ambition tiger woods wants playing vice captain next us ryder cup team` | World | 0.3502 |


As observed, the outcomes are **somewhat subpar** when compared to the performance of the TF-IDF based semantic search algorithm. Once more, despite the query **falling under the sports** category, the model yielded results from **different categories** such as **world** and **business**.

## MiniLM l6 v2 Results
### Random Query Results:
Here are the most similar samples to our random query with their similarity scores:
| Random Query | Category |
| --- | --- |
| `spurs defeat mavericks 94 80 tim duncan scored 27 points san antonio spurs held dallas 3 17 shooting fourth quarter 94 80 victory mavericks wednesday night` | Sports |

| Matched Query | Category | Similarity Score |
| --- | --- | --- |
| `spurs defeat mavericks 94 80 tim duncan scored 27 points san antonio spurs held dallas 3 17 shooting fourth quarter 94 80 victory mavericks wednesday night` | Sports | 1.0000 |
| `spurs run mavericks 107 89 ap ap devin brown sparked fourth quarter spurt two three point plays two dunks helping san antonio spurs beat dallas mavericks 107 89 monday night spoil pseudo coaching debut avery johnson` | Sports | 0.9195 |
| `spurs 107 mavericks 89 devin brown sparked fourth quarter spurt two three point plays two dunks helping san antonio spurs beat dallas mavericks 107 89 monday night spoil pseudo coaching debut avery johnson` | Sports | 0.9107 |
| `duncan leads spurs past hornets 83 69 ap ap tim duncan 19 points 12 rebounds lead san antonio spurs third straight victory 83 69 new orleans hornets friday night` | Sports | 0.9098 |
| `nba game summary dallas san antonio mavericks 4 3 road season spurs 18 straight regular season home games dating back last year dallas season low eight assists san antonio tx sports network tim` | Sports | 0.9064 |

### External Query Results:
Here are the most similar samples to our external query with their similarity scores:
| Externa; Query | Category |
| --- | --- |
| `Real Madrid beat Barcelona 4-1 in the Spanish Super Cup in Saudi Arabia, as El Clasico delivered drama and brilliant goals once again.` | Sports |

| Matched Query | Category | Similarity Score |
| --- | --- | --- |
| `spain real madrid crush levante ronaldo scored twice real madrid ended two game winless slide 5 0 spanish league victory seventh placed levante santiago bernabeu sunday` | Sports | 0.8590   |
| `barcelona beats real madrid spanish league barcelona moved ahead spanish league beating rival real madrid 3 0 saturday country 39 biggest match samuel eto 39 giovanni van bronckhorst scored first half ronaldinho` | Sports | 0.8589 |
| `liga sunday wrap madrid answer critics real madrid ended talk crisis club thumped levante 5 0 bernabeu valencia moved back champions league places 2 0 win mallorca` | Sports | 0.8574 |
| `barcelona 3 0 real madrid cameroon 39 samuel eto 39 fils helped barcelona trounce real madrid 3 0 move seven points clear great rivals spain 39 la liga` | Sports | 0.8547 |
| `real madrid ponders biggest champions league loss four years real madrid began yesterday 39 match bayer leverkusen bookmakers 39 favorite win champions league record nine time european champion finished worst defeat competition four years` | Sports | 0.8535 |


As evident from the results, **the attention mechanisms** play a **crucial role** in providing **contextualized embeddings** for each sample in the dataset. This feature enables us to obtain **the most accurate matching results for our queries**, which specifically discusses a **basketball match** between the **Spurs** and b. The model **successfully retrieves all documents** related to the Spurs and Mavericks, showcasing a commendable similarity score.

# Conclusion
In conclusion, based on the outcomes discussed above, it is evident that fundamental techniques like TF-IDF continue to perform remarkably well even without the use of neural networks. The results obtained with Doc2Vec demonstrate decent performance relying on fixed embeddings. However, the most effective technique appears to be the MiniLM transformer-based model, primarily owing to its utilization of attention mechanisms that can harness contextualized embeddings.
