# fakenewsMPE
Fake news model performance evaluation.

### List of classifiers:
#### Name: BTC
* **Paper:** Behind the cues: A benchmarking study for fake news detection
* **Description:** : This classifier uses only news article text data. The authors extract 70 stylistic and psycholinguistic features such as number of unique words, readability, sentiment scores generated using LIWC (Note that LIWC is commercial. The original authors used enterprise LIWC services to generate the scores and we used our copy of LIWC2015; both of which are not included in the repository). Additionally, they also transformed each word into a vector using [**GLOVE**](https://nlp.stanford.edu/projects/glove/) and then sum the vectors to generate a vector representation of each article. 

* [**Code repository**](https://github.com/lbozarth/fakenewsMPE/tree/master/src_classifiers_BTC)

#### Name: NELA
* **Paper:** Assessing the news landscape: A multi-module toolkit for evaluating the credibility of news
* **Description:** Similar to BTC, this classifier also uses psycholinguistic, stylistic, and sentiment-based features. . The authors use Linear Support Vector Machine (SVM) and Random Forest as their classification algorithms.
* [**Code repository**](https://github.com/BenjaminDHorne/The-NELA-Toolkit)

#### Name: RDEL
* **Paper:**  A simple but tough-to-beat baseline for the Fake News Challenge stance detection task
* **Description:** : This model first tokenizes text from news articles and extracts the most frequent ngrams (unigram, bigram). Then, for each news article, it con-
structs the corresponding term frequency-inverse document frequency (TF-IDF) vectors for article title and body separately, and computes the cosine similarity between the 2 vectors. Finally, the authors concatenated the features together and use Multilayer Perceptron to classify fake and real news articles.

* [**Code repository**](https://github.com/lbozarth/fakenewsMPE/tree/master/src_classifiers_RDEL)

#### Name: HOAX
* **Paper:** Some like it hoax: Automated fake news detection in social networks
* **Description:** The authors construct a user-article bipartite graph based on whether a user liked or shared an article or a post. They then use semi-supervised harmonic label propagation to classify unlabeled articles. This approach is based on the hypothesis that users who frequently like or share fake or low-quality content can be used to identify the quality of unlabeled content.

* [**Code repository**](https://github.com/gabll/some-like-it-hoax/tree/master/dataset)

#### Name: CSI
* **Paper:** Csi: A hybrid deep model for fake news detection
* **Description:** For each news article, this paper  first partitions user engagements (e.g., tweets) with an article based on timestamps of the posts. All engagements within a partition are treated as a single document. They then use LSTM to capture the temporal patterns of the documents. Additionally, the authors also build a user-user network with the edge weight being the number of shared articles between pairs of users. This network’s corresponding adjacency matrix is then used to generate lower dimensional features that capture the similarity of users’ article sharing behavior. Finally, both sets of features are integrated together using another neural network layer.

* [**Code repository**](https://github.com/sungyongs/CSI-Code)

#### Name: TAA
* **Paper:** A topic-agnostic approach for identifying fake news pages
* **Description:** This model uses both new article text data as well as the HTML-layout features of the webpage.
* [**Code repository**](https://github.com/soniacq/FakeNewsClassifier)


### Datasets:
#### Name: Election-2016
* **Description:** This dataset is collected by [Bode el al](https://muse.jhu.edu/book/74490) and is not availble for public sharing.  The data collection was performed using Sysomos MAP. For any given day between December, 2015, and January 1, 2017, this dataset includes i.) 5,000 tweets randomly sampled from all tweets that included the keyword “Trump”, and ii) 5,000 tweets similarly sampled from all that mentioned “Clinton”.



#### NELA-GT
* **Description:** This dataset is provided by Norregaard et al. They scraped news articles content via news domains' RSS feed. Each domain
has source-level veracity labels from 1 or more independent assessments (e.g., Media Bias Fact Check, News Guard).  available in the following link: [**Dataset Link**](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ULHLCB).
