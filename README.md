# Bitcoin Analysis with NLP

The project aims to explore the correlation between sentiment analysis of Reddit posts and fluctuations in Bitcoin prices. This involves analyzing the sentiment expressed in discussions on the Reddit platform to understand how it aligns with the dynamic changes in the value of Bitcoin.

The application of Natural Language Processing (NLP) is essential in extracting sentiment from textual data, such as comments and posts on Reddit. NLP methodologies enable the project to analyze and quantify the sentiment expressed in these texts, providing valuable insights into the collective mood of the online community. By leveraging NLP, the project seeks to uncover patterns and correlations between sentiment trends and Bitcoin price movements, enhancing our ability to make informed predictions in the volatile cryptocurrency market.

### Data Sourcing

The primary data source for this project is the Reddit platform, specifically the discussions and posts related to Bitcoin. The project focuses on extracting information from these user-generated texts to conduct sentiment analysis. Additionally, cryptocurrency market data, specifically Bitcoin price information, is sourced from CoinMarketCap.

To implement the web scraping methodology, the process involves utilizing the PRAW (Python Reddit API Wrapper) library to access Reddit's API. This allows for the extraction of relevant textual data from posts and comments in the specified subreddits. For cryptocurrency market data, web scraping techniques, possibly using BeautifulSoup, are applied to gather historical Bitcoin price information from CoinMarketCap.

### Data Cleaning and Preprocessing:
The data cleaning and preprocessing stage involves several crucial steps:

* Removal of duplicate and irrelevant information.
* Handling missing values in both Reddit posts and cryptocurrency market data.
* Textual data preprocessing, including the removal of special characters, URLs, and irrelevant terms.
* Tokenization of textual data, converting it into individual words.
* Removal of stopwords and converting words to lowercase.
* Ensuring consistency in date formats and aligning the temporal aspects of Reddit posts and market data.
* Normalization and Feature Extraction:
* Normalization is applied to the cryptocurrency market data, particularly the Bitcoin price, to ensure that the values are on a comparable scale. Feature extraction involves deriving relevant information from the textual data, such as sentiment scores and processed text, to create meaningful features for analysis.

#### Final Data Set Description: <br>
The final dataset is a structured and cleaned dataset that combines information from Reddit posts and cryptocurrency market data. It includes features like post titles, URLs, sentiment scores, and processed text from Reddit, along with Bitcoin price and related information. This consolidated dataset is prepared for subsequent analysis, allowing for the exploration of correlations between sentiment trends on Reddit and Bitcoin price fluctuations. The temporal alignment facilitates time-series analysis to uncover patterns and trends over time.


### Machine Learning Modeling

The chosen machine learning model for this project is a recurrent neural network (RNN), specifically using the Long Short-Term Memory (LSTM) architecture. The rationale behind selecting an LSTM-based RNN is its ability to capture sequential dependencies in time-series data, making it suitable for analyzing the temporal nature of Reddit sentiment and Bitcoin price fluctuations. LSTMs excel in handling long-term dependencies, which is crucial when dealing with sequential data like textual information and financial time series.

### Findings Summary
The key findings of the project revolve around the correlation between sentiment analysis of Reddit posts and short-term Bitcoin price fluctuations. The project aimed to uncover whether sentiment trends on social media platforms, specifically Reddit, could serve as a valuable indicator for predicting immediate changes in Bitcoin prices.

Insights gained through NLP and machine learning techniques include:

* **Temporal Correlation**: The project revealed temporal correlations between sentiment trends and subsequent price movements. Analyzing the sentiment of Reddit posts at specific time points provided indications of potential short-term shifts in Bitcoin prices.
* **Sequential Dependency**: The use of LSTM-based RNNs showcased their effectiveness in capturing sequential dependencies in both textual data and financial time series, enabling a more nuanced analysis of sentiment and price trends.

### Limitations and Challenges

* **Data Noisiness**: Social media data, including Reddit posts, can be noisy and subjective. The presence of sarcasm, irony, or ambiguous language poses challenges for sentiment analysis accuracy.
* **Market Dynamics**: Cryptocurrency markets are influenced by various factors, including external news, regulations, and macroeconomic trends. Isolating the impact of social media sentiment from other market drivers is inherently challenging.
* **Model Complexity**: Balancing the complexity of the machine learning model to prevent overfitting or underfitting posed a challenge. Striking the right balance was crucial for the model's ability to generalize to unseen data.


### Future Challenges

* **Fine-Tuning NLP Models**: Further refinement of sentiment analysis models to better handle nuances in language and context.
* **Incorporating External Factors**: Integrating additional external factors, such as news sentiment or macroeconomic indicators, to improve the model's predictive capabilities.
* **Ensemble Models**: Exploring ensemble models that combine predictions from multiple sources, including social media sentiment and traditional financial indicators.





