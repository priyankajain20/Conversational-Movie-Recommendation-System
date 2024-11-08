# CMPT 713 - Final Project - Spring 2024

**Team Name:  NeuralNarratives**

**Team Members:**

| Name | Student ID | Computing ID |
| :--- | :--- | :--- |
| Priyanka Sohanlal Jain | 301553408 | psj3@sfu.ca | 
| Pranjal Verma | 301562277 | pva14@sfu.ca |


## Problem Definition

Based on the input given by the user, our system will try to give best recommendations to the user. The problem that we are trying to address here is the cold start problem that a movie recommendation system faces while recommending movies. The cold start issue might arise in the context of movie recommendations when a new film is released, and we have insufficient data about it. As there is little information about the characteristics of the movie in these circumstances, it can be challenging for the recommendation engine to offer precise recommendations, without feedback and user reviews making the other features such as movie plot,genre,language as important features to look into to decide if it is popular or not. We will try to address this issue by using other features to predict the revenue class and thereby recommending a popular movie based on the revenue of the movies. 

## Data Source

[CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/)

**Links**

- [Data](https://vault.sfu.ca/index.php/s/x6bNCHyfzsQINTs)

- [Model File](https://vault.sfu.ca/index.php/s/eVZGBx8qWlxqrkY)

- [Video](https://vault.sfu.ca/index.php/s/jdKohfmkUU3bs1E)

- [Chatbot Agent](https://vault.sfu.ca/index.php/s/PYi7AsK7xvnWvW6)

## File definition

- `source/baseline.py` -> Source code for baseline solution

- `source/eda.py` -> Source code for Exploratory Data Analysis 

- `source/project.ipynb` -> Jupyter notebook for overall run of the project and analysis

- `source/utils.py` -> Different helper functions used in the project

- `source/featureEngineering.py` -> Source code for the 2nd approach. This second approach makes use of the movie plot attribute of each movie and applies Bag Of Words, then Tf-idf on the result aquired from Bag of Words, and to reduce the dimensionality of the dataframe we apply Singular Vector Decomposition. And finally test this data on Guassian Naive Baiyes.

- `source/clusterFeatureEngineering.py` -> Source code for 3rd approach. This third approach is an extension to the second approach where in we add one more column of cluster the dataframe from the 2nd approach belongs to. It also includes the evaluation of how well the clusters are formed using Silhoutte score and Calinski-Harabasz index. Also the clusters are formed using the elbow point.  And finally test this data on Guassian Naive Baiyes.

- `source/final.py` -> Source code for preparing the dataset for our proposed method. It takes on the Movie Genre, Movie Language and Movie Plots and concates all the string as textual data and form one attribute.

- `source/model.py` -> Source code for our BERT embedding based classification model. This file makes use of the dataset and then tokenizes the sequences and obtains the input ids and attention masks, which are then fed to our embedding classification model that uses bert layer, and then a fully connected layer to classify the revenue for each movie.

- `output/evaluation.py` -> Code for the evaluation of our proposed method

- `source/app.py` -> Source code to run the DialogFlow ChatBot.

## Project Execution

Please refer to the instructions given in this [FILE](RUNNING.md)

## Conclusion

In analyzing the CMU Movie Corpus, we applied Gaussian Naive Bayes (GNB) for classification after we utilized one-hot encoding for categorical values, which gave the baseline for comparison. 

Subsequently, we delved into the narrative content by processing movie plots. After text cleaning, we employed a sequence of techniques: Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), and Singular Value Decomposition (SVD). Extending our analysis, we integrated additional attributes into the dataset and applied K-means clustering before employing GNB. This step aimed to capture deeper patterns in the data, enhancing the model's ability to discern revenue categories. 

Finally, recognizing the importance of contextual information like movie plots, language, and genre, we constructed an advanced model using Bert embeddings. Leveraging these embeddings in a classification framework significantly boosted predictive accuracy compared to previous methods, underscoring the value of contextualized representations in revenue prediction tasks.

Overall, our iterative approach demonstrated the incremental gains achieved by incorporating diverse data representations and advanced modeling techniques for revenue classification.

---

Thank you!

---