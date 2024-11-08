import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re 
from sklearn.preprocessing import LabelEncoder
from utils import train_test
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')


lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

# Function for clean the given text by removing stopwords and lemmatization
def clean_text(text):

    plot_words = nltk.word_tokenize(text)
    # Removal of word
    punctutation_free_words = []
    for word in plot_words:
        if word.isalpha():
            punctutation_free_words.append(word.lower())

    # Stopwords Removal
    stop_words = set(stopwords.words('english'))
    stop_word_free_words = []
    for word in punctutation_free_words:
        if word not in stop_words:
            stop_word_free_words.append(word)

    # Lemmatization of the words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stop_word_free_words]
    return ' '.join(lemmatized_words)


def visualize_bow_features(bow_dataframe):
    """
    Visualize Bag of Words (BOW) features using Word Cloud.
    
    Parameters:
    - bow_dataframe: DataFrame containing BOW features.
    """
    print("Visualizing the features from Bag of Words (BOW)...")
    
    # Generate Word Cloud from BOW frequencies
    wc = WordCloud(
        background_color="white",
        width=1000,
        height=1000,
        max_words=100,
        relative_scaling=0.5,
        normalize_plurals=False
    ).generate_from_frequencies(bow_dataframe.sum())
    
    # Display Word Cloud
    plt.imshow(wc)
    plt.savefig('./visualizations/bow.jpeg', format = 'jpeg')

# nltk.download('punkt')
df = pd.read_csv("./data/FinalDataset.csv")
# print(df.columns)
row_count = df.shape[0]
print("Total Number of Records are: ", row_count)

def extract_bow_and_tf_idf_fetaures(plots):
    """
    Perform feature extraction on given plots applying bag of words and tf idf.
    
    Parameters:
    - plots: Input 'Cleaned_Movie_Plot' column.
    
    Returns:
    - Transformed DataFrame after feature engineering.
    """
    print(plots)
    print("Applying Bag of Words (BOW) for feature extraction...")
    vectorizer = CountVectorizer(ngram_range=(1,1))
    bow_transform = vectorizer.fit_transform(plots)
    bow_features_df = pd.DataFrame(bow_transform.toarray(), columns=vectorizer.get_feature_names_out())
    print("Bag of Words (BOW) applied.")
    print(bow_features_df.shape)

    visualize_bow_features(bow_features_df)

    print("Applying TF-IDF for feature extraction...")
    transformer = TfidfTransformer()
    tfidf_transform = transformer.fit_transform(bow_features_df)
    tf_idf_df = pd.DataFrame(tfidf_transform.toarray(), columns=vectorizer.get_feature_names_out())
    print("TF-IDF applied.")
    print(tf_idf_df.shape)

    return tf_idf_df

def feature_engineer(df):
    """
    Perform feature engineering on the given DataFrame.
    
    Parameters:
    - df: Input DataFrame containing 'Movie_Plot' column.
    
    Returns:
    - Transformed DataFrame after feature engineering.
    """
    print("Feature engineering...")
    # print(df.head())
    df['Cleaned_Movie_Plot'] = df['Movie_Plot'].apply(clean_text)
    
    feature_df = extract_bow_and_tf_idf_fetaures(df['Cleaned_Movie_Plot'])

    print("Applying Singular Value Decomposition (SVD) for Latent Semantic Analysis (LSA)...")
    svd = TruncatedSVD(n_components=400, n_iter=2, random_state=42)
    svd_transform_array = svd.fit_transform(feature_df)
    svd_dataframe = pd.DataFrame(svd_transform_array)
    print("SVD applied.")
    print(svd_dataframe.shape)

    return svd_dataframe

result_df = feature_engineer(df)
result_df.to_csv('./data/MoviePlotFeatures.csv')

def get_X_Y():
    label_encoder = LabelEncoder()
    df1 = pd.read_csv("./data/FinalDataset.csv")
    df = pd.read_csv("./data/MoviePlotFeatures.csv")
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]

    Y = label_encoder.fit_transform(df1['Movie_Revenue_Category'])
    return df, Y

X,Y=get_X_Y()
train_test(X,Y)