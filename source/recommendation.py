import pandas as pd
from sklearn.metrics import pairwise_distances
from utils import get_top_recommendations
from sklearn.cluster import KMeans

df = pd.read_csv("./data/ClusterFeatures.csv")
df = df.drop(columns=['Unnamed: 0.1'])

def RecommendMovieSimilarityBasedMovies(movieName, top=10):
    movieName = movieName.lower()
    df['Movie_Name_Preprocessed'] = [name.lower() for name in df['Movie_Name']]
    inputMovie_df = df[df['Movie_Name_Preprocessed'] == movieName]
    inputMovie_df = inputMovie_df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name', 'Movie_Name_Preprocessed'])
    df_copy = df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name', 'Movie_Name_Preprocessed'])

    distanceList = pairwise_distances(inputMovie_df, df_copy, metric='euclidean')
    ind = distanceList[0].argsort()[1:top+1]
    
    movieList = [df.iloc[i]['Movie_Name'] for i in ind]
    
    return movieList

def RecommendGenreBasedMovies(genre, top = 10):
    movieList = get_top_recommendations(df, genre, top)
    return movieList

def RecommendLanguageBasedMovies(language, top = 10):
    movieList = get_top_recommendations(df, language, top)
    return movieList

def RecommendTimePeriodBasedMovies(start_year, end_year, top = 10):
    start_year = int(start_year)
    end_year = int(end_year)
    
    time_df = df[df['Movie_Release_Year'] >= start_year]
    time_df = time_df[time_df['Movie_Release_Year'] <= end_year]
    if len(time_df) == 0:
        return []
    final_df = time_df[time_df['Movie_Revenue_Category'] == 'High']
    row_count = final_df.shape[0]
    if row_count < 10:
        final_df = time_df[(time_df['Movie_Revenue_Category'] == 'High') 
                            & (time_df['Movie_Revenue_Category'] == 'High_Med')]
        row_count = final_df.shape[0]
        if row_count < 10:
            final_df = time_df[(time_df['Movie_Revenue_Category'] == 'High') 
                            & (time_df['Movie_Revenue_Category'] == 'High_Med')
                            & (time_df['Movie_Revenue_Category'] == 'Low_Med')]
            row_count = final_df.shape[0]
            if row_count < 10:
                final_df = time_df
    row_count = final_df.shape[0]
    if row_count < top:
        top = row_count
    final_df_copy = final_df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name'])
    kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(final_df_copy)

    movieList = []
    distanceList = pairwise_distances(kmeans.cluster_centers_, final_df_copy, metric='euclidean')
    ind = distanceList[0].argsort()[:top]
    for i in ind:
        row = final_df.iloc[i]
        movieList.append(row['Movie_Name'])
    return movieList
