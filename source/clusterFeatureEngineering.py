import pandas as pd
import re 
from sklearn.preprocessing import LabelEncoder
from utils import train_test
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score,pairwise_distances

# Read Final Dataset and Movie Plot Features CSV files
finalDataset_df = pd.read_csv("./data/FinalDataset.csv")
moviePlot_df = pd.read_csv("./data/MoviePlotFeatures.csv")
# Drop unnecessary columns from Final Dataset
finalDataset_df_copy = finalDataset_df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name'])

# Concatenate DataFrames
new_df = pd.concat([finalDataset_df_copy, moviePlot_df], axis=1)
df1 = new_df[new_df.isna().any(axis=1)]

# Selecting k=4 clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(new_df)

# Add cluster feature to the final dataset
finalDataset_df["Cluster_Feature"] = clusters

# Concatenate final dataset with movie plot features
clusterFeature_df = pd.concat([finalDataset_df, moviePlot_df], axis=1)
clusterFeature_df.to_csv('./data/ClusterFeatures.csv')
 
silhouette_avg = silhouette_score(new_df, clusters)
print("The average silhouette score is :", silhouette_avg)

# Calculate the Calinski-Harabasz index
ch_score = calinski_harabasz_score(new_df, clusters)
print("The Calinski-Harabasz index is :", ch_score)

# Calculate the average distance between each point and its cluster centroid
closeness = []
for i in range(4):
    cluster_points = new_df[clusters == i]
    centroid = kmeans.cluster_centers_[i]
    dist = pairwise_distances(cluster_points, [centroid])
    closeness.append(dist.mean())

# Calculate the distance between each pair of cluster centroids
separation = pairwise_distances(kmeans.cluster_centers_)

# Calculate mean closeness and separation
mean_closeness = sum(closeness) / len(closeness)
mean_separation = separation.mean()

# Print the results
print(f"Closeness within Clusters: {mean_closeness}")
print(f"Separation between Clusters: {mean_separation}")

def get_X_Y():
    label_encoder = LabelEncoder()
    df = pd.read_csv("./data/ClusterFeatures.csv")
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]

    Y = label_encoder.fit_transform(df['Movie_Revenue_Category'])
    df = df.drop(columns=[ 'Movie_ID', 'Movie_Plot', 'Movie_Name','Movie_Revenue_Category'])
    return df, Y

X,Y=get_X_Y()
train_test(X,Y)