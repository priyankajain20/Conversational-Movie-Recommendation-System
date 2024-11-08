import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict

def get_top_recommendations(df, type_selected, top=10):
    type_selected = type_selected.lower()
    if type_selected in df.columns:
        type_selected_df = df[df[type_selected] == 1]
        type_selected_df.loc[:,'Movie_Revenue_Category'] = pd.Categorical(type_selected_df['Movie_Revenue_Category'], categories=['High', 'High_Med', 'Low_Med', 'Low'], ordered=True)
        final_df = type_selected_df.sort_values(by='Movie_Revenue_Category')
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
    else:
        return []
    
# Applying model with the cross validation technique
def apply_model_with_cross_validation(model, independent_variables, dependent_variable, folds):
    # skfold = StratifiedKFold(n_splits=folds, shuffle=True)
    results = cross_val_predict(estimator=model, X=independent_variables, y=dependent_variable,cv=folds)
    conf_mat = confusion_matrix(dependent_variable, results)
    class_rep = classification_report(dependent_variable, results)
    return {"Classification Report": class_rep,"Confusion Matrix": conf_mat,"Prediction Results": results}

def train_test(X,Y):
    # Creating Gaussian Naive Bayes classifier to be applied
    model = GaussianNB()
    print("Using Gaussian Naive Bayes classifier...")
    result = apply_model_with_cross_validation(model, X, Y, 10)
    print(result["Classification Report"])
    print(result["Confusion Matrix"])