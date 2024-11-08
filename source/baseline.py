import pandas as pd
import re 
from sklearn.preprocessing import LabelEncoder
from utils import train_test
import pandas as pd
import numpy as np
import json
from dateutil.parser import parse
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("./data/movie.metadata.tsv", sep ='\t', header= None)
df.columns = ['Movie_ID','Freebase_Movie_ID','Movie_Name','Movie_Release_Date','Movie_Box_Office_Revenue','Movie_Runtime','Movie_Languages','Movie_Countries','Movie_Genres']
df.dropna(inplace=True)

# getting number of rows in dataset
rows = df.shape[0]
print('Number of Records: ',rows)

# Working on fetching the movie plot into dataframe
moviePlotFile = './data/plot_summaries.txt'

with open(moviePlotFile, "r", encoding="utf8") as file:
    moviePlotFileLineList = [line for line in file]

moviePlotList = []

for row in df['Movie_ID']:
    plotLine = next((line for line in moviePlotFileLineList if str(row) in line), None)
    if plotLine:
        _, _, plot = plotLine.partition('\t')
        plot = plot.replace("\n", "").replace("\\", "").strip()
        moviePlotList.append(plot)
    else:
        moviePlotList.append('')

df["Movie_Plot"] = moviePlotList

# Removing rows with empty Movie_Plot columns
df = df[df["Movie_Plot"] != '']

# Get the updated row count
row_count = df.shape[0]
print('Number of Records:', row_count)

# Write movie names in a csv file
df['Movie_Name'].to_csv('./data/MovieNames.csv', index=False)

# Bin movie box office revenue and obtain bins
df['Movie_Revenue_Category'], cutbin = pd.qcut(df['Movie_Box_Office_Revenue'], 4,labels=['Low','Low_Med','High_Med','High'], retbins=True)

# Clean up Movie_Languages column
df['Movie_Languages_Dict'] = df['Movie_Languages'].apply(lambda row: {key: value.lower()
                                                                      .replace("language","").strip() for key, value in json.loads(row).items()})

# Count language occurrences and create language columns
language_counter = Counter()
for row in df['Movie_Languages_Dict']:
    for language in row.values():
        language_counter[language] += 1
        df[language] = df['Movie_Languages_Dict'].apply(lambda x: 1 if language in x.values() else 0)

# Convert language counter to DataFrame
language_dataframe = pd.DataFrame.from_dict(language_counter, orient='index', columns=['Language Occurrence'])
language_dataframe.index.name = 'Language Name'

# Write DataFrame to CSV
language_dataframe.to_csv('./data/LanguageInfo.csv')

# Print total number of languages
print("Total Number of Languages:", len(language_counter))

# Clean up Movie_Countries column
df['Movie_Countries_Dict'] = df['Movie_Countries'].apply(lambda row: {key: value.lower() for key, value in json.loads(row).items()})

# Count country occurrences and create country columns
country_counter = Counter()
for row in df['Movie_Countries_Dict']:
    for country in row.values():
        country_counter[country] += 1
        df[country] = df['Movie_Countries_Dict'].apply(lambda x: 1 if country in x.values() else 0)

# Convert country counter to DataFrame
country_dataframe = pd.DataFrame.from_dict(country_counter, orient='index', columns=['Country Occurrence'])
country_dataframe.index.name = 'Country Name'

# Write DataFrame to CSV
country_dataframe.to_csv('./data/CountryInfo.csv')

# Print total number of countries
print("Total Number of Countries:", len(country_counter))

# Clean up Movie_Genres column
df['Movie_Genres_Dict'] = df['Movie_Genres'].apply(lambda row: {key: value.lower() for key, value in json.loads(row).items()})

# Count genre occurrences and create genre columns
genre_counter = Counter()
for row in df['Movie_Genres_Dict']:
    for genre in row.values():
        genre_counter[genre] += 1
        df[genre] = df['Movie_Genres_Dict'].apply(lambda x: 1 if genre in x.values() else 0)

# Convert genre counter to DataFrame
genre_dataframe = pd.DataFrame.from_dict(genre_counter, orient='index', columns=['Genre Occurrence'])
genre_dataframe.index.name = 'Genre Name'

# Write DataFrame to CSV
genre_dataframe.to_csv('./data/GenreInfo.csv')

# Print total number of genres
print("Total Number of Genres:", len(genre_counter))

# Extract release years from Movie_Release_Date
releaseYearList = []
for row in df['Movie_Release_Date']:
    year = parse(row, fuzzy=True).year
    releaseYearList.append(year)

df['Movie_Release_Year'] = releaseYearList

# Drop the columns which will no more be needed
columns_to_drop = ['Freebase_Movie_ID', 'Movie_Release_Date', 'Movie_Box_Office_Revenue', 'Movie_Languages', \
                   'Movie_Languages_Dict', 'Movie_Countries', 'Movie_Countries_Dict', 'Movie_Genres', 'Movie_Genres_Dict']
df.drop(columns=columns_to_drop, inplace=True)

# Write the DataFrame to CSV file
df.to_csv('./data/FinalDataset.csv', index=False)

def get_X_Y():
    label_encoder = LabelEncoder()
    df = pd.read_csv("./data/FinalDataset.csv")
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]

    Y = label_encoder.fit_transform(df['Movie_Revenue_Category'])
    df = df.drop(columns=[ 'Movie_ID', 'Movie_Plot', 'Movie_Name','Movie_Revenue_Category'])
    return df, Y

X,Y=get_X_Y()
train_test(X,Y)

