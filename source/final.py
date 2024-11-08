import json
import pandas as pd


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
# Now moviePlotFileLineList contains all the lines from the file

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


df.drop(columns=['Freebase_Movie_ID','Movie_ID'], inplace=True)

df['Movie_Language'] = ['' for x in range(len(df))]
df['Movie_Genre'] = ['' for x in range(len(df))]

# Clean up Movie_Languages column
df['Movie_Languages_Dict'] = df['Movie_Languages'].apply(lambda row: {key: value.lower()
                                                .replace("language","").strip() for key, value in json.loads(row).items()})
for index, row in df.iterrows():
    r = row['Movie_Languages_Dict']
    l = ''
    for language in r.values():
        l += language + ' '
    df.loc[index, 'Movie_Language'] = l

# Clean up Movie_Genres column
df['Movie_Genres_Dict'] = df['Movie_Genres'].apply(lambda row: {key: value.lower() for key, value in json.loads(row).items()})
for index, row in df.iterrows():
    r = row['Movie_Genres_Dict']
    g = ''
    for genre in r.values():
        g += genre + ' '
    df.loc[index, 'Movie_Genre'] = g

for index, row in df.iterrows():
    df.loc[index, 'Movie_Plot_combined'] = row['Movie_Genre'] + row['Movie_Language'] + "." + row['Movie_Plot']

# Bin movie box office revenue and obtain bins
df['Movie_Revenue_Category'], cutbin = pd.qcut(df['Movie_Box_Office_Revenue'], 4, \
                                               labels=['Low','Low_Med','High_Med','High'], retbins=True)

df = df[['Movie_Revenue_Category', "Movie_Plot_combined"]]
df.rename(columns={'Movie_Plot_combined': 'Movie_Plot'}, inplace=True)

df.to_csv("./data/DataForModel.csv")