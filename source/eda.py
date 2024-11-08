import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./data/GenreInfo.csv")
# Sort the dataframe by Genre Occurrence in descending order
df_sorted = df.sort_values(by='Genre Occurrence', ascending=False)

# Select the top 10 genres
top_10_genres = df_sorted.head(10).sort_values(by='Genre Occurrence', ascending=True)

# Plotting the top 10 genres
plt.barh(top_10_genres['Genre Name'], top_10_genres['Genre Occurrence'], color='#157CA6')
plt.ylabel('Genre Name')
plt.xlabel('Genre Occurrence')
plt.title('Top 10 Movie Genres')
plt.savefig('./visualizations/genre.jpeg', format = 'jpeg')

df = pd.read_csv("./data/LanguageInfo.csv")
# Sort the dataframe by language Occurrence in descending order
df_sorted = df.sort_values(by='Language Occurrence', ascending=False)

# Select the top 5 genres
top_5_languages = df_sorted.head(5).sort_values(by='Language Occurrence', ascending=True)

# Plotting the top 5 genres
plt.barh(top_5_languages['Language Name'], top_5_languages['Language Occurrence'], color='#157CA6')
plt.ylabel('Language Name')
plt.xlabel('Language Occurrence')
plt.title('Top 5 Movie Languages')
plt.savefig('./visualizations/language.jpeg', format = 'jpeg')

df = pd.read_csv("./data/CountryInfo.csv")
# Sort the dataframe by country Occurrence in descending order
df_sorted = df.sort_values(by='Country Occurrence', ascending=False)

# Select the top 5 countries
top_5_countries = df_sorted.head(5).sort_values(by='Country Occurrence', ascending=True)

# Plotting the top 5 countries
plt.barh(top_5_countries['Country Name'], top_5_countries['Country Occurrence'], color='#157CA6')
plt.ylabel('Country Name')
plt.xlabel('Country Occurrence')
plt.title('Top 5 Movie Production Countries')
plt.savefig('./visualizations/country.jpeg', format = 'jpeg')