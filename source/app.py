from flask import Flask, request
from recommendation import RecommendGenreBasedMovies, RecommendLanguageBasedMovies, RecommendMovieSimilarityBasedMovies, RecommendTimePeriodBasedMovies
from dateutil.parser import parse

app = Flask(__name__)

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    req = request.get_json(silent = True, force = True)
    fulfillmentText = "Hello!! From local webhook."
    intent = req['queryResult']['intent']['displayName']
    if intent == 'MovieSimilarityBased':
        movieName = req['queryResult']['parameters']['MovieName']
        initialText = 'Recommendation of movies based on similarity are as follows: '
        movieList = RecommendMovieSimilarityBasedMovies(movieName)
        fulfillmentText = createResponseString(initialText, movieList)
    elif intent == 'GenreBased':
        genre = req['queryResult']['parameters']['MovieGenreType']
        movieList = RecommendGenreBasedMovies(genre)
        initialText = 'Recommendation of movies based on ' + genre + ' genre are as follows: '
        fulfillmentText = createResponseString(initialText, movieList)
    elif intent == 'LanguageBased':
        language = req['queryResult']['parameters']['MovieLanguage']
        movieList = RecommendLanguageBasedMovies(language)
        initialText = 'Recommendation of movies based on ' + language + ' language are as follows: '
        fulfillmentText = createResponseString(initialText, movieList)
    elif intent == 'ReleaseTimeBased':
        startYear = req['queryResult']['parameters']['date-period']['startDate']
        startYear = parse(startYear, fuzzy=True).year
        endYear = req['queryResult']['parameters']['date-period']['endDate']
        endYear = parse(endYear, fuzzy=True).year
        movieList = RecommendTimePeriodBasedMovies(startYear, endYear)
        initialText = 'Recommendation of movies based on year of release are as follows: '
        fulfillmentText = createResponseString(initialText, movieList)
    else:
        initialText = ''
        fulfillmentText = createResponseString(initialText, [])

    return {
        "fulfillmentText": fulfillmentText,
        "source": "localwebhook"
    }

def createResponseString(initialText, movieNameList):
    resultString = initialText
    if len(movieNameList) == 0:
        return 'Sorry we do not have movie to recommend for your given text. Please try something different.'
    for i in range(len(movieNameList)):
        resultString += str(i+1) + '. ' + movieNameList[i] + '\n '
    return resultString

if __name__ == '__main__':
    app.run(debug = True, port = 2412)