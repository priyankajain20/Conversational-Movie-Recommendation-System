# Instructions to Execute the Project

## Environment

The project is designed to run on the virtual environment. 
1.      python3 -m venv venv
        source venv/bin/activate
        pip3 install -r requirements.txt

## Requirements for Dialogflow setup

Download the [ngrok](https://ngrok.com/download) and follow the steps to add `auth-token` by [signing up](https://dashboard.ngrok.com/signup) and copying the auth-token from there. 

## Execution Steps

If you want to take a look at all the processing and outputs happening during the execution of the project, kindly follow the instructions given below to run each script individually:

1.     git clone git@github.sfu.ca:psj3/nlpclass-1241-g-NeuralNarratives.git

2.     cd project

Setup the [virtual environment](#environment)

Download the [Model File](https://vault.sfu.ca/index.php/s/eVZGBx8qWlxqrkY) and [Data](https://vault.sfu.ca/index.php/s/x6bNCHyfzsQINTs) provided and paste it in the `project/source` directory.

Now to run the code files for our project, run the following command to navigate to the `source` directory.

3.      cd source

Now to run the baseline file and observe the results for Guassian Naive Baiyes, run the following command. This file creates a ```FinalDataset.csv```, ```GenreInfo.csv```, ```CountryInfo.csv```, ```LanguageInfo.csv```

4.      python3 baseline.py

In order to run the 2nd approach : Feature Engineering using BOW, TF-IDF and SVD, run this command. This file creates the ```MoviePlotFeatures.csv```

5.     python3 featureEngineering.py

For our 3rd approach that is adding a cluster column, run this command. This file creates the ```ClusterFeatures.csv```

6.     python3 clusterFeatureEngineering.py

To run our proposed method, run the below command. Make sure you have downloaded the [Model](https://vault.sfu.ca/index.php/s/eVZGBx8qWlxqrkY) into the `project/source` directory. The `final.py` file creates the ```DataForModel.csv```. 

7.      python3 final.py
        python3 model.py 

This will generate two csv files namely `predicted.csv` and `test.csv` in the `project/output` directory. 

## Evaluation steps

In order to see the evaluation metrics for our proposed BERT Embedding based classification model, run the following command. Make sure you have the [Model](https://vault.sfu.ca/index.php/s/eVZGBx8qWlxqrkY) in the `project/source` directory. 

1.      cd ../output
        python3 evaluation.py


## Instructions to Setup the DialogFlow Chatbot

- Go to [DialogFlow](https://dialogflow.cloud.google.com/#/login) and login using your google credentials.

- Click `Yes` in order to agree to the Terms and Conditions and click on `Accept`.

- Now click on the `Create Agent` button.

- Give the name `NLP_Chatbot` and click on `Create` to create you chatbot agent.

- Now you don't need to create your own `Intent` or `Entities` as I have created it for you. Download the [.zip file](https://vault.sfu.ca/index.php/s/PYi7AsK7xvnWvW6) and [Import](https://botflo.com/courses/dialogflow-es-quickstart-templates/lessons/how-to-export-and-import-dialogflow-es-agent-zip-file/) it and click `Save`

- Now if you click on `Intents` or `Entities` on the left, you would see that new intents and entities have been created.

- For the next step, make sure you have installed [ngrok](#instructions-to-setup-the-dialogflow-chatbot)

- Follow the steps mentioned in this [Link](https://www.codersarts.com/post/how-integrated-python-with-dialogflow-how-to-build-a-chatbot-using-python-and-dialogflow-ngrok) and run this command ```python3 app.py```. Note that our code works for `port 2412 `. Therefore in order to make a ngrok tunnel, run this command on your terminal ```ngrok http 2412```.

- After the link for ChatBot is successfully established, give the chatbot a try and have some fun with it!

---

Thank You!

---
