# Disaster REsponse

Installations:

There is a requirements file to install the reqired libraries

-----------------------------------------
Project Motivation:

This project is part of the DataScience Nanodegree from udacity
The idea is the to apply what I learned in nlp and data Engineering.<br>
The project is a Disaster Response to classify messages into different categories of disasters and then display the results through web app.

------------------------------------------

File Descriptions:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # to proccess the data
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py # to train and save a classifier
|- classifier.pkl  # saved model 

Usage:

Run the web app, from app directory: `python run.py`

---------------------------------------------

Acknowledgment:

I would like to thank <a href=https://appen.com/#get_in_touch>Appen </a> for the data.
