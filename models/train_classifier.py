import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import nltk
import pickle
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Load data from the given database

    output:
            Dataframe
    '''
    # create connection to the database and read the data
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('messages',engine) 
    
    # split into X and y
    X = df.message
    y = df.iloc[:,4:]
    
    return X, y, y.columns


def tokenize(text):
    '''
    tokenize and lemmatize the given text

    output:
            List pf tokens
    '''
    # tokenize messages
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize and clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build the model pipeline

    output:
            Pipeline
    '''
    # create the model pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(smooth_idf=True)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model and print the results
    Method used for evaluation is a classification_report of every class

    '''
    # get the results and convert them into dataframe to print evaluation
    results = model.predict(X_test)
    results_df = pd.DataFrame(results, columns = category_names)
    
    # print the evaluation
    print('-------------------- Model evaluation ---------------------')
    for c in results_df.columns:
        print(f'class {c}: ------------------------')
        print(classification_report(Y_test[[c]], results_df[[c]]))


def save_model(model, model_filepath):
    '''
    Save the given model as pickle in the provided path

    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()