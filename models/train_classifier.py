# import statements
import sys
import numpy as np
import pandas as pd

# text processing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# tools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix


def load_data(database_filepath):
    '''docstring'''

    # load database table into dataframe
    df = pd.read_sql_table(table_name='disaster_messages', 
                            con='sqlite:///{}'.format(database_filepath))

    # define feature and target variables X and Y
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''docstring'''

    # replace urls with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # remove everything except letters and numbers
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # tokenize string
    tokens = word_tokenize(text)

    # lemmatize word tokens
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize and remove whitespace
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model(parameters):
    '''docstring'''

    # build pipeline
    # key = step name, value = transformer or estimator object
    pipeline = Pipeline([
        
        # FeatureUnion: enables transformation in parallel before training
        ('features', FeatureUnion([
            
            # process 1: text processing
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
            
            # process 2: placeholder for parallel step           
        ])),
        
        # training estimator
        ('clf', RandomForestClassifier())
    ])
       
    # instantiate GridSearch with pipeline and parameters
    model = GridSearchCV(estimator=pipeline, 
                         param_grid=parameters, 
                         n_jobs=-1, 
                         verbose=2, 
                         return_train_score=True)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''docstring'''

    # create predictions
    Y_pred = model.predict(X_test)

    # create classification report and convert to dataframe
    # TODO: set index category_names
    report = classification_report(Y_test, Y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print('\n')
    print(report)
    
    # multi lable confusion matrix
    # TODO: category_names labels?
    confusion_matrix = multilabel_confusion_matrix(Y_test, Y_pred)
    print('\n')
    print(confusion_matrix)

    # GridSearch output
    print('Best score:', model.best_estimator_)
    print('Best estimator:', model.best_estimator_)
    print('Best params:', model.best_estimator_)

    # results matrix (select columns)
    # TODO: test in jupyter
    cv_results = pd.DataFrame(model.cv_results_);
    print('\n')
    print(cv_results)

    return report, confusion_matrix, cv_results


def save_model(model, model_filepath):
    '''docstring'''

    # TODO: complete function
    pass
    


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # create train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        

        # define model parameters (estimators + values)
        parameters = [
            {"clf": [RandomForestClassifier()],
            "clf__n_estimators": [1, 2]},
            {"clf": [MultiOutputClassifier(LinearSVC())],
            "clf__estimator__C": [1.0, 2]},
            {'clf': [OneVsRestClassifier(LogisticRegression())],
            'clf__estimator__solver': ['sag', 'saga']}
        ]


        print('Building model...')
        model = build_model(parameters)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

