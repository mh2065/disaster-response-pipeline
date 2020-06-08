# import statements
import sys
import numpy as np
import pandas as pd

# text processing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])
stop_words = stopwords.words('english')

# transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import pickle, joblib
from timeit import default_timer as timer

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix


def load_data(database_filepath):
    '''Load data from sqlite database'''

    # load database table into dataframe
    df = pd.read_sql_table(table_name='disaster_messages', 
                            con='sqlite:///{}'.format(database_filepath))

    # define feature and target variables X and Y
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''Perform text cleaning, normalisation and tokenization on text'''

    # normalize
    # replace urls with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # remove everything except letters and numbers
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # set all text lower case
    text = text.lower()

    # tokenize text string
    tokens = word_tokenize(text)

    # remove stop words
    # tokens = [t for t in tokens if t not in stopwords.words("english")]
    # Calling stopwords.words causes PicklingError (why?)
    # --> use stop_words in CountVectorizer directly instead
    #  --> this causes warning: ['doe', 'ha', 'wa'] not in stop_words
    tokens = [t for t in tokens if t not in stop_words]

    # lemmatize word tokens
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # lemmatize and remove whitespace
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model(parameters):
    '''Setup ML model pipeline'''

    # build pipeline
    # key = step name, value = transformer or estimator object
    pipeline = Pipeline([
        
        # FeatureUnion: enables transformation in parallel before training
        ('features', FeatureUnion([
            
            # process 1: text processing
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, 
                                         # stop_words=stop_words,
                                         # lowercase=True
                                         )),
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
    '''Evaluate outcome of model'''

    # create predictions
    Y_pred = model.predict(X_test)

    # create classification report and convert to dataframe
    report = classification_report(Y_test, Y_pred, 
                                target_names=category_names, 
                                output_dict=True)
    
    # transpose report so that scores are in columns
    report = pd.DataFrame(report).transpose()
    print('\n')
    print(report)
    print('\n')

    # multi lable confusion matrix
    confusion_matrix = multilabel_confusion_matrix(Y_test, Y_pred)

    # open text file
    confusion_matrix_report = open('./models/reports/confusion_matrix.txt', 'w')

    for i in range(len(category_names)):
        text = '{}\n{}\n{}\n'.format(category_names[i], '-' * 20, confusion_matrix[i])
        print(text)   
        confusion_matrix_report.write(text)

    # close text file
    confusion_matrix_report.close()


    # results matrix (select columns)
    cv_results = pd.DataFrame(model.cv_results_)
    cv_results.sort_values(by='rank_test_score', inplace=True)
    cv_results = cv_results[['mean_fit_time',
                            'param_clf',
                            'mean_test_score',
                            'rank_test_score',
                            'mean_train_score']]
    print('\n')
    print(cv_results)

    # GridSearch output
    print('\nGridSearch result:\n' + '-' * 20)
    print('Best score:', model.best_estimator_)
    print('Best estimator:', model.best_estimator_)
    print('Best params:', model.best_estimator_)

    # TODO: write training reports to textfile..
    with pd.ExcelWriter('./models/reports/report.xlsx') as writer:
        report.to_excel(writer, sheet_name='report')
        cv_results.to_excel(writer, sheet_name='cv_results')


    return report, confusion_matrix, cv_results


def save_model(model, model_filepath):
    '''Save the final ML model as pickle file'''

    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    '''Perform all steps to train the classifier'''

    if len(sys.argv) == 3:

        # start execution timer
        start = timer()

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # create train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        

        # define model parameters (estimators + values)
        parameters = [
            {'clf': [RandomForestClassifier()],
            'clf__n_estimators': [10, 20, 100],
            'clf__max_depth': [None],
            'clf__criterion': ['gini']},
            {'clf': [MultiOutputClassifier(LinearSVC())],
            'clf__estimator__C': [10.0, 20.0, 100.0],
            'clf__estimator__max_iter': [1000]},
            {'clf': [OneVsRestClassifier(LogisticRegression())],
            'clf__estimator__C': [10.0, 20.0, 100.0],
            'clf__estimator__solver': ['sag']}
        ]

        '''
        parameters = [
            {'clf': [RandomForestClassifier()],
            'clf__n_estimators': [2, 10, 100],
            'clf__max_depth': [None, 5, 10],
            'clf__criterion': ['gini', 'entropy']},
            {'clf': [MultiOutputClassifier(LinearSVC())],
            'clf__estimator__C': [10.0, 100],
            'clf__estimator__max_iter': [1000, 3000]},
            {'clf': [OneVsRestClassifier(LogisticRegression())],
            'clf__estimator__C': [10.0, 100],
            'clf__estimator__solver': ['sag', 'saga']}
        ]'''

        
        print('Building model...')
        model = build_model(parameters)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        # end timer
        end = timer()
        print("\nTime: {} minutes".format(round((end - start) / 60, 2)))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

