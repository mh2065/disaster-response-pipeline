### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Instructions (MS Windows):

1. Run the following commands in the project's root directory to set up your database and model:

    - To run ETL pipeline that cleans data and stores in database:

        python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db

    - To run ML pipeline that trains classifier and saves:

        python ./models/train_classifier.py ./data/DisasterResponse.db ./models/classifier.pkl

2. Run the following command in the app's directory to run your web app:

        python run.py

3. Open browser an go to:

        localhost:3001



There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

Classification of disaster response messages and using trained classifier in web app.

This project is part of the Udacity Data Science nanodegree.

## File Descriptions <a name="files"></a>

process_data.py
train_classifier.py
run.py

## Results<a name="results"></a>

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit must be given to Figure 8 for the data.

Feel free to use the code here as you like!
