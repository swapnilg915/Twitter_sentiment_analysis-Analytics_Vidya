Twitter Sentiment Analysis using SVM and TF-IDF.

This project is on Twitter sentiment analysis. It is a solution for the Analytics vidya competetion - "Twitter Sentiment Analysis".
It classifies the user tweets into 2 classes i.e Hate speech vs Non hate speech.

Trained a SVM model with TF-IDF. It gives very good accuracy and F1 score. 

F1-Score = 76.69 %

Rank - 30 / 6700

Requirements -
python 2
Sklearn
numpy
pandas

command to train and run the model => python svc_sentiment_analyzer.py

steps 

step 1 - Read dataset uing pandas
step 2 - Remove special characters and punctuations like ?, ' " # @. lower the case of all tweets.
step 3 - Remove stopwords. the package used here is many_stop_words which conatins approximately 950 english stopwords.
step 4 - Lemmatize the tweets to reduce the dimensionality. Lemmatizer used here is NLTK's WordNetLemmatizer.
step 5 - convert text tweets to features using TF-IDF. (Train and Test)
step 6 - Train the classifier.
step 7 - Predict the test labels (unseen tweets) using trained model.
step 8 - store the predicted labels in csv file using pandas, for submission.
