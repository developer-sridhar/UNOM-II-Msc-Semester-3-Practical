import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load dataset
msg = pd.read_csv('naivetext.csv', names=['message', 'label'])
print('The Dimensions of the dataset', msg.shape)

# Map labels to numeric
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# --- FIX START ---
# Drop rows where 'labelnum' is NaN
msg.dropna(subset=['labelnum'], inplace=True)
# --- FIX END ---

X = msg.message
Y = msg.labelnum

print(X)
print(Y)

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=42) # Added random_state for reproducibility
print('\nThe Total Number of Training Data:', ytrain.shape)
print('\nThe Total Number of Test Data:', ytest.shape)

# Vectorize text
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

print('\nThe Words or Tokens in the text documents\n')
print(count_vect.get_feature_names_out())

# Create DataFrame of token counts (optional, for inspection)
df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names_out())

# Train Naive Bayes classifier
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

# Evaluate model
print('\nAccuracy of the Classifier is', metrics.accuracy_score(ytest, predicted))
print('\nConfusion Matrix')
print(metrics.confusion_matrix(ytest, predicted))
print('\nThe Value of Precision', metrics.precision_score(ytest, predicted, zero_division=0))
print('\nThe Value of Recall', metrics.recall_score(ytest, predicted, zero_division=0))

