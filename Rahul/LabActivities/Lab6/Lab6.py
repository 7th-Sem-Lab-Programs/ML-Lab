from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn import metrics

#Selecting news categories
categories = ['sci.space', 'soc.religion.christian', 'comp.graphics', 'sci.med','sci.crypt']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

#Applying Tfidf 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#initialising the classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

#Classifying new Sample Input 
print("Sample input: ")
docs_new = ['Sun is a star', 'WPA2 is compromised', 'Jesus saves us all','Anti depressants have side effects']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

#Performance evaluation
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
X_test_counts = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = clf.predict(X_test_tfidf)

print("\nAccuracy: ",np.mean(predicted == twenty_test.target))
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
print("confusion matrix: \n",metrics.confusion_matrix(twenty_test.target, predicted))

