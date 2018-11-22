from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cat = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# twenty_train = load_files('/home/harshams/Documents/Lab/Prog6/20news-bydate-train',categories = cat,encoding='utf-8',decode_error ='ignore')
# twenty_test = load_files('/home/harshams/Documents/Lab/Prog6/20news-bydate-test',categories = cat,encoding='utf-8',decode_error ='ignore')
twenty_train = fetch_20newsgroups(subset = 'train', categories = cat, shuffle = True)
twenty_test = fetch_20newsgroups(subset = 'test', categories = cat, shuffle = True)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_tf = count_vect.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
# X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
mod = MultinomialNB()
mod.fit(X_train_tfidf,twenty_train.target)
X_test_tf = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)
predicted = mod.predict(X_test_tfidf)

print("Accuracy: ",accuracy_score(twenty_test.target,predicted))
print(classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target,predicted))