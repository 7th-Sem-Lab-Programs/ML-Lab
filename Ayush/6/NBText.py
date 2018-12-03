from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cat = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train',categories=cat,shuffle='true')
test = fetch_20newsgroups(subset='test',categories=cat,shuffle='true')

from sklearn.feature_extraction.text import CountVectorizer
cnt_vect = CountVectorizer()
X_train_tf = cnt_vect.fit_transform(train.data);

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_trans = TfidfTransformer()
X_train_tfidf = tfidf_trans.fit_transform(X_train_tf);

from sklearn.naive_bayes import MultinomialNB
mod = MultinomialNB()
mod.fit(X_train_tfidf, train.target)
X_test_tf = cnt_vect.transform(test.data)
X_test_tfidf = tfidf_trans.transform(X_test_tf)
predicted = mod.predict(X_test_tfidf)

print("Accuracy   : ",accuracy_score(test.target,predicted))
print("Report     : \n",classification_report(test.target,predicted))
print("Conf Matrix: \n",confusion_matrix(test.target,predicted))
