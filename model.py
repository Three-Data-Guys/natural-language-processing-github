import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

from prepare import wrangle_data
import prepare

def predict_readme_language(string):
    string = prepare.clean_html(string)
    string = prepare.basic_clean(string)
    string = prepare.tokenize(string)
    string = prepare.remove_stopwords(string)
    string = prepare.lemmatize(string)
    string = cv.transform([string])
    
    return tree.predict(string)

df = wrangle_data()

cv = CountVectorizer()
X = cv.fit_transform(df.lemmatized)
y = df.language

tree = DecisionTreeClassifier(max_depth=10, random_state=13)
tree.fit(X, y)