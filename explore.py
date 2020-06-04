"""This notebook is for helper functions for exploration"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import prepare
import nltk

import re

from wordcloud import WordCloud

df = prepare.wrangle_data()

#### idf viz #######

col = df.lemmatized

def make_sparse_matrix(col, n=15):

    cv = CountVectorizer()
    X = cv.fit_transform(col)

    tfidf = TfidfVectorizer()
    bag_of_words_tfidf = tfidf.fit_transform(col)

    sparse_mask = freq_df['all'].sort_values(ascending = False).head(n).index

    df_of_words_tfidf = pd.DataFrame(X.todense(), columns=tfidf.get_feature_names())
    df_of_words = df_of_words_tfidf[sparse_mask]
    
    return df_of_words

def make_idf_list():
    col_list = list(df_of_words.columns)
    idf_list = []
    for l in col_list:
        x = df_of_words[l].apply(lambda x: x > 0).sum()
        idf_list.append(x)
    return idf_list

def make_viz_df():
    idf_df = pd.DataFrame({'word': col_list,
                    'appearances': idf_list,  
                      })
    idf_df['idf'] = np.log(800 / idf_df.appearances)
    return idf_df

def make_idf_viz():
    
    col = df.lemmatized
    df_of_words = make_sparse_matrix(col, n=15)
    
    idf_list = make_idf_list()
    
    idf_df = make_viz_df()
    
    sns.barplot(x=idf_df.word, y=idf_df.idf)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation='vertical')
    plt.title('IDF for a Given Word')
    plt.show()




####### DAVID ##########
# re
def get_js_words(df):
    js_words = (' '.join(df[df.language == 'JavaScript'].lemmatized))
    return re.sub(r'[^\w\s]', '', js_words).split()

def get_js_freq(df):
    return pd.Series(get_js_words(df), name='JavaScript').value_counts()

    
def get_python_words(df):    
    python_words = (' '.join(df[df.language == 'Python'].lemmatized))
    return re.sub(r'[^\w\s]', '', python_words).split()
    
def get_python_freq(df):
    return pd.Series(get_python_words(df), name='Python').value_counts()


def get_java_words(df):
    java_words = (' '.join(df[df.language == 'Java'].lemmatized))
    return re.sub(r'[^\w\s]', '', java_words).split()

def get_java_freq(df):
    return pd.Series(get_java_words(df), name='Java').value_counts()


def get_php_words(df):
    php_words = (' '.join(df[df.language == 'PHP'].lemmatized))
    return re.sub(r'[^\w\s]', '', php_words).split()
    
def get_php_freq(df):    
    return pd.Series(get_php_words(df), name='PHP').value_counts()

    
def get_all_words(df):
    all_words = (' '.join(df.lemmatized))
    return re.sub(r'[^\w\s]', '', all_words).split()
    
def get_all_freq(df):
    return pd.Series(get_all_words(df)).value_counts()

def get_word_count(df):
    all_freq = get_all_freq(df)
    java_freq = get_java_freq(df)
    js_freq = get_js_freq(df)
    php_freq = get_php_freq(df)
    python_freq = get_python_freq(df)
     
    return (pd.concat([all_freq, js_freq,  python_freq, java_freq, php_freq], axis=1, sort=True)
        .set_axis(['all', 'JavaScript', 'Python', 'Java', 'PHP'], axis=1, inplace=False)
        .fillna(0)
        .apply(lambda s: s.astype(int)))

def viz_most_common(word_counts):
    (word_counts
     .assign(JavaScript=word_counts.JavaScript / word_counts['all'],
             PHP=word_counts.PHP / word_counts['all'],
             Python=word_counts.Python / word_counts['all'],
             Java=word_counts.Java / word_counts['all'])
     .sort_values(by=['all'])
     [['JavaScript','PHP', 'Python','Java']]
     .tail(20)
     .sort_values(by='JavaScript')
     .plot.barh(stacked=True, figsize=(16, 8)))

    plt.title('Proportion of JavaScript vs PHP vs Python vs Java for the 20 most common words')
    
    
# Word Clouds
def get_single_word_cloud(df):
    '''
    Displays a word cloud for the top twenty reoccuring words in each language
    '''
    all_cloud = WordCloud(background_color='gainsboro', height=400, width=1300).generate(' '.join(get_all_words(df)))
    python_cloud = WordCloud(background_color='bisque', height=225, width=500).generate(' '.join(get_python_words(df)))
    js_cloud = WordCloud(background_color='mistyrose', height=225, width=500).generate(' '.join(get_js_words(df)))
    php_cloud = WordCloud(background_color='lightskyblue', height=225, width=500).generate(' '.join(get_php_words(df)))
    java_cloud = WordCloud(background_color='thistle', height=225, width=500).generate(' '.join(get_java_words(df)))

    plt.figure(figsize=(12, 10))
    axs = [plt.axes([0, 2/3, 1, 1/3]) , plt.axes([0, 1/3, .50, 1/3]), plt.axes([.5, 1/3, .50, 1/3]), 
           plt.axes([0, 0, .5, 1/3]), plt.axes([.5, 0, .5, 1/3])]

    axs[0].imshow(all_cloud)
    axs[1].imshow(js_cloud)
    axs[2].imshow(python_cloud)
    axs[3].imshow(php_cloud)
    axs[4].imshow(java_cloud)

    axs[0].set_title('All Words')
    axs[1].set_title('JavaScript')
    axs[2].set_title('Python')
    axs[3].set_title('PHP')
    axs[4].set_title('Java')

    for ax in axs:
        ax.axis('off')


# Bigrams
def top_twenty_bigrams(df):
    '''
    Returns dataframe of bigram counts for each language
    '''
    # twenty most frequent bigrams for all words
    top_20_all_bigrams = (pd.Series(nltk.ngrams(get_all_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for javascript
    top_20_js_bigrams = (pd.Series(nltk.ngrams(get_js_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for java
    top_20_java_bigrams = (pd.Series(nltk.ngrams(get_java_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for php
    top_20_php_bigrams = (pd.Series(nltk.ngrams(get_php_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for python
    top_20_python_bigrams = (pd.Series(nltk.ngrams(get_python_words(df), 2))
                    .value_counts()
                    .head(20))
    
    return (pd.concat([top_20_all_bigrams, top_20_js_bigrams, top_20_java_bigrams, 
                                     top_20_php_bigrams, top_20_python_bigrams], axis=1, sort=True)
        .set_axis(['all_bigram','JavaScript', 'Java', 'PHP', 'Python'], axis=1, inplace=False)
        .fillna(0)
        .apply(lambda s: s.astype(int)))
       
def viz_top_bigram(bigram_word_counts):
    (bigram_word_counts
         .assign(JavaScript=bigram_word_counts.JavaScript / bigram_word_counts['all_bigram'],
                 PHP=bigram_word_counts.PHP / bigram_word_counts['all_bigram'],
                 Python=bigram_word_counts.Python / bigram_word_counts['all_bigram'],
                 Java=bigram_word_counts.Java / bigram_word_counts['all_bigram'])
         .sort_values(by=['all_bigram'])
         [['JavaScript','PHP', 'Python','Java']]
         .tail(20)
         .sort_values(by='JavaScript')
         .plot.barh(stacked=True, figsize=(16, 8)))

    plt.title('Bigram Proportion of JavaScript vs PHP vs Python vs Java for the 20 most common words')
    


def get_bigram_word_cloud(df):
    '''
    Displays the top 20 most reoccuring bigrams for each language 
    '''
    # twenty most frequent bigrams for all words
    top_20_all_bigrams = (pd.Series(nltk.ngrams(get_all_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for javascript
    top_20_js_bigrams = (pd.Series(nltk.ngrams(get_js_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for java
    top_20_java_bigrams = (pd.Series(nltk.ngrams(get_java_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for php
    top_20_php_bigrams = (pd.Series(nltk.ngrams(get_php_words(df), 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for python
    top_20_python_bigrams = (pd.Series(nltk.ngrams(get_python_words(df), 2))
                    .value_counts()
                    .head(20))
    
    all_bigram = {k[0] + ' ' + k[1]: v for k, v in top_20_all_bigrams.to_dict().items()}
    java_bigram = {k[0] + ' ' + k[1]: v for k, v in top_20_java_bigrams.to_dict().items()}
    js_bigram = {k[0] + ' ' + k[1]: v for k, v in top_20_js_bigrams.to_dict().items()}
    php_bigram = {k[0] + ' ' + k[1]: v for k, v in top_20_php_bigrams.to_dict().items()}
    python_bigram = {k[0] + ' ' + k[1]: v for k, v in top_20_python_bigrams.to_dict().items()}
    
    
    
    
    all_words_bigram_cloud = WordCloud(background_color='gainsboro', height=400, width=1300).generate_from_frequencies(all_bigram)
    java_bigram_cloud = WordCloud(background_color='thistle', height=225, width=500).generate_from_frequencies(java_bigram)
    js_bigram_cloud = WordCloud(background_color='mistyrose', height=225, width=500).generate_from_frequencies(js_bigram)
    php_bigram_cloud = WordCloud(background_color='lightskyblue', height=225, width=500).generate_from_frequencies(php_bigram)
    python_bigram_cloud = WordCloud(background_color='bisque', height=225, width=500).generate_from_frequencies(python_bigram)

    plt.figure(figsize=(12, 10))
    
    axs = [plt.axes([0, 2/3, 1, 1/3]) , plt.axes([0, 1/3, .50, 1/3]), plt.axes([.5, 1/3, .50, 1/3]), 
           plt.axes([0, 0, .5, 1/3]), plt.axes([.5, 0, .5, 1/3])]
    
    axs[0].imshow(all_words_bigram_cloud)
    axs[1].imshow(java_bigram_cloud)
    axs[2].imshow(js_bigram_cloud)
    axs[3].imshow(php_bigram_cloud)
    axs[4].imshow(python_bigram_cloud)
    
    axs[0].set_title('All Bigram Words')
    axs[1].set_title('Java')
    axs[2].set_title('JavaScript')
    axs[3].set_title('PHP')
    axs[4].set_title('Python')

    for ax in axs:
        ax.axis('off')


##########################################################