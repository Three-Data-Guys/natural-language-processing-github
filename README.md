# Natural Language Processing of Github README

> Predict GitHub repository programming lanuages through Machine Learning NLP.

## Purpose

Predict the main programming language of a Github repository based on the contents of the README

## Acquisition

Scrapped the names of 800 repos, focusing on repos with the primary language being JavaScript, Python, Java, or PHP. Then we made API requests to acquire the README contents.

## Prepare

Reduced files down to keywords using text normalization with string methods, unicode conversion, regex substitution, and word lemmatization.

## Explore

Explored the documents by looking the common words shared between the documents and programming languages, the importance of words in each of the documents and the languages, and difference in the amount of words used by each of the these languages.

## Modeling

Created a model that used a Decision Tree Classifier that uses the individual words of the readme to predict the language of the test set with 75% accurac

### Audience

Diverse audience of individuals with an interest in natural language processing

### Deliverables

- A well-documented jupyter notebook that contains your analysis

- One or two google slides suitable for a general audience that summarize your findings. Include a well-labelled visualization in your slides.

## Data Dictionary

| Feature | Description |
|--- |--- |
| repo | url - github.com/
| language | The programming language the repo is about |
| readme_contents | The string contents of the readme file |
| stemmed | The end of a word is removed to form the base word |
| lemmatized | The word is reduced to base word through the use of dictionary words |
| word_list | List of all of the word captured |
| unique_words | Each word only counted for once |
| non_single_words | Words that appear more then once, anywhere |
| word_count_simple | Count of the number of spaces in the lemmatized words |
| word_count | Count of all of the lemmatized words |
| unique_count | Count of all words, only counted once |
| non_single_count | Count of all words appearing more then once, anywhere |
| percent_unique | Percentage of unique words |
| percent_repeat | Percentage of words that repeated |
| percent_one_word | Percentage of words that appeared only once |
| percent_non_single | Percentage of words that appeared more than once |
|--- |--- |
| Data Frames | Description |
|--- |--- |
| bigram_word_counts | dataframe of biram counts |
| freq_df | dataframe of value counts for each word |
| language_df | dataframe of what language each word shows up in and how often |
| word_counts | dataframe of value count of all words |

## How to Reproduce

1. Clone this repo

2. Run all cells of the file github_nlp.ipynb

### acquire.py

A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.

### prepare.py

wrangle_data() creates a dataframe from data.json after cleaning and preparing it.

### model.py

predict_readme_language(string)  takes a string input and return the programming language
