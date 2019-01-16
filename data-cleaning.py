import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import pickle
import re
import string

df = pd.read_csv('dataset.txt')

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
}

def clean_text(text, remove_stopwords = True):
    #converting words to lowercase

    text = text.lower()
    #print('text =>', text)

    new_text = []

    for word in text.split():
        #print(word)
        
        if word in contractions:
            new_text.append(word)
        else:
            if remove_stopwords:
                text = text.split()
                stops = set(stopwords.words("english"))
                text = [w for w in text if not w in stops]
                text = " ".join(text)

    #print(text + '\n')
    punctuations = list(string.punctuation)                
    text = [i for i in text if i not in punctuations]                
    return "".join(text)


clean_summaries = []

for summary in df.Summary:
    text = clean_text(summary, remove_stopwords= False)
    clean_summaries.append(text)
    print('Summary' + summary + ' is cleaned to' + text + "\n")
print('Summaries are cleaned')

clean_news = []
for news in df.News:
    text = clean_text(news)
    clean_news.append(text)
    print('news' + news + '  is cleaned. to ' + text + "##############################" + '\n')
print("News are cleaned.")

#print(clean_summaries[1:5])
#print(clean_news[1:5])


stories = list()

for i, news in enumerate(clean_news):
    stories.append({
            'story': text,
            'highlight': clean_summaries[i]
        })

pickle.dump(stories, open('dataset_story.pkl', 'wb'))







