## Making some imports
import pandas as pd
import numpy as np
import pickle
import os 
from data_cleaning import clean_text

## defining constants and paths
news_articles_path = 'data/News/'
news_summaries_path = 'data/Summaries'
news_articles = []
summaries = []

# a function to make all the news articles as a list
def make_news_article(path):
    for files_and_dir in os.walk(path):
        for file in files_and_dir[2]:
            news = open(files_and_dir[0]+ '/' + file, 'r')
            news_articles.append(news.read())
            news.close()

# a function to make all the news summaries as a list
def make_summaries(path):
    for files_and_dir in os.walk(path):
        for file in files_and_dir[2]:
            summary = open(files_and_dir[0]+ '/' + file, 'r')
            summaries.append(summary.read())
            summary.close()


make_news_article(news_articles_path)
make_summaries(news_summaries_path)

#print(news_articles)
#print(summaries)

# Cleaning the news and summaries
clean_news = [clean_text(news) for news in news_articles]
clean_summaries = [clean_text(summary) for summary in summaries]


# Creating a dictionary that maps each word with its frequency
word2count = {}
for news, summary in zip(clean_news, clean_summaries):
    for word in news.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
    for word in summary.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


# Creating dictionaries that maps each word of questions and answers with a unique integer
# The Word2count dict contains the words and their frequency , so we will only keep the words which
# appear a certain number of times.
MIN_WORD_FREQUENCY = 10
word_number = 0
newswords2int = {}
for word, count in word2count.items():
    if count >= MIN_WORD_FREQUENCY:
        newswords2int[word] = word_number
        word_number += 1
word_number = 0
summarywords2int = {}
for word, count in word2count.items():
    if count >= MIN_WORD_FREQUENCY:
        summarywords2int[word] = word_number
        word_number += 1

# Adding special tokens into the dict - <EOS> for end of string , <SOS> for start of string
# <PAD> for user input , <OUT> for filtered out words
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    newswords2int[token] = len(newswords2int) + 1
for token in tokens:
    summarywords2int[token] = len(summarywords2int) + 1

# Creating the inverse dictionary of the answerswords2int dict
summaryints2words = {w_i: w for w, w_i in summarywords2int.items()}

# Adding the <SOS> and <EOS> at the start and end of each string respectively in the 
# clean_answers list as, this is the target .
for i in range(len(clean_summaries) - 1):
    clean_summaries[i] = clean_summaries[i] + ' <EOS>'
    
# Translating all the questions and answers into their respective unique integers.
# and Replacing the words which were filtered out by value of '<OUT>'
news_into_ints = []
for news in clean_news:
    ints = []
    for word in news.split():
        if word not in newswords2int:
            ints.append(newswords2int['<OUT>'])
        else:
            ints.append(newswords2int[word])
    news_into_ints.append(ints)
summary_into_ints = []
for summary in clean_summaries:
    ints = []
    for word in summary.split():
        if word not in summarywords2int:
            ints.append(summarywords2int['<OUT>'])
        else:
            ints.append(summarywords2int[word])
    summary_into_ints.append(ints)
    
# Sorting questions and answers by their lenghts to optimise the traning process
sorted_clean_news = []
sorted_clean_summaries = []
for length in range(1, len(news_into_ints) - 1):
    for i in enumerate(news_into_ints):
        if len(i[1]) == length:
            sorted_clean_news.append(news_into_ints[i[0]])
            sorted_clean_summaries.append(summary_into_ints[i[0]])





