## Making some imports
import pandas as pd
import numpy as np
import pickle
import os 

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

df = pd.DataFrame({
        'News': news_articles,
        'Summary': summaries
    })

print(df.head())

df.to_csv('dataset.txt')
