## Making some imports
import pandas as pd
import numpy as np
import pickle
import os 
import tensorflow as tf
from data_cleaning import preprocess_sentence

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
clean_news = [preprocess_sentence(news) for news in news_articles]
clean_summaries = [preprocess_sentence(summary) for summary in summaries]

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset():    
    word_pairs = [[news, summaries]  for news, summaries in zip(clean_news, clean_summaries)]
    return word_pairs[:30000]
	
pairs = create_dataset()
df = pd.DataFrame({
	'news': [news for news, summaries in pairs],
	'summaries': [summaries for news, summaries in pairs]
})

df.to_csv('dataset_summary.csv')


# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
    
    self.create_index()
    
  def create_index(self):
    for phrase in self.lang:
      self.vocab.update(phrase.split(' '))
    
    self.vocab = sorted(self.vocab)
    
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(self.vocab):
      self.word2idx[word] = index + 1
    
    for word, index in self.word2idx.items():
      self.idx2word[index] = word
	  
def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset():
    # creating cleaned input, output pairs
    pairs = create_dataset()

    # index language using the class defined above    
    inp_lang = LanguageIndex(news for news, summary in pairs)
    targ_lang = LanguageIndex(summary for news, summary in pairs)
    
    # Vectorize the input and target languages
    
    # Spanish sentences
    input_tensor = [[inp_lang.word2idx[s] for s in news.split(' ')] for news, summary in pairs]
    
    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in summary.split(' ')] for news, summary in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar
	
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset()

from sklearn.model_selection import train_test_split

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 4
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 256
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)