# Research Summarizer

This is a web app , a mobile app and a chrome extention made for the researchers . This app would provide the research papers summary in the main feed , 
where user can bookmark, faorite etc the research paper. Also the user can view the complete research paper and store it as pdf. The UI would be 
more like tinder swipe.

- Research summarization - Deep Learning Tensorflow and Keras models using Seq2Seq architecture and word-level modelling .

- Research Recommendation - Deep Learning Tensorflow and Keras models using Apriori and Eclat Algo.

- Web App - Nodejs + ReactJS + Tensorflowjs

- Mobile App - React Native + Tensorflow Lite

- Chrome Extention - NodeJS + Tensorflowjs

# Current Progress 

Since I am in need of a personal GPU, I have been training the models on the free GPU sessions on Kaggle and colab. Now , since the free sessions have limits
I could not train the model on massive datasets, but I managed to train the models on 1000 sentences of the [AMAZON-FINE-FOOD-REVIEW](https://www.kaggle.com/snap/amazon-fine-food-reviews/) dataset,
since it contains both the text and summary. I also tried the model to run on the BBC news dataset, but , unfortunately the free session could not manage the massive size of the dataset.
Thus I trained the model for 10 epochs , using the GRU cell and seq2seq with attention mechanism . So, being said that , these are the results I have got.

```python
summarize('this is a very healthy dog food . good for their digestion . also good for small puppies . my dog eats her required amount at every feeding',
          encoder, 
          decoder, 
          inp_lang, 
          targ_lang, 
          max_length_inp, 
          max_length_targ)

```

    Input: <start> this is a very healthy dog food . good for their digestion . also good for small puppies . my dog eats her required amount at every feeding <end>
    : love too <end> 
    


![png](summarizer_files/summarizer_22_1.png)





    'love too <end> '
	
# How you can help ?

I am completely estatic , that you have thought of helping me. If you wish to do so, kindly find me and my kernels on KAGGLE, at [https://kaggle.com/justdvnsh](https://kaggle.com/justdvnsh)
