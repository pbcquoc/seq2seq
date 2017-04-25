# Making chatbot using naive seq2seq model

# Overview
  The input is twitter chat that contains questions and their responsed answers. I clean all messy characters.
  In training state, we forward questions and try to predict answers. The model is followed by
  ![GitHub Logo](http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png)

  In sampling pharse, i use multinomial distribution to sampling next word.

# Dependencies
  * Python 2.7
  * Tensorflow 1.xx
  * I only test it on Ubuntu 16.04
# Preprocess
  All sequences is padded to be same fixed length by using PAD token, and all questions is reversed according to Sutskever et al., 2014  
  > Q : [ PAD, PAD, PAD, PAD, PAD, PAD, “?”, “you”, “are”, “How” ]  
  > A : [ GO, “I”, “am”, “fine”, “.”, EOS, PAD, PAD, PAD, PAD ]
# Training
  I use tensorflow so if you haven't installed tensorflow yet, just following the [link](https://www.tensorflow.org/install/) and install approriate tensorflow for your pc
  To train model 
```python
python seq2seq.py
```
  Here i plot chart of cross-entropy loss. the loss is still going down at the end of training phase. It mean the model is learning somethings.
  ![png](https://github.com/pbcquoc/pbcquoc.github.io/blob/master/media/img/seq2seq/training_phrase.png)

  I show somes answers at iter
  > Q: if youre gonna fall asleep in class just stay home  
  > A: 
  > Q: this was my favorite play of his jumped off the screen
  > A:
  > Q: former president george hw bush said hell vote for hillary clinton according to sources close to bush
  > A:
# Evaluate
