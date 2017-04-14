# Making chatbot using naive seq2seq model

# Overview
  The input is twitter chat that contains questions and their responsed answers. I clean all messy characters.
  In training state, we forward questions and try to predict answers. The model is followed by
  ![GitHub Logo](http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png)

# Dependencies
  * Python 2.7
  * Tensorflow 1.xx
  * I only test it on Ubuntu 16.04
# Preprocess
  All sequences is padded to be same fixed length by using PAD token, and all questions is reversed according to Sutskever et al., 2014  
  > Q : [ PAD, PAD, PAD, PAD, PAD, PAD, “?”, “you”, “are”, “How” ]
  > A : [ GO, “I”, “am”, “fine”, “.”, EOS, PAD, PAD, PAD, PAD ]
# Training
  I use tensorflow so if you haven't installed tensorflow yet, just following the link and install approriate tensorflow for your pc
# Evaluate
