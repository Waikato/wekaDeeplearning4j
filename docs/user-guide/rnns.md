# Sequence Classification and Regression

The `RnnSequenceClassifier` allows for the construction of neural networks containing recurrent units. The following layer types are supported for these architectures:

- LSTM
- GravesLSTM
- RnnOutputLayer

## Working with Text Data

This model is a good fit for classification and regression tasks on text data. Text can be interpreted as a sequence of so called *tokens*, where a token can be e.g. a character, word, sentence or even a whole document. These tokens can further be mapped with the help of an [embedding](https://en.wikipedia.org/wiki/Word_embedding) into a vector space defined by the embedding. Therefore, a text document can be represented as a sequence of vectors. This can be achieved by using the [TextEmbeddingInstanceIterator](data.md#textembeddinginstanceiterator) and providing an embedding that was previously downloaded (e.g. Google's pretrained News model from [here](https://code.google.com/archive/p/word2vec/)). 

Currently supported embedding formats are:

- ARFF
- CSV
- CSV gzipped
- Google binary format
- DL4J compressed format