# Text Processing in WekaDeeplearning4j
There are currently two main deep learning architectures supported to process text data, as explained in the below.
Text can be interpreted as a sequence of so called *tokens*, where a token can be e.g. a character, word, sentence or even a whole document. These tokens can further be mapped with the help of an [embedding](https://en.wikipedia.org/wiki/Word_embedding) into a vector space defined by the embedding. Therefore, a text document can be represented as a sequence of vectors. This can be achieved by using the [Cnn/RnnTextEmbeddingInstanceIterator](data.md#cnnrnntextembeddinginstanceiterator) and providing an embedding that was previously downloaded (e.g. Google's pretrained News model from [here](https://code.google.com/archive/p/word2vec/)). 

### Using Convolutional Neural Networks
To use convolution on text data, it is necessary to correctly preprocess the input into a certain shape and make sure to set the convolution layers accordingly. A good blog post on this is [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/).

A few important things to keep in mind when building a CNN on text data:

- Start with `ConvolutionLayer`
- Adding multiple `ConvolutionLayer` sequentially will simulate an input for each layer and merge the output of all layers to a new output with the number of layer as depth
- `ConvolutionMode` of each `ConvolutionLayer` must be set to `SAME`
- The list of `ConvolutionLayer` must be followed by a `GlobalPoolingLayer`

See also: [Java Examples](../examples/train_rnn.md#java-cnn)

Use the `CnnTextEmbeddingInstanceIterator` or `CnnTextFilesEmbeddingInstanceIterator` accordingly.

### Using Recurrent Neural Networks

The `RnnSequenceClassifier` allows for the construction of neural networks containing recurrent units. The following layer types are supported for these architectures:

- LSTM
- GravesLSTM
- RnnOutputLayer

Use the `RnnTextEmbeddingInstanceIterator` or `RnnTextFilesEmbeddingInstanceIterator` accordingly.
### Embeddings

Currently supported embedding formats are:

- ARFF
- CSV
- CSV gzipped
- Google binary format
- DL4J compressed format

#### Polyglot Embeddings
To convert [Polyglot Embeddings](https://sites.google.com/site/rmyeid/projects/polyglot) into CSV files run the following in python 3.3 or higher (requires numpy and pandas):
```python
import pickle
import numpy as np
import pandas as pd

# Define input/output files
fin = './polyglot-sv.pkl'
fout = './polyglot-sv.csv'

# Load embedding
words, embeddings = pickle.load(open(fin, 'rb'), encoding='latin1')

# Print embedding size
m, n = embeddings.shape
print("Emebddings contains {} words embedded as vectors of length {}".format(m, n))

# Use pandas to export to .csv
print("Starting conversion ...")
df = pd.DataFrame(embeddings)
df.insert(0, "word", words)
df.to_csv(fout, header=False, index=False, sep=" ")

print("Finished conversion from .pkl to .csv")

```

### Weka Filters
We provide two filters that will generate a word embedding from a given text corpus. These will not transform the input data into processed instances (as e.g. `StringToWordVector` does) but output a word embedding that contains tuples of words and their associated vectors in the embedding space. This embedding can be saved and further used in a `Cnn/RnnEmbeddingInstanceIterator` (`location of word vectors` option).

 1. __Dl4jStringToWord2Vec__: calculates word embeddings on a string attribute using the [Word2Vec](https://code.google.com/archive/p/word2vec/) method
 2. __Dl4jStringToGlove__: calculates word embeddings on a string attribute using the [Glove]( https://nlp.stanford.edu/projects/glove/) method.
