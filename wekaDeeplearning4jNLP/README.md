# WekaDeepLearning4jNLP

This is a [WEKA](http://www.cs.waikato.ac.nz/~ml/weka/) package for calculating [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) from String attributes. The packages wraps the implementations from [Deeplearning4j](https://deeplearning4j.org/).


### Filters

 1. __Dl4jStringToWord2Vec__: calculates word embeddings on a string attribute using the [Word2Vec](https://code.google.com/archive/p/word2vec/) method
 2. __Dl4jStringToGlove__: calculates word embeddings on a string attribute using the [Glove]( https://nlp.stanford.edu/projects/glove/) method.

### Loaders

1. __Word2VecLoader__: loads Word2Vec seriliazed embeddings files (bin, bin.gz) into Weka. 


## Installation
Run the following command using WEKA 3.8 or superior:

```bash
java -cp weka.jar weka.core.WekaPackageManager -install-package https://github.com/felipebravom/WekaDeepLearning4jNLP/releases/download/v1.0.0/WekaDeepLearning4jNLP1.0.0.zip
```

	
	
 ## Examples
The package can be used from the Weka GUI or the command line.

 ```bash
java -cp ${WEKA_HOME}/weka.jar .Dl4jStringToWord2Vev \
    -i datasets/text/ReutersCorn-train.arff -allowParallelTokenization \
    -batchSize 512 \
    -learningRate 0.025 \
    -minLearningRate 1.0E-4 \
    -negative 0.0 \
    -sampling 0.0 \
    -useHierarchicSoftmax \
    -action WORD_VECTOR \
    -concat_words 15 \
    -epochs 1 \
    -iterations 1 \
    -layerSize 100 \
    -minWordFrequency 5 \
    -preprocessor "weka.dl4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor " \
    -seed 1 \
    -stopWordsHandler "weka.dl4j.text.stopwords.Dl4jRainbow " \
    -index 1 \
    -tokenizerFactory "weka.dl4j.text.tokenization.tokenizerfactory.TweetNLPTokenizerFactory " \
    -windowSize 5 \
    -workers 2
```
      
