#!/usr/bin/env bash

java -cp ${WEKA_HOME}/weka.jar weka.Run .Dl4jStringToWord2Vec \
    -i ../wekaDeeplearning4jCore/datasets/text/ReutersCorn-train.arff \
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
