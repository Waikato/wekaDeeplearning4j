#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator "weka.dl4j.iterators.ImageDataSetIterator -height 28 -imagesLocation ../wekaDeeplearning4jCore/datasets/nominal/mnist-minimal -numChannels 1 -bs 16 -width 28" \
     -normalization "Standardize training data" \
     -zooModel "weka.dl4j.zoo.LeNet" \
     -iterationListener "weka.dl4j.listener.BatchListener" \
     -numEpochs 100 \
     -t ../wekaDeeplearning4jCore/datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 66
