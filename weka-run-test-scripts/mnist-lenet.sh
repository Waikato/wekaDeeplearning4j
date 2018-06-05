#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -height 28 -imagesLocation $HOME/datasets/nominal/mnist-full -numChannels 1 -bs 128 -width 28" \
     -normalization "Standardize training data" \
     -zooModel "weka.dl4j.zoo.LeNet" \
     -iteration-listener "weka.dl4j.listener.EpochListener -n 1" \
     -numEpochs 10 \
     -t $HOME/datasets/nominal/mnist.meta.full.arff \
     -split-percentage 66
