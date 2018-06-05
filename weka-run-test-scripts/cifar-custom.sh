#!/bin/bash

java -Xmx8g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -height 32 -imagesLocation $HOME/datasets/cifar10/train -numChannels 3 -bs 128 -width 32" \
     -normalization "Standardize training data" \
     -zooModel "weka.dl4j.zoo.$1" \
     -iteration-listener "weka.dl4j.listener.EpochListener -n 1" \
	 -config "weka.dl4j.NeuralNetConfiguration -updater \"weka.dl4j.updater.Adam -lr 0.1\"" \
     -numEpochs 50 \
     -t $HOME/datasets/cifar10/cifar10.arff \
     -split-percentage 66
