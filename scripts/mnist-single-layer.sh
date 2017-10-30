#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -height 28 -imagesLocation ../wekaDeeplearning4jCore/datasets/nominal/mnist-minimal -numChannels 1 -bs 64 -width 28" \
     -layer "weka.dl4j.layers.OutputLayer " \
     -numEpochs 10 \
     -t ../wekaDeeplearning4jCore/datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 66
