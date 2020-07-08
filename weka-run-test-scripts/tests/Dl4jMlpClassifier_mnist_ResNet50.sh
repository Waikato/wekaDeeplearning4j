#!/bin/bash

java -Xmx8g weka.Run \
    .Dl4jMlpClassifier \
    -zooModel ".Dl4jResNet50" \
    -iterator ".ImageInstanceIterator -bs 8 -height 224 -width 224 -numChannels 3 -imagesLocation ../src/test/resources/nominal/mnist-minimal" \
    -t ../datasets/nominal/mnist.meta.tiny.arff \
    -numEpochs 3 \
    -split-percentage 66
