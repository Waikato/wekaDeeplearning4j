#!/bin/bash

java -Xmx5g weka.Run \
    .Dl4jMlpClassifier \
    -zooModel ".KerasEfficientNet -variation EFFICIENTNET_B2" \
    -iterator ".ImageInstanceIterator -height 224 -width 224 -numChannels 3 -imagesLocation ../src/test/resources/nominal/mnist-minimal" \
    -t ../datasets/nominal/mnist.meta.tiny.arff \
    -numEpochs 3 \
    -split-percentage 66
