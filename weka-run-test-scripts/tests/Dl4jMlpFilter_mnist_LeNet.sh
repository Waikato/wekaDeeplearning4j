#!/bin/bash

java -Xmx5g weka.Run \
    .Dl4jMlpFilter \
    -i ../src/test/resources/nominal/mnist_784_train_minimal.arff \
    -c last \
    -iterator ".ConvolutionInstanceIterator" \
    -zooModel ".Dl4jLeNet" \
    -default-feature-layer
