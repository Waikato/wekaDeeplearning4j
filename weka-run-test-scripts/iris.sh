#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -layer "weka.dl4j.layers.DenseLayer -nOut 32 -activation \"weka.dl4j.activations.ActivationReLU \" " \
     -layer "weka.dl4j.layers.OutputLayer -activation \"weka.dl4j.activations.ActivationSoftmax \" " \
     -numEpochs 10 \
     -t ../datasets/nominal/iris.arff \
     -split-percentage 66
