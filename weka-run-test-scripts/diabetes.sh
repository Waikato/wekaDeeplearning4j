#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator "weka.dl4j.iterators.instance.DefaultInstanceIterator -bs 1" \
     -normalization "Standardize training data" \
     -layer "weka.dl4j.layers.DenseLayer -nOut 10 " \
     -layer "weka.dl4j.layers.OutputLayer -activation \"weka.dl4j.activations.ActivationSoftmax \" -lossFn \"weka.dl4j.lossfunctions.LossMCXENT \"" \
     -numEpochs 10 \
     -t ../datasets/numeric/diabetes_numeric.arff \
     -split-percentage 66
