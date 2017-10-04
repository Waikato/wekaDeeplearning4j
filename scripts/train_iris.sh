#!/bin/bash

java -Xmx5g -cp $WEKA_HOME/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 0 \
     -layer "weka.dl4j.layers.DenseLayer -units 10 -activation tanh -init XAVIER" \
     -layer "weka.dl4j.layers.OutputLayer -units 3 -activation softmax -init XAVIER -loss MCXENT" \
     -iterator "weka.dl4j.iterators.DefaultDataSetIterator -bs 150 -iters 10" \
     -optim STOCHASTIC_GRADIENT_DESCENT \
     -updater NESTEROVS \
     -lr 0.1 \
     -momentum 0.9 \
     -t ../datasets/nominal/iris.arff \
     -no-cv
