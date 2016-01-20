#!/bin/bash

java -Xmx5g -cp ../build/classes/:$WEKA_HOME/weka.jar:/Users/cjb60/github/weka-fresh/build/testcases/:../lib/* weka.Run \
     .Dl4jMlpClassifier \
     -S 0 \
     -layer "weka.dl4j.layers.DenseLayer -units 10 -activation tanh -init XAVIER" \
     -layer "weka.dl4j.layers.OutputLayer -units 3 -activation softmax -init XAVIER -loss MCXENT" \
     -iters 100 \
     -optim STOCHASTIC_GRADIENT_DESCENT \
     -updater NESTEROVS \
     -lr 0.1 \
     -momentum 0.9 \
     -bs 1 \
     -t ../datasets/iris.arff \
     -no-cv
