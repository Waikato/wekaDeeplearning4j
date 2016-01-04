#!/bin/bash

java -Xmx5g -cp ../build/classes/:$WEKA_HOME/weka.jar:/Users/cjb60/github/weka-fresh/build/testcases/:../lib/* weka.Run \
     .ChrisDL4JClassifier \
     -S 0 \
     -layer "weka.dl4j.layers.DenseLayer -num_units 3 -activation tanh -weight_init XAVIER" \
     -layer "weka.dl4j.layers.OutputLayer -activation softmax -weight_init XAVIER -loss MCXENT" \
     -num_iters 100 \
     -optim STOCHASTIC_GRADIENT_DESCENT \
     -updater SGD \
     -learning_rate 0.1 \
     -momentum 0.9 \
     -t ../datasets/iris.arff \
     -no-cv
