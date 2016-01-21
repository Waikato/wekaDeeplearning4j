#!/bin/bash

java -Xmx5g -cp $WEKA_HOME/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 0 \
     -layer "weka.dl4j.layers.DenseLayer -units 2 -activation relu -init XAVIER" \
     -layer "weka.dl4j.layers.OutputLayer -units 1 -activation identity -init XAVIER -loss MSE" \
     -iters 100 \
     -optim STOCHASTIC_GRADIENT_DESCENT \
     -updater NESTEROVS \
     -lr 0.01 \
     -momentum 0.9 \
     -bs 100 \
     -t ../datasets/diabetes_numeric.arff \
     -no-cv
