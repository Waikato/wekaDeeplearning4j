#!/bin/bash

#java -verbose:class weka.Run .J48 -t ~/datasets/UCI/iris.arff | grep -v "Loaded java" | grep -v "Loaded sun" | grep -v "Loaded com.sun" | grep -v "jdk.internal"
java -verbose:class -Xmx5g -cp $WEKA_HOME/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 0 \
     -layer "weka.dl4j.layers.DenseLayer -units 10 -activation tanh -init XAVIER" \
     -layer "weka.dl4j.layers.OutputLayer -units 3 -activation softmax -init XAVIER -loss MCXENT" \
     -iterator "weka.dl4j.iterators.DefaultDataSetIterator -bs 150 -iters 10" \
     -optim STOCHASTIC_GRADIENT_DESCENT \
     -updater NESTEROVS \
     -lr 0.1 \
     -momentum 0.9 \
     -t ../../datasets/iris.arff \
     -no-cv | grep -v "Loaded java" | grep -v "Loaded sun" | grep -v "Loaded com.sun" | grep -v "jdk.internal" | grep -v "weka.jar" | python extract_jar.py
