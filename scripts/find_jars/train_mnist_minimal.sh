#!/usr/bin/env bash

java -verbose:class -Xmx5g -cp $WEKA_HOME/weka.jar weka.Run \
     .Dl4jMlpClassifier -S 2 \
     -layer "weka.dl4j.layers.Conv2DLayer -num_filters 8 -filter_size_x 5 -filter_size_y 5 -stride_x 1 -stride_y 1 -activation relu -init XAVIER" \
     -layer "weka.dl4j.layers.Pool2DLayer -pool_size_x 2 -pool_size_y 2 -stride_x 2 -stride_y 2 -pool_type max" \
     -layer "weka.dl4j.layers.Conv2DLayer -num_filters 16 -filter_size_x 3 -filter_size_y 3 -stride_x 1 -stride_y 1 -activation relu -init XAVIER" \
     -layer "weka.dl4j.layers.Pool2DLayer -pool_size_x 2 -pool_size_y 2 -stride_x 2 -stride_y 2 -pool_type max" \
     -layer "weka.dl4j.layers.Conv2DLayer -num_filters 32 -filter_size_x 3 -filter_size_y 3 -stride_x 1 -stride_y 1 -activation relu -init XAVIER" \
     -layer "weka.dl4j.layers.Pool2DLayer -pool_size_x 2 -pool_size_y 2 -stride_x 2 -stride_y 2 -pool_type max" \
     -layer "weka.dl4j.layers.OutputLayer -units 10 -activation softmax -init XAVIER -p 0.0 -l1 0.0 -l2 0.0 -loss MCXENT" \
     -iterator "weka.dl4j.iterators.ImageDataSetIterator -bs 1 -iters 100 -width 28 -height 28 -channels 1 -location ../../datasets/mnist-minimal" \
     -optim STOCHASTIC_GRADIENT_DESCENT \
     -lr 0.01 -momentum 0.9 -updater NESTEROVS \
     -output-debug-info \
     -t ../../datasets/nominal/mnist.meta.minimal.arff \
     -batch-size 100 \
     -no-cv | grep -v "Loaded java" | grep -v "Loaded sun" | grep -v "Loaded com.sun" | grep -v "jdk.internal" | grep -v "weka.jar" | python extract_jar.py
