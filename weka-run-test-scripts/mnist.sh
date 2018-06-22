#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
	 -S "42" \
	 -normalization "Standardize training data" \
	 -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -height 28 -width 28 -imagesLocation ../src/test/resources/nominal/mnist-minimal -numChannels 1" \
	 -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -rows 3 -columns 3 -strideRows 1 -strideColumns 1" \
	 -layer "weka.dl4j.layers.SubsamplingLayer -rows 2 -columns 2 -strideRows 2 -strideColumns 2" \
	 -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -rows 3 -columns 3 -strideRows 1 -strideColumns 1" \
	 -layer "weka.dl4j.layers.SubsamplingLayer -rows 2 -columns 2 -strideRows 2 -strideColumns 2" \
	 -layer "weka.dl4j.layers.OutputLayer " \
     -t ../datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 66
