#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -height 28 -imagesLocation ../datasets/nominal/mnist-minimal -numChannels 1 -bs 64 -width 28" \
     -layer "weka.dl4j.layers.OutputLayer " \
	 -early-stopping "weka.dl4j.earlystopping.EarlyStopping -valPercentage 10 -maxEpochsNoImprovement 5" \
     -numEpochs 50 \
     -t ../datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 66
