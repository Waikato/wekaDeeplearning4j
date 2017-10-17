#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -layer "weka.dl4j.layers.OutputLayer -activation \"weka.dl4j.activations.ActivationSoftmax \" -adamMeanDecay 0.9 -adamVarDecay 0.999 -biasInit 1.0 -l1Bias 0.0 -l2Bias 0.0 -blr 0.01 -dist \"weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0\" -dropout 0.0 -epsilon 1.0E-6 -gradientNormalization None -gradNormThreshold 1.0 -L1 0.0 -L2 0.0 -name \"Output layer\" -lr 0.01 -lossFn \"weka.dl4j.lossfunctions.LossMCXENT \" -momentum 0.9 -rho 0.0 -rmsDecay 0.95 -updater NESTEROVS -weightInit XAVIER" \
     -numEpochs 10 \
     -t ../wekaDeeplearning4jCore/datasets/nominal/iris.arff \
     -split-percentage 66
