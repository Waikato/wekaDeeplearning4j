#!/bin/bash

java -Xmx5g weka.Run \
     .Dl4jMlpClassifier \
     -t "../src/test/resources/nominal/randomly_generated.arff" \
     -split-percentage 66