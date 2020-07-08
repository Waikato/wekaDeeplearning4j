#!/bin/bash

java -Xmx5g weka.Run \
    .Dl4jMlpFilter \
    -i ../datasets/nominal/mnist.meta.minimal.arff \
    -c last \
    -iterator ".ImageInstanceIterator -height 224 -width 224 -numChannels 3 -imagesLocation ../src/test/resources/nominal/mnist-minimal" \
    -layer-extract ".DenseLayer -name res4a_branch2b" \
    -layer-extract ".DenseLayer -name flatten_1"
