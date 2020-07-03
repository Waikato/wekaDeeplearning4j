#!/bin/bash

java -Xmx5g weka.Run \
    .Dl4jMlpFilter \
    -i ../datasets/nominal/mnist.meta.minimal.arff \
    -c last \
    -iterator ".ImageInstanceIterator -height 224 -width 224 -numChannels 3 -imagesLocation ../src/test/resources/nominal/mnist-minimal" \
    -zooModel ".KerasEfficientNet -variation EFFICIENTNET_B1" \
    -layer-extract ".DenseLayer -name block4c_expand_conv" \
    -layer-extract ".DenseLayer -name top_dropout"
