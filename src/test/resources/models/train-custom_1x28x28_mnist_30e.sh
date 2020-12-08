# Run from repository root

java weka.Run .Dl4jMlpClassifier -iterator \
  ".ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -numChannels 1 -height 28 -width 28 -bs 16"  \
  -layer ".ConvolutionLayer -nFilters 8" \
  -layer ".SubsamplingLayer  -poolingType MAX"  \
  -layer ".ConvolutionLayer -nFilters 8 " \
  -layer ".SubsamplingLayer"  \
  -layer ".OutputLayer" \
  -numEpochs 30 -t datasets/nominal/mnist.meta.minimal.arff -split-percentage 80 \
  -d src/test/resources/models/custom_1x28x28_mnist_30e.model
