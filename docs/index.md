# WekaDeeplearning4J: Deep Learning using Weka
![Logo](img/Weka_3_full.png)
WekaDeeplearning4J is a deep learning package for the [Weka](https://www.cs.waikato.ac.nz/ml/weka/index.html) workbench. It is developed to incorporate the modern techniques of deep learning into Weka. The backend is provided by the [Deeplearning4J](https://deeplearning4j.org/) Java library. 

The source code for this package is available on [GitHub](https://github.com/Waikato/wekaDeeplearning4j). The java-doc can be found [here](https://waikato.github.io/wekaDeeplearning4j/).

## Functionality
All functionality of this package is accessible via the Weka GUI, the commandline and programmatically in Java.

The following Neural Network Layers are available to build sophisticated architectures:
 
- **ConvolutionLayer**: applying convolution, useful for images and text embeddings
- **DenseLayer**: all units are connected to all units of its parent layer
- **SubsamplingLayer**: subsample from groups of units of the parent layer by different strategies (average, maximum, etc.)
- **BatchNormalization**: applies the common batch normalization strategy on the activations of the parent layer
- **LSTM**: uses long short term memory approach
- **GlobalPoolingLayer**: apply pooling over time for RNNs and pooling for CNNs applied on sequences
- **OutputLayer**: generates classification / regression outputs

Further configurations can be found in the [Getting Started](user-guide/getting-started.md) and the [Examples](examples) sections.
![Weka workbench GUI](img/gui.png)
