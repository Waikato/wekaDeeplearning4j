# WekaDeeplearning4J: Deep Learning using Weka

WekaDeeplearning4J is a deep learning package for the [Weka](https://www.cs.waikato.ac.nz/ml/weka/index.html) workbench. It is developed to incorporate the modern techniques of deep learning into Weka. The backend is provided by the [Deeplearning4J](https://deeplearning4j.org/) Java library. 

The source code for this package is available on [GitHub](https://github.com/Waikato/wekaDeeplearning4j). The java-doc can be found [here](https://waikato.github.io/wekaDeeplearning4j/doc/).

![Weka workbench GUI](img/gui.png)

**This documentation is still work in progress.**

## Functionality
All functionality of this package is accessible via the Weka GUI, the commandline and programmatically in Java.

The following Neural Network Layers are available to build sophisticated architectures:
 
- **ConvolutionLayer**: applying convolution, useful for images and text embeddings
- **DenseLayer**: all units are connected to all units of its parent layer
- **SubsamplingLayer**: subsample from groups of units of the parent layer by different strategies (average, maximum, etc.)
- **BatchNormalization**: applies the common batch normalization strategy on the activations of the parent layer
- **OutputLayer**: generates `N` outputs based on a given activation function

