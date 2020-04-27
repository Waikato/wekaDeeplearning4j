# The MNIST Dataset - Feature Extraction
  
![Mnist Example 0](../img/mnist/img_11854_0.jpg)
![Mnist Example 1](../img/mnist/img_11253_1.jpg)
![Mnist Example 2](../img/mnist/img_10320_2.jpg)
![Mnist Example 3](../img/mnist/img_10324_3.jpg)
![Mnist Example 4](../img/mnist/img_40694_4.jpg)
![Mnist Example 5](../img/mnist/img_10596_5.jpg)
![Mnist Example 6](../img/mnist/img_19625_6.jpg)
![Mnist Example 7](../img/mnist/img_12452_7.jpg)
![Mnist Example 8](../img/mnist/img_10828_8.jpg)
![Mnist Example 9](../img/mnist/img_10239_9.jpg)

Instead of training a full neural network on your dataset, you may like to try using a pretrained model 
as a feature extractor and fitting a more simple model to those features.

You can use any pretrained model in the model zoo for this task, so try a few and see which works best for your task!

### Concatenating Activations
By default the filter takes features from the final dense/pooling layer of the model (before the classification layer). 
You can also take features from any intermediary layer and concatenate them.

### Activation Pooling 
An important parameter when using intermediate layers is the filter's `PoolingType`. 
Activations from intermediate layers are often 3-dimensional for a given instance, so they need
to be reduced into a 1-dimensional vector. 
There are 4 pooling methods currently supported:
- `PoolingType.MAX` (default)
- `PoolingType.AVG`
- `PoolingType.SUM`
- `PoolingType.MIN`
 
These pool the 2nd and 3rd dimension into a single value, i.e., activations of 
[512, 26, 26] (512 26x26 feature maps) are pooled into shape [512]. You can also specify `PoolingType.NONE`
which simply flattens the extra dimensions (aforementioned example would become shape [346112]). 
`PoolingType` does not need to be specified when using the default activation layer - the outputs are already the
 correct dimensionality.

## Example 1: Default MNIST Minimal
The following example walks through using a pretrained ResNet50 (from the Deeplearning4j model Zoo)
as a feature extractor on the MNIST dataset, and fitting Weka's Random Forest algorithm to the dataset.
This only takes 1-2 minutes on a modern CPU &mdash; much faster than training a neural network from scratch.

The steps shown below split this into two steps; storing the featurized dataset, and fitting a Weka classifier to the dataset.
It can be combined into a single command with a filtered classifier, however, the method shown below
is more efficient as the dataset featurizing (which is the most expensive part of this operation) 
is only done once (would be done 10 times using 10-fold CV with a FilteredClassifier). It's then
much faster to try out different Weka classifiers.

Note that the first time this is run, it will need to download the pretrained weights, so actual runtime
may be longer. 

### Commandline
```bash
$ java -cp $WEKA_HOME/weka.jar weka.Run \
    .Dl4jMlpFilter \
        -i datasets/nominal/mnist.meta.minimal.arff \
        -o mnist-rn50.arff \
        -c last \
        -decimal 20 \
        -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -bs 12" \
        -layer-extract "weka.dl4j.layers.DenseLayer -name flatten_1" \
        -isZoo true 
        -zooModel "weka.dl4j.zoo.Dl4JResNet50"
```
We now have a standard `.arff` file that can be fit to like any numerical dataset
```bash
$ java -cp $WEKA_HOME/weka.jar weka.Run .RandomForest -t mnist-rn50.arff
```


### Java
```java
// Load the dataset
Instances instances = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
instances.setClassIndex(1);
Dl4jMlpFilter myFilter = new Dl4jMlpFilter();

// Create our iterator, pointing it to the location of the images
ImageInstanceIterator imgIter = new ImageInstanceIterator();
imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
// Set batch size to thread count if using CPU
imgIter.setTrainBatchSize(12);
myFilter.setImageInstanceIterator(imgIter);

// Load our pretrained model
Dl4JResNet50 zooModel = new Dl4JResNet50();
myFilter.setZooModelType(zooModel);

// Run the filter, using the model as a feature extractor
myFilter.setInputFormat(instances);
Instances transformedInstances = Filter.useFilter(instances, myFilter);

// You could save the instances at this point to an arff file to be used with other classifiers via:
// https://waikato.github.io/weka-wiki/formats_and_processing/save_instances_to_arff/

// CV our Random Forest classifier on the extracted features
Evaluation evaluation = new Evaluation(transformedInstances);
int numFolds = 10;
evaluation.crossValidateModel(new RandomForest(), transformedInstances, numFolds, new Random(1));
System.out.println(evaluation.toSummaryString());
System.out.println(evaluation.toMatrixString());
```

### GUI

The first step is to open the MNIST meta ARFF file in the Weka Explorer `Preprocess` tab via `Open File`. 
A randomly sampled MNIST dataset of 420 images is provided in the WekaDeeplearning4j package for testing purposes 
(`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist.meta.minimal.arff`). 

Then, select the the `Dl4jMlpFilter` in the filter panel. Click in the box to open the filter settings.

![Classifier](../img/gui/featurize-std-filter.png)

To correctly load the images it is further necessary to select the `Image-Instance-Iterator` as `instance iterator` 
and point it to the MNIST directory that contains the actual image files (`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist-minimal/`). 

If running on CPU then set mini-batch size to your machine's thread count. 
If you run into GPU memory issues then use a smaller mini-batch size.

![Image Instance Iterator](../img/gui/image-instance-iterator.png)

`Dl4jResNet50` is already selected as the feature extractor model, and will by default use the final dense layer
 activations as the image features.
The other filter options can be left as default; they'll be explained in the next example.

Click `Ok` and `Apply` to begin processing your dataset. After completion, you should see your newly processed dataset!

![Processed Dataset](../img/gui/featurize-std-processed.png)

Simply switch to the `Classify` tab to start applying different WEKA classifiers to your newly transformed dataset.  
### Results
Using `RandomForest` gives us 83% accuracy - certainly not SOTA but given the simplicity and speed of the method it's not bad!

```text
Correctly Classified Instances         349               83.0952 %
Incorrectly Classified Instances        71               16.9048 %
Kappa statistic                          0.812 
Mean absolute error                      0.12  
Root mean squared error                  0.2179
Relative absolute error                 66.6557 %
Root relative squared error             72.629  %
Total Number of Instances              420     

=== Confusion Matrix ===

  a  b  c  d  e  f  g  h  i  j   <-- classified as
 39  0  0  0  1  1  0  0  0  0 |  a = 0
  0 46  0  0  0  0  1  0  0  0 |  b = 1
  2  0 34  3  0  0  2  0  0  0 |  c = 2
  0  0  2 36  1  3  0  0  1  1 |  d = 3
  3  1  0  0 30  0  1  0  0  6 |  e = 4
  0  0  0  7  0 28  0  1  1  1 |  f = 5
  0  0  0  0  0  1 40  0  0  0 |  g = 6
  0  0  0  1  0  0  0 41  0  2 |  h = 7
  0  0  0  4  1  4  1  1 28  2 |  i = 8
  1  0  1  0  3  0  0 10  0 27 |  j = 9
```

## Example 2: MNIST Using Activation Layer Concatenation and Pooling
This example shows concatenating an intermediary convolution layer (`res4a_branch2b`) to the default layer (`flatten_1`) and using `PoolingType.AVG`
to average pool the extra dimensions from `res4a_branch2b`.

### Commandline
```bash
$ java -cp $WEKA_HOME/weka.jar weka.Run \
    .Dl4jMlpFilter \
        -i datasets/nominal/mnist.meta.minimal.arff \
        -o mnist-rn50-concat.arff \
        -c last
        -decimal 20 \
        -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -bs 16" \
        -poolingType AVG \
        -zooModel "weka.dl4j.zoo.Dl4JResNet50" \ 
        -layer-extract "weka.dl4j.layers.DenseLayer -name res4a_branch2b" \
        -layer-extract "weka.dl4j.layers.DenseLayer -name flatten_1" 
         
```
We now have a standard `.arff` file that can be fit to like any numerical dataset
```bash
$ java -cp $WEKA_HOME/weka.jar weka.Run .RandomForest -t mnist-rn50-concat.arff
```

### Java
```java
// Load the dataset
Instances instances = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
instances.setClassIndex(1);
Dl4jMlpFilter myFilter = new Dl4jMlpFilter();

// Load our pretrained model (must be done *before* specifying extra transformation layers)
Dl4JResNet50 zooModel = new Dl4JResNet50();
myFilter.setZooModelType(zooModel);

// Concatenate activations from an intermediate convolution layer
myFilter.addTransformationLayerName("res4a_branch2b");
// Set the pooling type to average
myFilter.setPoolingType(PoolingType.AVG);

// Create our iterator, pointing it to the location of the images
ImageInstanceIterator imgIter = new ImageInstanceIterator();
imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
// Featurize 16 instances at a time
imgIter.setTrainBatchSize(16);
myFilter.setImageInstanceIterator(imgIter);

// Run the filter, using the model as a feature extractor
myFilter.setInputFormat(instances);
Instances transformedInstances = Filter.useFilter(instances, myFilter);

// You could save the instances at this point to an arff file to be used with other classifiers via:
// https://waikato.github.io/weka-wiki/formats_and_processing/save_instances_to_arff/

// CV our Random Forest classifier on the extracted features
Evaluation evaluation = new Evaluation(transformedInstances);
int numFolds = 10;
evaluation.crossValidateModel(new RandomForest(), transformedInstances, numFolds, new Random(1));
System.out.println(evaluation.toSummaryString());
System.out.println(evaluation.toMatrixString());
```

### GUI
The first step is to open the MNIST meta ARFF file in the Weka Explorer `Preprocess` tab via `Open File`. 
A randomly sampled MNIST dataset of 420 images is provided in the WekaDeeplearning4j package for testing purposes 
(`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist.meta.minimal.arff`). 

Then, select the the `Dl4jMlpFilter` in the filter panel. Click in the box to open the filter settings.

![Classifier](../img/gui/featurize-std-filter.png)

To correctly load the images it is further necessary to select the `Image-Instance-Iterator` as `instance iterator` 
and point it to the MNIST directory that contains the actual image files (`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist-minimal/`). 

If you run into memory issues then use a smaller mini-batch size.

![Image Instance Iterator](../img/gui/image-instance-iterator.png)

`Dl4jResNet50` is already selected as the feature extractor model. To add `res4a_branch2b` as another feature extraction layer, we edit the `Feature extraction layers` property.
Click to open the array editor, and click the `DenseLayer` specification to open the editor for our new layer.

![Feature Extraction Layers](../img/gui/featurize-concat-layers.png)

When adding another feature extraction layer, only the `layer name` property needs to be set. Click `OK` and `Add` to add the newly created layer:

![Feature Extraction Layers](../img/gui/featurize-concat-layers2.png)

Note that the order of feature extraction layers may have an effect on accuracy obtained by a WEKA classifier.
Set the `Pooling Type` property to `AVG`, and click `Ok` and `Apply` to begin processing your dataset. 
After completion, you should see your newly processed dataset! 
The attributes are named after the layer they were derived from, so more investigation can be done around which layer provides the most informative features (e.g., using the `Select Attributes` panel in WEKA).

![Processed Dataset](../img/gui/featurize-concat-processed.png)

Simply switch to the `Classify` tab to start applying different WEKA classifiers (this example uses `RandomForest`) to the newly transformed dataset.  

### Results
Unfortunately, adding this extra layer lowered the accuracy (perhaps adding too many unneccessary features). 
Try playing around with some other layers/classifiers to try improve the accuracy.

```text
Correctly Classified Instances         327               77.8571 %
Incorrectly Classified Instances        93               22.1429 %
Kappa statistic                          0.7538
Mean absolute error                      0.114 
Root mean squared error                  0.2148
Relative absolute error                 63.3374 %
Root relative squared error             71.5949 %
Total Number of Instances              420    
```