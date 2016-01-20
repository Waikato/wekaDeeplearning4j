# wekaDeeplearning4j
DL4J wrapper for WEKA. Original code written by Mark Hall. This package currently introduces a new classifier,
`Dl4jMlpClassifier`, which allows arbitrary-depth MLPs to be built with a degree of flexibility (e.g. type of weight initialisation,
loss function, gradient descent algorithm, etc.).

<img src="https://raw.githubusercontent.com/christopher-beckham/wekaDeeplearning4j/master/images/gui.png" alt="img" width="700" />

Not many tests have been written for this classifier yet, so expect it to be quite buggy!

## Installation
Simply run the `build.sh` script in the core directory. This assumes:
* You have Ant and Maven installed.
* WEKA's `weka.jar` file resides somewhere in your Java classpath. The latest and greatest WEKA installation is highly recommended; you
  can get the .jar of the nightly snapshot [here](http://www.cs.waikato.ac.nz/~ml/weka/snapshots/developer-branch.zip).

## Usage

An example script is provided that can be run on the Iris dataset in the `scripts` directory.

```
java -Xmx5g -cp $WEKA_HOME/weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 0 \
     -layer "weka.dl4j.layers.DenseLayer -units 10 -activation tanh -init XAVIER" \
     -layer "weka.dl4j.layers.OutputLayer -units 3 -activation softmax -init XAVIER -loss MCXENT" \
     -iters 100 \
     -optim STOCHASTIC_GRADIENT_DESCENT \
     -updater NESTEROVS \
     -lr 0.1 \
     -momentum 0.9 \
     -bs 1 \
     -t ../datasets/iris.arff \
     -no-cv
```

This trains a one-hidden-layer MLP with 10 units on the Iris dataset. Nesterov momentum is used in conjunction with SGD and the initial
learning rate and momentum is set to 0.1 and 0.9, respectively. The network is trained for 100 iterations.

## Caveats

* Make sure that the last layer in the network is an output layer. Somewhat confusingly, you can also specify the number 
of units in this layer -- this has no effect and will internally change itself to the number of classes in the dataset.
* The training batch size parameter (`-bs`) currently does not work -- full batch gradient descent is always used.
