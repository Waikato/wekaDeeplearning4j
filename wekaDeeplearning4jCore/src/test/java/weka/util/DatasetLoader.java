package weka.util;

import weka.core.Instances;
import weka.dl4j.iterators.ImageDataSetIterator;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileReader;

public class DatasetLoader {

    /**
     * Number of classes in the iris dataset
     */
    public static final int NUM_CLASSES_IRIS = 3;

    /**
     * Number of classes in the mnist dataset
     */
    public static final int NUM_CLASSES_MNIST = 10;

    /**
     * Number of classes in the diabetes dataset
     */
    public static final int NUM_CLASSES_DIABETES = 1;

    /**
     * Number of instances in the iris dataset
     */
    public static final int NUM_INSTANCES_IRIS = 150;

    /**
     * Number of instances in the mnist dataset
     */
    public static final int NUM_INSTANCES_MNIST = 420;

    /**
     * Number of instances in the diabetes dataset
     */
    public static final int NUM_INSTANCES_DIABETES = 43;


    /**
     * Load the mnist minimal dataset with an ImageDataSetIterator
     *
     * @return ImageDataSetIterator
     */
    public static ImageDataSetIterator loadMiniMnistImageIterator() {
        return loadMnistImageIterator("datasets/nominal/mnist-minimal");
    }

    /**
     * Load the mnist minimal dataset with an ImageDataSetIterator
     *
     * @return ImageDataSetIterator
     */
    public static ImageDataSetIterator loadMediumMnistImageIterator() {
        return loadMnistImageIterator("datasets/nominal/mnist-minimal-2101");
    }

    /**
     * Load the mnist minimal dataset with an ImageDataSetIterator
     *
     * @return ImageDataSetIterator
     */
    public static ImageDataSetIterator loadMnistImageIterator(String path) {
        ImageDataSetIterator imgIter = new ImageDataSetIterator();
        imgIter.setImagesLocation(new File(path));
        final int height = 28;
        final int width = 28;
        final int channels = 1;
        imgIter.setHeight(height);
        imgIter.setWidth(width);
        imgIter.setNumChannels(channels);
        return imgIter;
    }

    /**
     * Load the iris arff file
     *
     * @return Iris data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadIris() throws Exception {
        Instances data = new Instances(new FileReader("datasets/nominal/iris.arff"));
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    /**
     * Load the diabetes arff file
     *
     * @return Diabetes data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadDiabetes() throws Exception {
        Instances data = new Instances(new FileReader("datasets/numeric/diabetes_numeric.arff"));
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    /**
     * Load the mnist minimal meta arff file
     *
     * @return Mnist minimal meta data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadMiniMnistMeta() throws Exception {
        return loadMetaData("datasets/nominal/mnist.meta.minimal.arff");
    }

    /**
     * Load the mnist medium meta arff file
     *
     * @return Mnist minimal meta data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadMediumMnistMeta() throws Exception {
        return loadMetaData("datasets/nominal/mnist.meta.minimal.2101.arff");
    }

    /**
     * Load the mnist minimal meta arff file
     *
     * @return Mnist minimal meta data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadMetaData(String path) throws Exception {
        Instances data = new Instances(new FileReader(path));
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }


}
