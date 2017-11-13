package weka.util;

import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;

import java.io.File;
import java.io.FileReader;


/**
 * Utility class for loading datasets in JUnit tests
 *
 * @author Steven Lang
 */

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
     * Load the mnist minimal dataset with an ImageInstanceIterator
     *
     * @return ImageInstanceIterator
     */
    public static ImageInstanceIterator loadMiniMnistImageIterator() {
        return loadMnistImageIterator("src/test/resources/nominal/mnist-minimal");
    }


    /**
     * Load the mnist minimal dataset with an ImageInstanceIterator
     *
     * @return ImageInstanceIterator
     */
    public static ImageInstanceIterator loadMnistImageIterator(String path) {
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
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
        Instances data = new Instances(new FileReader("src/test/resources/nominal/iris.arff"));
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    /**
     * Load the iris arff file with missing values
     *
     * @return Iris data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadIrisMissingValues() throws Exception {
        Instances data = new Instances(new FileReader("src/test/resources/nominal/iris-missing-values.arff"));
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
        Instances data = new Instances(new FileReader("src/test/resources/numeric/diabetes_numeric.arff"));
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
        return loadArff("src/test/resources/nominal/mnist.meta.minimal.arff");
    }

    /**
     * Load the mnist minimal arff file
     *
     * @return Mnist minimal arff data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadMiniMnistArff() throws Exception {
        return loadArff("src/test/resources/nominal/mnist_784_train_minimal.arff");
    }

    /**
     * Load the wine_date arff file
     *
     * @return Wine date data
     * @throws Exception IO error.
     */
    public static Instances loadWineDate() throws Exception {
        return loadArff("src/test/resources/date/wine_date.arff");
    }

    /**
     * Load the fishcatch arff file
     *
     * @return Fish catch data
     * @throws Exception IO error.
     */
    public static Instances loadFishCatch() throws Exception {
        return loadArff("src/test/resources/numeric/fishcatch.arff");
    }


    /**
     * Load the mnist minimal meta arff file
     *
     * @return Mnist minimal meta data as Instances
     * @throws Exception IO error.
     */
    public static Instances loadArff(String path) throws Exception {
        Instances data = new Instances(new FileReader(path));
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }


}
