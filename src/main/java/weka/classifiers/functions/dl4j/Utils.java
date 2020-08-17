/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Utils.java
 * Copyright (C) 2016-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions.dl4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.*;
import weka.dl4j.PoolingType;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.zoo.AbstractZooModel;

import javax.imageio.ImageIO;

/**
 * Utility routines for the Dl4jMlpClassifier
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 */
public class Utils {

  /**
   * Logger instance
   */
  private static final Logger logger = LoggerFactory.getLogger(Utils.class);

  /**
   * Converts a set of training instances to a DataSet. Assumes that the instances have been
   * suitably preprocessed - i.e. missing values replaced and nominals converted to binary/numeric.
   * Also assumes that the class index has been set
   *
   * @param insts the instances to convert
   * @return a DataSet
   */
  public static DataSet instancesToDataSet(Instances insts) {
    INDArray data = Nd4j.zeros(insts.numInstances(), insts.numAttributes() - 1);
    INDArray outcomes = Nd4j.zeros(insts.numInstances(), insts.numClasses());

    for (int i = 0; i < insts.numInstances(); i++) {
      double[] independent = new double[insts.numAttributes() - 1];
      double[] dependent = new double[insts.numClasses()];
      Instance current = insts.instance(i);
      for (int j = 0; j < current.numValues(); j++) {
        int index = current.index(j);
        double value = current.valueSparse(j);

        if (index < insts.classIndex()) {
          independent[index] = value;
        } else if (index > insts.classIndex()) {
          // Shift by -1, since the class is left out from the feature matrix and put into a separate
          // outcomes matrix
          independent[index - 1] = value;
        }
      }

      // Set class values
      if (insts.numClasses() > 1) { // Classification
        final int oneHotIdx = (int) current.classValue();
        dependent[oneHotIdx] = 1.0;
      } else { // Regression (currently only single class)
        dependent[0] = current.classValue();
      }

      INDArray row = Nd4j.create(independent);
      data.putRow(i, row);
      outcomes.putRow(i, Nd4j.create(dependent));
    }
    return new DataSet(data, outcomes);
  }

  /**
   * Converts a set of training instances to a DataSet prepared for the convolution operation using
   * the height, width and number of channels
   *
   * @param height image height
   * @param width image width
   * @param channels number of image channels
   * @param insts the instances to convert
   * @return a DataSet
   */
  public static DataSet instancesToConvDataSet(
      Instances insts, int height, int width, int channels) {
    DataSet ds = instancesToDataSet(insts);
    INDArray data = Nd4j.zeros(insts.numInstances(), channels, width, height);
    ds.getFeatures();

    for (int i = 0; i < insts.numInstances(); i++) {
      INDArray row = ds.getFeatures().getRow(i);
      row = row.reshape(1, channels, height, width);
      data.putRow(i, row);
    }

    return new DataSet(data, ds.getLabels());
  }

  /**
   * Compute the model score on a given iterator.
   *
   * @param model Model
   * @param iter Iterator
   * @return Model score on iterator data
   */
  public static double computeScore(ComputationGraph model, DataSetIterator iter) {
    double scoreSum = 0;
    int numBatches = 0;

    // Iterate batches
    iter.reset();
    while (iter.hasNext()) {
      DataSet next;
      if (iter instanceof AsyncDataSetIterator
          || iter instanceof CachingDataSetIterator) {
        next = getNext(iter);
      } else {
        // TODO: figure out which batch size is feasible for inference
        final int batch = iter.batch() * 8;
        next = Utils.getNext(iter, batch);
      }
      scoreSum += model.score(next);
      numBatches++;
    }

    // Get average score
    double score = 0;
    if (numBatches != 0) {
      score = scoreSum / numBatches;
    }
    iter.reset();
    return score;
  }

  /**
   * Copies the attribute name and values of a given nominal attribute
   * @param oldAttribute attribute to copy
   * @return duplicated nominal attribute
   */
  public static Attribute copyNominalAttribute(Attribute oldAttribute) {
    String[] classValues = new String[oldAttribute.numValues()];
    for (int classValI = 0; classValI < oldAttribute.numValues(); classValI++) {
      classValues[classValI] = oldAttribute.value(classValI);
    }
    return new Attribute(oldAttribute.name(), Arrays.asList(classValues));
  }

  /**
   * Helper function for getting the new layer name when using the Dl4jMlpFilter
   *
   * The attributes are named after the layer they originated from, so this function
   * counts throught the attributes per layer, comparing it with the given index
   * to determine a) which layer the activation came from and b) which number activation
   * this is
   * @param attributesPerLayer
   * @param i
   * @return
   */
  public static String getAttributeName(Map<String, Long> attributesPerLayer, int i) {
    if (attributesPerLayer == null) {
      return "transformedAttribute" + i;
    }

    int layerSum = 0;
    for (Map.Entry<String, Long> entry : attributesPerLayer.entrySet()) {
      long numAttributesForLayer = entry.getValue();
      if (layerSum + numAttributesForLayer > i) {
        return String.format("%s-%d", entry.getKey(), i - layerSum);
      }
      layerSum += numAttributesForLayer;
    }


    return null;
  }

  /**
   * Convert an arbitrary NDArray to Weka instances
   *
   * @param ndArray Input array
   * @param inputFormat Format to use for the instances
   * @param attributesPerLayer Hashmap of layer names and how many attributes there are per layer
   * @return Instances object
   * @throws WekaException Invalid input
   */
  public static Instances ndArrayToInstances(INDArray ndArray, Instances inputFormat, Map<String, Long> attributesPerLayer) throws WekaException {
    int numInstances = (int) ndArray.size(0);
    long[] shape = ndArray.shape();
    int dims = shape.length;
    if (dims != 2) {
      throw new WekaException("Invalid input, NDArray shape needs to be two dimensional "
          + "but was " + Arrays.toString(shape));
    }

    long numAttributes = shape[1];
    int classI = -1;
    if (inputFormat != null) {
      classI = (int) (numAttributes - 1);
    }

    // Create the new attribute names
    ArrayList<Attribute> atts = new ArrayList<>();
    for (int i = 0; i < numAttributes; i++) {
      if (i == classI && inputFormat != null) {
        if (inputFormat.classAttribute().isNominal())
          atts.add(copyNominalAttribute(inputFormat.classAttribute()));
        else
          atts.add(new Attribute(inputFormat.classAttribute().name()));
      } else {
        atts.add(new Attribute(getAttributeName(attributesPerLayer, i)));
      }
    }

    // Actually create the instances from the values in the given NDArray
    Instances instances = new Instances("Transformed", atts, numInstances);
    instances.setClassIndex(classI);
    for (int i = 0; i < numInstances; i++) {
      INDArray row = ndArray.get(NDArrayIndex.point(i));
      double[] instanceVals = row.toDoubleVector();
      Instance inst = new DenseInstance(1.0, instanceVals);
      inst.setDataset(instances);
      instances.add(inst);
    }

    return instances;
  }

  /**
   * Access private field of a given object.
   *
   * @param obj Object to be accessed
   * @param fieldName Field name
   * @param <T> Return type
   * @return Value of field with name {@code fieldName}
   */
  public static <T> T getFieldValue(Object obj, String fieldName) {
    try {
      Field f = obj.getClass().getSuperclass().getDeclaredField(fieldName);
      f.setAccessible(true);
      T field = (T) f.get(obj);
      return field;
    } catch (NoSuchFieldException | IllegalAccessException e) {
      e.printStackTrace();
      throw new RuntimeException("Could not access private field " + fieldName + " of " + obj.getClass()
          );
    }
  }

  /**
   * Set private field of a given object.
   *
   * @param obj Object to be accessed
   * @param fieldName Field name
   * @param value Field value to be set
   * @param <T> Field type
   */
  public static <T> void setFieldValue(Object obj, String fieldName, T value) {
    try {
      Field f = obj.getClass().getSuperclass().getDeclaredField(fieldName);
      f.setAccessible(true);
      f.set(obj, value);
    } catch (NoSuchFieldException | IllegalAccessException e) {
      e.printStackTrace();
      throw new RuntimeException("Could not access private field " + fieldName + " of " + obj.getClass());
    }
  }

  /**
   * Invoke a method on a given object.
   *
   * @param obj Object to be referenced
   * @param methodName Method name which is to be invoked
   * @param args Method arguments
   * @param <T> Return type
   * @return Method return value
   */
  public static <T> T invokeMethod(Object obj, String methodName, Object... args) {
    try {
      Class<?>[] parameterTypes = new Class[args.length];
      for (int i = 0; i < args.length; i++) {
        parameterTypes[i] = args[i].getClass();
      }

      Class<?> clazz = obj.getClass().getSuperclass();
      Method method = clazz.getDeclaredMethod(methodName, parameterTypes);
      method.setAccessible(true);
      return (T) method.invoke(obj, args);
    } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
      e.printStackTrace();
      throw new RuntimeException("Could not access private method " + methodName + " of " +
          "CnnSentenceDataSetIterator");
    }
  }

  /**
   * Run some code-block using the local class loader from a given class.
   *
   * @param clz Class to use the classloader from
   * @param block Code block to run
   */
  public static void runWithLocalClassloader(Class clz, VoidCallable block) {
    // Obtain the new loader
    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      // Switch to the new loader
      Thread.currentThread().setContextClassLoader(clz.getClassLoader());

      // Call the actual code block
      block.call();
    } finally {
      // Switch back to the old loader
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  /**
   * Determines if the activations need reshaping
   * @param activationAtLayer Activations in question
   * @return true if the activations need reshaping (too high dimensionality)
   */
  public static boolean needsReshaping(INDArray activationAtLayer) {
    return activationAtLayer.shape().length != 2;
  }

  /**
   * Applies the pooling function to the given feature map
   * @param array feature map to pool
   * @param poolingType pooling function to apply
   * @return pooled value
   */
  public static float poolNDArray(INDArray array, PoolingType poolingType) {
    if (poolingType == PoolingType.MAX) {
      return array.maxNumber().floatValue();
    } else if (poolingType == PoolingType.AVG) {
      return array.meanNumber().floatValue();
    } else if (poolingType == PoolingType.SUM) {
      return  array.sumNumber().floatValue();
    } else if (poolingType == PoolingType.MIN) {
      return array.minNumber().floatValue();
    } else {
      throw new IllegalArgumentException(String.format("Pooling type %s not supported, only " +
              "MAX, AVG, SUM, MIN supported", poolingType));
    }
  }

  /**
   * Shape will either be something like [1, 56, 56, 128] or [1, 128, 56, 56]
   * If it's the former then return true
   * @param activations
   * @return true if the activations are in channels-last format
   */
  public static boolean isChannelsLast(INDArray activations) {
    long[] shape = activations.shape();
    long width = shape[2];
    return shape[3] != width;
  }

  /**
   * Reshape the activations, either by pooling or simply multiplying the extra dimensions together
   * @param activationAtLayer 4d activations e.g., [batch_size, 512, 64, 64]
   * @param poolingType Pooling type to use to lower the dimensionality
   * @return 2D activations
   */
  public static INDArray reshapeActivations(INDArray activationAtLayer, PoolingType poolingType) {
    long[] resultShape = activationAtLayer.shape();

    int batchSize = (int) resultShape[0];
    int numFeatureMaps = (int) resultShape[1];
    int featureMapWidth = (int) resultShape[2];
    int featureMapHeight = (int) resultShape[3];

    // Simply multiply all the extra dimensions together if we're using no pooling
    if (poolingType == PoolingType.NONE) {
      int extraDimensions = numFeatureMaps * featureMapWidth * featureMapHeight;
      return activationAtLayer.reshape(new int[] {batchSize, extraDimensions});
    } else {
      // Otherwise, create a pooled batch
      float[][] pooledBatch = new float[batchSize][numFeatureMaps];

      for (int batchItem = 0; batchItem < batchSize; batchItem++) {
        INDArray batchItemActivations = activationAtLayer.get(NDArrayIndex.point(batchItem)); // 3D array e.g. shape = [512, 64, 64]
        for (int featureMapIndex = 0; featureMapIndex < numFeatureMaps; featureMapIndex++) {
          INDArray featureMap = batchItemActivations.get(NDArrayIndex.point(featureMapIndex));  // 2D array e.g. shape = [64, 64]
          pooledBatch[batchItem][featureMapIndex] = poolNDArray(featureMap, poolingType);
        }
      }

      return new NDArray(pooledBatch);
    }
  }

  /**
   * Appends the input Instances classes to the INDArray
   * @param result activations
   * @param input original Instances
   * @return activations with class value appended
   */
  public static INDArray appendClasses(INDArray result, Instances input) {
    INDArray classes = Nd4j.zeros(result.shape()[0], 1);
    for (int i = 0; i < classes.length(); i++) {
      Instance inst = input.instance(i);
      classes.putScalar(i, inst.classValue());
    }
    return Nd4j.concat(1, result, classes);
  }

  /**
   * Converts the newly transformed instances to an Instances object
   * @param result activations generated from feature layers
   * @param input original input Instances
   * @param attributesPerLayer Hashmap stating the feature layers and how many attributes each has
   * @return
   * @throws Exception
   */
  public static Instances convertToInstances(INDArray result, Instances input, Map<String, Long> attributesPerLayer) throws Exception {
    if (result == null) {
      return new Instances(input, 0);
    } else {
      return Utils.ndArrayToInstances(result, input, attributesPerLayer);
    }
  }

  /**
   * Fix for issue with JVM crashing
   * https://github.com/eclipse/deeplearning4j/issues/8976#issuecomment-639946904
   *
   * It is recommended to use this helper function in WekaDeeplearning4j rather than using iter.next() directly.
   * @param iter DatasetIterator to get images from
   * @return Next DataSet
   */
  public static DataSet getNext(DataSetIterator iter) {
    return iter.next().copy();
  }

  /**
   * Fix for issue with JVM crashing
   * https://github.com/eclipse/deeplearning4j/issues/8976#issuecomment-639946904
   *
   * It is recommended to use this helper function in WekaDeeplearning4j rather than using iter.next() directly.
   * @param iter DatasetIterator to get images from
   * @param num Batch size to get
   * @return Next DataSet
   */
  public static DataSet getNext(DataSetIterator iter, int num) {
    return iter.next(num).copy();
  }

  /**
   * Checks whether the path exists - a little tidier than the code it wraps
   * @param path Path to check
   * @return True if the path exists, false otherwise
   */
  public static boolean pathExists(String path) {
    return new File(path).exists();
  }

  /**
   * @param serializedModelFile File to check
   * @return true if the user has selected a file to load the model from
   */
  public static boolean notDefaultFileLocation(File serializedModelFile) {
    // Has the model file location been set to something other than the default
    return !serializedModelFile.getPath().equals(defaultFileLocation());
  }

  /**
   * The default location for a file parameter
   * @return Default file path
   */
  public static String defaultFileLocation() {
    return WekaPackageManager.getPackageHome().getPath();
  }

  /**
   * Tries to load from a saved model file (if it exists), otherwise loads the given zoo model
   * @param serializedModelFile Saved model path
   * @param zooModelType Type of Zoo Model
   * @return Dl4jMlpClassifier with the loaded ComputationGraph
   * @throws WekaException From errors occurring during loading the model file
   */
  public static Dl4jMlpClassifier tryLoadFromFile(File serializedModelFile, AbstractZooModel zooModelType) throws WekaException {
    Dl4jMlpClassifier model;
    if (notDefaultFileLocation(serializedModelFile)) {
      // First try load from the WEKA binary model file
      try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serializedModelFile))) {
        model = (Dl4jMlpClassifier) ois.readObject();
      } catch (Exception e) {
        throw new WekaException("Couldn't load Dl4jMlpClassifier from model file");
      }
    } else {
      if (zooModelType == null) {
        throw new WekaException("No model file supplied nor zoo model specified");
      }
      // If that fails, try loading from selected zoo model (or keras file)
      model = new Dl4jMlpClassifier();
      model.setZooModel(zooModelType);
    }
    model.setFilterMode(true);
    return model;
  }

  /**
   * Load a Dl4jMlpClassifier for use with the given instances and iterator
   * @param data Instances to prime the model with
   * @param serializedModelFile Saved model file
   * @param zooModelType Type of Zoo Model
   * @param instanceIterator Instance iterator to prime the model with
   * @return Dl4jMlpClassifier setup with the instances and iterator
   * @throws Exception From errors occurring during loading the model file, or from intializing from the data
   */
  public static Dl4jMlpClassifier loadModel(Instances data, File serializedModelFile,
                                            AbstractZooModel zooModelType, AbstractInstanceIterator instanceIterator) throws Exception {
    Dl4jMlpClassifier model = tryLoadFromFile(serializedModelFile, zooModelType);

    model.setInstanceIterator(instanceIterator);

    // If we're loading from a previously trained model, we don't need to intialize the classifier again,
    // We do need to, however, if we're loading from a fresh zoo model
    if (!notDefaultFileLocation(serializedModelFile))
      model.initializeClassifier(data);

    return model;
  }

  /**
   * Load a Dl4jMlpClassifier for use in the Playground - no need to supply Instances or InstanceIterators
   * @param serializedModelFile Saved model file
   * @param zooModelType Type of Zoo Model
   * @return Dl4jMlpClassifier ready to be used in the Playground
   * @throws WekaException From errors occurring during loading the model file, or from intializing from the data
   */
  public static Dl4jMlpClassifier loadPlaygroundModel(File serializedModelFile, AbstractZooModel zooModelType) throws WekaException {
    Dl4jMlpClassifier model = tryLoadFromFile(serializedModelFile, zooModelType);

    if (!Utils.notDefaultFileLocation(serializedModelFile))
      model.loadZooModelNoData(2, 1, zooModelType.getShape()[0]);

    return model;
  }

  public static void saveNDArray(INDArray array, String filenamePrefix) {
    BufferedImage img = Utils.imageFromINDArray(array);
    try {
      ImageIO.write(img, "png", new File(filenamePrefix + ".png"));
    } catch (IOException ex) {
      ex.printStackTrace();
    }
  }

  /**
   * Takes an INDArray containing an image loaded using the native image loader
   * libraries associated with DL4J, and converts it into a BufferedImage.
   * The INDArray contains the color values split up across three channels (RGB)
   * and in the integer range 0-255.
   *
   * @param array INDArray containing an image in order [N, C, H, W] or [C, H, W]
   * @return BufferedImage
   */
  public static BufferedImage imageFromINDArray(INDArray array) {
    long[] shape = array.shape();

    boolean is4d = false;

    if (shape.length == 4) {
      is4d = true;
      System.out.println("Map is 4d");
    }

    long height = shape[1];
    long width = shape[2];

    if (is4d) {
      height = shape[2];
      width = shape[3];
    }

    BufferedImage image = new BufferedImage((int) width, (int) height, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        int red, green, blue;

        if (is4d) {
          red = array.getInt(0, 2, y, x);
          green = array.getInt(0, 1, y, x);
          blue = array.getInt(0, 0, y, x);
        } else {
          red = array.getInt(2, y, x);
          green = array.getInt(1, y, x);
          blue = array.getInt(0, y, x);
        }

        //handle out of bounds pixel values
        red = Math.min(red, 255);
        green = Math.min(green, 255);
        blue = Math.min(blue, 255);

        red = Math.max(red, 0);
        green = Math.max(green, 0);
        blue = Math.max(blue, 0);
        image.setRGB(x, y, new Color(red, green, blue).getRGB());
      }
    }
    return image;
  }
}
