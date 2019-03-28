
package weka.classifiers.functions.dl4j;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.*;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Utility routines for the Dl4jMlpClassifier
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 */
public class Utils {

  /** Logger instance */
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
        next = iter.next();
      } else {
        // TODO: figure out which batch size is feasible for inference
        final int batch = iter.batch() * 8;
        next = iter.next(batch);
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
   * Convert an arbitrary NDArray to Weka instances
   * @param ndArray Input array
   * @return Instances object
   * @throws WekaException Invalid input
   */
  public static Instances ndArrayToInstances(INDArray ndArray) throws WekaException {
    int batchsize = (int) ndArray.size(0);
    long[] shape = ndArray.shape();
    int dims = shape.length;
    if (dims < 2){
      throw new WekaException("Invalid input, NDArray shape needs to be at least two dimensional "
          + "but was " + Arrays.toString(shape));
    }

    long prod = Arrays.stream(shape).reduce(1, (left, right) -> left * right);
    prod = prod/ batchsize;

    ArrayList<Attribute> atts = new ArrayList<>();
    for (int i = 0; i < prod; i++) {
      atts.add(new Attribute("transformedAttribute" + i));
    }
    Instances instances = new Instances("Transformed", atts, batchsize);
    for (int i = 0; i < batchsize; i++) {
      INDArray row = ndArray.getRow(i);
      INDArray flattenedRow = Nd4j.toFlattened(row);
      Instance inst = new DenseInstance(atts.size());
      for (int j = 0; j < flattenedRow.size(1); j++) {
        inst.setValue(j, flattenedRow.getDouble(j));
      }
      inst.setDataset(instances);
      instances.add(inst);
    }

    return instances;
  }

  /**
   * Access private field of a given object.
   *
   * @param obj       Object to be accessed
   * @param fieldName Field name
   * @param <T>       Return type
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
      throw new RuntimeException("Could not access private field " + fieldName + " of " +
              "CnnSentenceDataSetIterator");
    }
  }

  /**
   * Set private field of a given object.
   *
   * @param obj       Object to be accessed
   * @param fieldName Field name
   * @param value     Field value to be set
   * @param <T>       Field type
   */
  public static <T> void setFieldValue(Object obj, String fieldName, T value) {
    try {
      Field f = obj.getClass().getSuperclass().getDeclaredField(fieldName);
      f.setAccessible(true);
      f.set(obj, value);
    } catch (NoSuchFieldException | IllegalAccessException e) {
      e.printStackTrace();
      throw new RuntimeException("Could not access private field " + fieldName + " of " +
              "CnnSentenceDataSetIterator");
    }
  }

  /**
   * Invoke a method on a given object.
   *
   * @param obj        Object to be referenced
   * @param methodName Method name which is to be invoked
   * @param args       Method arguments
   * @param <T>        Return type
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
     * @param clz   Class to use the classloader from
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
}
