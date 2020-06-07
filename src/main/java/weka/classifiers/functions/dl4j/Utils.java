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
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.dl4j.PoolingType;

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

  public static Attribute copyNominalAttribute(Attribute oldAttribute) {
    String[] classValues = new String[oldAttribute.numValues()];
    for (int classValI = 0; classValI < oldAttribute.numValues(); classValI++) {
      classValues[classValI] = oldAttribute.value(classValI);
    }
    return new Attribute(oldAttribute.name(), Arrays.asList(classValues));
  }

  public static Instances ndArrayToInstances(INDArray ndArray) throws WekaException {
    return ndArrayToInstances(ndArray, null, null);
  }

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


  public static boolean needsReshaping(INDArray activationAtLayer) {
    return activationAtLayer.shape().length != 2;
  }

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

  public static INDArray reshapeActivations(INDArray activationAtLayer, PoolingType poolingType) {
    long[] resultShape = activationAtLayer.shape();

    if (poolingType == PoolingType.NONE) {
      int extraDimension = (int) (resultShape[1] * resultShape[2] * resultShape[3]);
      return activationAtLayer.reshape(new int[] {(int) resultShape[0], extraDimension});
    } else {
      float[][] pooledBatch = new float[(int) resultShape[0]][(int) resultShape[1]];

      for (int batchItem = 0; batchItem < pooledBatch.length; batchItem++) {
        // 3D array e.g. shape = [512, 64, 64]
        INDArray itemActivations = activationAtLayer.get(NDArrayIndex.point(batchItem));
        for (int activationI = 0; activationI < resultShape[1]; activationI++) {
          // 2D array e.g. shape = [64, 64]
          INDArray attributeActivations = itemActivations.get(NDArrayIndex.point(activationI));
          pooledBatch[batchItem][activationI] = poolNDArray(attributeActivations, poolingType);
        }
      }

      return new NDArray(pooledBatch);
    }
  }

  public static INDArray appendClasses(INDArray result, Instances input) {
    NDArray classes = (NDArray) Nd4j.zeros(result.shape()[0], 1);
    for (int i = 0; i < classes.length(); i++) {
      Instance inst = input.instance(i);
      classes.putScalar(i, inst.classValue());
    }
    return Nd4j.concat(1, result, classes);
  }

  public static Instances convertToInstances(INDArray result, Instances input, Map<String, Long> attributesPerLayer) throws Exception {
    if (result == null) {
      return new Instances(input, 0);
    } else {
      return Utils.ndArrayToInstances(result, input, attributesPerLayer);
    }
  }

  public static DataSet getNext(DataSetIterator iter) {
    return iter.next().copy();
  }

  public static DataSet getNext(DataSetIterator iter, int num) {
    return iter.next(num).copy();
  }
}
