/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Utils.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.functions.dl4j;

import java.util.ArrayList;
import java.util.Arrays;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;

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
   * Converts a set of training instances corresponding to a time series to a DataSet. For RNNs, the
   * input of the network is 3 dimensional of dimensions [numExamples, numInputs, numTimeSteps]
   * Assumes that the instances have been suitably preprocessed - i.e. missing values replaced and
   * nominals converted to binary/numeric. Also assumes that the class index has been set
   *
   * @param insts the instances to convert
   * @return a DataSet
   */
  public static DataSet RNNinstancesToDataSet(Instances insts) {
    INDArray data = Nd4j.ones(1, insts.numAttributes() - 1, insts.numInstances());
    INDArray labels = Nd4j.ones(1, insts.numClasses(), insts.numInstances());
    double[] outcomes = new double[insts.numInstances()];

    for (int i = 0; i < insts.numInstances(); i++) {
      double[] independent = new double[insts.numAttributes() - 1];
      int index = 0;
      Instance current = insts.instance(i);
      for (int j = 0; j < insts.numAttributes(); j++) {
        if (j != insts.classIndex()) {
          independent[index++] = current.value(j);
        } else {
          outcomes[i] = current.classValue();
        }
      }

      INDArray ind = Nd4j.create(independent, new int[] {1, insts.numAttributes() - 1, 1});
      for (int k = 0; k < independent.length; k++) {
        data.putScalar(0, k, i, ind.getDouble(k));
      }
    }
    INDArray outcomesNDArray = Nd4j.create(outcomes, new int[] {1, insts.numInstances(), 1});
    labels.putColumn(0, outcomesNDArray);

    DataSet dataSet = new DataSet(data, labels);
    return dataSet;
  }

  /**
   * Converts an instance to an INDArray. Only the values of the non-class attributes are copied
   * over.
   *
   * @param inst
   * @return the INDArray
   */
  public static INDArray instanceToINDArray(Instance inst) {
    INDArray result = Nd4j.ones(1, inst.numAttributes() - 1);
    double[] independent = new double[inst.numAttributes() - 1];
    int index = 0;
    for (int i = 0; i < inst.numAttributes(); i++) {
      if (i != inst.classIndex()) {
        independent[index++] = inst.value(i);
      }
    }
    result.putRow(0, Nd4j.create(independent));

    return result;
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
}
