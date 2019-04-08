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
 * ConvolutionInstanceIterator.java
 * Copyright (C) 2016-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.iterators.instance;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.dl4j.iterators.dataset.DefaultDataSetIterator;

import java.util.Enumeration;
import weka.dl4j.iterators.instance.api.ConvolutionalIterator;

/**
 * Converts the given Instances object into a DataSet and then constructs and returns a
 * DefaultDataSetIterator. This iterator is designed for training convolutional networks on data
 * that is represented as standard WEKA instances. It enables specification of filter width and
 * height, and number of channels.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 * @author Steven Lang
 */
public class ConvolutionInstanceIterator extends DefaultInstanceIterator implements
        ConvolutionalIterator {

  /** The version ID used for serializing objects of this class */
  private static final long serialVersionUID = -3101209034945158130L;

  /** The desired output height */
  protected int height = 28;

  /** The desired output width */
  protected int width = 28;

  /** The desired number of channels */
  protected int numChannels = 1;

  @OptionMetadata(
    displayName = "desired width",
    description = "The desired width of the images (default = 28).",
    commandLineParamName = "width",
    commandLineParamSynopsis = "-width <int>",
    displayOrder = 1
  )
  public int getWidth() {
    return width;
  }

  public void setWidth(int width) {
    this.width = width;
  }

  @OptionMetadata(
    displayName = "desired height",
    description = "The desired height of the images (default = 28).",
    commandLineParamName = "height",
    commandLineParamSynopsis = "-height <int>",
    displayOrder = 2
  )
  public int getHeight() {
    return height;
  }

  public void setHeight(int height) {
    this.height = height;
  }

  @OptionMetadata(
    displayName = "desired number of channels",
    description = "The desired number of channels (default = 1).",
    commandLineParamName = "numChannels",
    commandLineParamSynopsis = "-numChannels <int>",
    displayOrder = 3
  )
  public int getNumChannels() {
    return numChannels;
  }

  public void setNumChannels(int numChannels) {
    this.numChannels = numChannels;
  }


  /**
   * Returns the actual iterator.
   *
   * @param data the dataset to use
   * @param seed the seed for the random number generator
   * @param batchSize the batch size to use
   * @return the DataSetIterator
   */
  @Override
  public DataSetIterator getDataSetIterator(Instances data, int seed, int batchSize) {
    // Convert Instances to DataSet
    DataSet dataset = Utils.instancesToConvDataSet(data, getHeight(), getWidth(), getNumChannels());

    return new DefaultDataSetIterator(dataset, batchSize);
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClassHierarchy(this.getClass(),super.getClass()).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    return Option.getOptionsForHierarchy(this, super.getClass());
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    Option.setOptionsForHierarchy(options, this, super.getClass());
  }

  public String globalInfo() {
    return "Instance iterator that reads flattened matrices represented as "
        + "column-wise formatted vectors in the ARFF dataset and transforms them into the shape "
        + "(height x width x numChannels). It is necessary, that the "
        + "height*width*numChannels is equal to the number of attributes "
        + "in the ARFF file (excluding the class attribute).";
  }
}
