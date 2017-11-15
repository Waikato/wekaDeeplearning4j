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
 *    ImageDatasetIterator.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.dl4j.iterators.instance;

import org.datavec.api.split.CollectionInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.dl4j.ArffMetaDataLabelGenerator;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.InvalidObjectException;

/**
 * An iterator that loads images.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 * @author Steven Lang
 */
public class ImageInstanceIterator extends AbstractInstanceIterator implements Convolutional {

  /** The version ID used for serializing objects of this class */
  private static final long serialVersionUID = -3701309032945158130L;

  /** The desired output height */
  protected int height = 28;

  /** The desired output width */
  protected int width = 28;

  /** The desired number of channels */
  protected int numChannels = 1;

  /** The location of the folder containing the images */
  protected File imagesLocation = new File(System.getProperty("user.dir"));

  public int getTrainBatchSize() {
    return batchSize;
  }

  @OptionMetadata(
    displayName = "size of mini batch",
    description = "The mini batch size to use in the iterator (default = 1).",
    commandLineParamName = "bs",
    commandLineParamSynopsis = "-bs <int>",
    displayOrder = 0
  )
  public void setTrainBatchSize(int trainBatchSize) {
    batchSize = trainBatchSize;
  }

  @OptionMetadata(
    displayName = "directory of images",
    description = "The directory containing the images (default = user home).",
    commandLineParamName = "imagesLocation",
    commandLineParamSynopsis = "-imagesLocation <string>",
    displayOrder = 1
  )
  public File getImagesLocation() {
    return imagesLocation;
  }

  public void setImagesLocation(File imagesLocation) {
    this.imagesLocation = imagesLocation;
  }

  @OptionMetadata(
    displayName = "desired width",
    description = "The desired width of the images (default = 28).",
    commandLineParamName = "width",
    commandLineParamSynopsis = "-width <int>",
    displayOrder = 2
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
    displayOrder = 3
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
    displayOrder = 4
  )
  public int getNumChannels() {
    return numChannels;
  }

  public void setNumChannels(int numChannels) {
    this.numChannels = numChannels;
  }

  /**
   * Validates the input dataset
   *
   * @param data the input dataset
   * @throws Exception if validation is unsuccessful
   */
  public void validate(Instances data) throws Exception {

    if (!getImagesLocation().isDirectory()) {
      throw new FileNotFoundException("Directory not valid: " + getImagesLocation());
    }
    if (!(data.attribute(0).isString() && data.classIndex() == 1)) {
      throw new InvalidObjectException(
          "An ARFF is required with a string attribute and a class attribute");
    }
  }

  /**
   * This just returns the number of channels.
   *
   * @param data the dataset to compute the number of attributes from
   * @return the number of channels
   */
  @Override
  public int getNumAttributes(Instances data) {
    return getNumChannels();
  }

  /**
   * Returns the image recorder.
   *
   * @param data the dataset to use
   * @return the image recorder
   * @throws Exception
   */
  protected ImageRecordReader getImageRecordReader(Instances data) throws Exception {
    ArffMetaDataLabelGenerator labelGenerator =
        new ArffMetaDataLabelGenerator(data, getImagesLocation().toString());
    ImageRecordReader reader =
        new ImageRecordReader(getHeight(), getWidth(), getNumChannels(), labelGenerator);
    CollectionInputSplit cis = new CollectionInputSplit(labelGenerator.getPathURIs());
    reader.initialize(cis);

    return reader;
  }

  /**
   * This method returns the iterator. Scales all intensity values: it divides them by 255.
   *
   * @param data the dataset to use
   * @param seed the seed for the random number generator
   * @param batchSize the batch size to use
   * @return the iterator
   * @throws Exception
   */
  @Override
  public DataSetIterator getDataSetIterator(Instances data, int seed, int batchSize)
      throws Exception {

    batchSize = Math.min(data.numInstances(), batchSize);
    validate(data);
    ImageRecordReader reader = getImageRecordReader(data);

    final int labelIndex = 1; // Use explicit label index position
    final int numPossibleLabels = data.numClasses();
    DataSetIterator tmpIter =
        new RecordReaderDataSetIterator(reader, batchSize, labelIndex, numPossibleLabels);
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(tmpIter);
    tmpIter.setPreProcessor(scaler);
    return tmpIter;
  }

  public String globalInfo() {
    return "Instance iterator that reads images based on the meta-data "
        + "given in the ARFF file.";
  }
}
