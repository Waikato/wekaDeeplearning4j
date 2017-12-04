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
 *    DefaultInstanceIterator.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.dl4j.iterators.instance;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.UUID;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.InFileAndMemoryDataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InFileDataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.core.OptionMetadata;
import weka.dl4j.iterators.dataset.TextEmbeddingDataSetIterator;

/**
 * Converts the given Instances object into a DataSet and then constructs and returns a
 * TextInstanceIterator.
 *
 * <p>Assumes the instance object is of the following structure:
 *
 * <ul>
 *   <li>Attribute 0: text (e.g. a elementwise document)
 *   <li>Attribute 1: class
 * </ul>
 *
 * @author Steven Lang
 */
@Slf4j
public class TextInstanceIterator extends AbstractInstanceIterator {

  /** The ID used to serialize this class */
  private static final long serialVersionUID = 1316260988724548474L;

  /** Word vector file location */
  protected File wordVectorLocation = new File(System.getProperty("user.dir"));

  /** Loaded word vectors */
  protected transient WordVectors wordVectors;

  /** Truncation length (maximum number of tokens per document) */
  protected int truncateLength = 100;

  @Override
  public void validate(Instances data) throws InvalidInputDataException {
    if (!data.attribute(0).isString()) {
      throw new InvalidInputDataException("The first attribute has to be the document.");
    }
    if (data.classIndex() != 1) {
      throw new InvalidInputDataException(
          "The class index must be 1. Class in the given data: " + data.classIndex());
    }
    if (data.numAttributes() != 2) {
      throw new InvalidInputDataException(
          "There must be exactly two attributes: 1) Text 2) Label. "
              + "The given data consists of "
              + data.numAttributes()
              + " attributes.");
    }
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
  public DataSetIterator getDataSetIterator(Instances data, int seed, int batchSize)
      throws InvalidInputDataException, IOException {
    validate(data);
    initWordVectors();
    return new TextEmbeddingDataSetIterator(
        data, wordVectors, batchSize, truncateLength);
  }

  @OptionMetadata(
    displayName = "location of word vectors",
    description = "The word vectors location.",
    commandLineParamName = "wordVectorLocation",
    commandLineParamSynopsis = "-wordVectorLocation <string>",
    displayOrder = 1
  )
  public File getWordVectorLocation() {
    return wordVectorLocation;
  }

  public void setWordVectorLocation(File file) {
    this.wordVectorLocation = file;
    initWordVectors();
  }

  /** Initialize the word vectors from the given file */
  protected void initWordVectors() {
    if (wordVectors == null) {
      log.debug("Loading word vector model");
      wordVectors = WordVectorSerializer.loadStaticModel(wordVectorLocation);
    }
  }

  @OptionMetadata(
    displayName = "truncation length",
    description = "The maximum number of tokens per document (default = 100).",
    commandLineParamName = "truncationLength",
    commandLineParamSynopsis = "-truncationLength <int>",
    displayOrder = 2
  )
  public int getTruncateLength() {
    return truncateLength;
  }

  public void setTruncateLength(int truncateLength) {
    this.truncateLength = truncateLength;
  }

  public String globalInfo() {
    return "Instance iterator reads rows from the given ARFF file. This "
        + "iterator is not compatible with convolution layers. See also: "
        + "ConvolutionInstanceIterator/ImageInstanceIterator.";
  }
}
