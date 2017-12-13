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
package weka.dl4j.iterators.instance.sequence.text;

import java.io.IOException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.dl4j.iterators.dataset.sequence.text.TextEmbeddingDataSetIterator;

/**
 * Converts the given Instances object into a DataSet and then constructs and returns a
 * TextEmbeddingInstanceIterator.
 *
 * <p>Assumes the instance with the following attributes:
 *
 * <ul>
 *   <li>Text (e.g. a elementwise document)
 *   <li>Class
 * </ul>
 *
 * @author Steven Lang
 */
@Slf4j
public class TextEmbeddingInstanceIterator extends AbstractTextEmbeddingIterator {

  /** The ID used to serialize this class */
  private static final long serialVersionUID = 1316260988724548474L;

  @Override
  public void validate(Instances data) throws InvalidInputDataException {
    if (!((data.attribute(0).isString() && data.classIndex() == 1)
        || (data.attribute(1).isString() && data.classIndex() == 0))) {
      throw new InvalidInputDataException(
          "An ARFF is required with a string attribute and a class attribute");
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
    final LabeledSentenceProvider prov = getSentenceProvider(data);
    return new TextEmbeddingDataSetIterator(
        data,
        wordVectors,
        tokenizerFactory,
        tokenPreProcess,
        stopwords,
        prov,
        batchSize,
        truncateLength);
  }

  @Override
  public void initialize() {
    super.initialize();
  }

  public String globalInfo() {
    return "Text iterator that reads documents line wise from an ARFF file. Each document is then "
        + "processed by the tokenization, stopwords, token-preprocessing and afterwards mapped into "
        + "an embedding space with the given word-vector model.";
  }
}
