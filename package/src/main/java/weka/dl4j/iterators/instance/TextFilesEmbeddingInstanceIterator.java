package weka.dl4j.iterators.instance;

import java.io.File;
import java.io.IOException;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.core.OptionMetadata;
import weka.dl4j.iterators.dataset.TextFilesEmbeddingDataSetIterator;

/**
 * Converts the given Instances object into a DataSet and then constructs and returns a
 * TextEmbeddingInstanceIterator.
 *
 * <p>Assumes the instance object has the following two attributes:
 *
 * <ul>
 *   <li>Path to text file
 *   <li>Class
 * </ul>
 *
 * @author Steven Lang
 */
public class TextFilesEmbeddingInstanceIterator extends TextEmbeddingInstanceIterator {

  private static final long serialVersionUID = -1065956690877737854L;
  private File textsLocation = new File(System.getProperty("user.dir"));

  @Override
  public DataSetIterator getDataSetIterator(Instances data, int seed, int batchSize)
      throws InvalidInputDataException, IOException {
    validate(data);
    initWordVectors();
    return new TextFilesEmbeddingDataSetIterator(data, wordVectors, tokenizerFactory, tokenPreProcess, stopwords, batchSize, truncateLength, textsLocation);
  }

  /**
   * Validates the input dataset
   *
   * @param data the input dataset
   * @throws InvalidInputDataException if validation is unsuccessful
   */
  public void validate(Instances data) throws InvalidInputDataException {

    if (!getTextsLocation().isDirectory()) {
      throw new InvalidInputDataException("Directory not valid: " + getTextsLocation());
    }
    if (!((data.attribute(0).isString() && data.classIndex() == 1)
          || (data.attribute(1).isString() && data.classIndex() == 0))) {
      throw new InvalidInputDataException(
          "An ARFF is required with a string attribute and a class attribute");
    }
  }

  @OptionMetadata(
    displayName = "directory of text files",
    description = "The directory containing the text files (default = user home).",
    commandLineParamName = "textsLocation",
    commandLineParamSynopsis = "-textsLocation <string>",
    displayOrder = 3
  )
  public File getTextsLocation() {
    return textsLocation;
  }

  public void setTextsLocation(File textsLocation) {
    this.textsLocation = textsLocation;
  }

  public String globalInfo() {
    return "Text iterator that reads documents from each file that is listed in a meta arff file. "
        + "Each document is then "
        + "processed by the tokenization, stopwords, token-preprocessing and afterwards mapped into "
        + "an embedding space with the given word-vector model.";
  }
}
