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
 * <p>Assumes the instance object is of the following structure:
 *
 * <ul>
 *   <li>Attribute 0: path to text file
 *   <li>Attribute 1: class
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
    return new TextFilesEmbeddingDataSetIterator(
        data, wordVectors, batchSize, truncateLength, textsLocation);
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
    if (!(data.attribute(0).isString() && data.classIndex() == 1)) {
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
}
