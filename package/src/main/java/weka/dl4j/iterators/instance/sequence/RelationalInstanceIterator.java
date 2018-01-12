package weka.dl4j.iterators.instance.sequence;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.core.OptionMetadata;
import weka.dl4j.iterators.dataset.sequence.RelationalDataSetIterator;
/**
 * Converts the given Instances containing a single relational attribute into a DataSet.
 *
 * @author Steven Lang
 */
public class RelationalInstanceIterator extends AbstractSequenceInstanceIterator {

  private static final long serialVersionUID = 3713170652162476693L;
  /** Maximum sequence length */
  protected int truncateLength = 100;
  /** Index of the relational attribute */
  protected int relationalAttributeIndex = 0;

  @Override
  public DataSetIterator getDataSetIterator(Instances data, int seed, int batchSize)
      throws Exception {
    validate(data);
    return new RelationalDataSetIterator(data, batchSize, truncateLength, relationalAttributeIndex);
  }

  @Override
  public void validate(Instances data) throws InvalidInputDataException {
    if (!data.attribute(relationalAttributeIndex).isRelationValued()){
      throw new InvalidInputDataException("The attribute at index <"
          + relationalAttributeIndex + " is not relational.");
    }
  }

  @OptionMetadata(
    displayName = "truncation length",
    description = "The maximum number of instances per row (default = 100).",
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

  @OptionMetadata(
    displayName = "relational attribute index",
    description = "Set the relational attribute's index (default = 0)",
    commandLineParamName = "relationalAttributeIndex",
    commandLineParamSynopsis = "-relationalAttributeIndex <int>",
    displayOrder = 2
  )
  public int getRelationalAttributeIndex() {
    return relationalAttributeIndex;
  }

  public void setRelationalAttributeIndex(int relationalAttributeIndex) {
    this.relationalAttributeIndex = relationalAttributeIndex;
  }
}
