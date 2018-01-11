package weka.dl4j.iterators.instance.sequence.text.cnn;

import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.dl4j.iterators.dataset.sequence.text.cnn.CnnSentenceDataSetIterator;
import weka.dl4j.iterators.instance.sequence.text.AbstractTextEmbeddingIterator;

/**
 * Iterator that constructs datasets from text data for convolutional networks.
 *
 * @author Steven Lang
 */
public class CnnTextEmbeddingInstanceIterator extends AbstractTextEmbeddingIterator {

  private static final long serialVersionUID = 3417451906101970927L;

  @Override
  public DataSetIterator getDataSetIterator(Instances data, int seed, int batchSize) {
    initialize();
    LabeledSentenceProvider clsp = getSentenceProvider(data);
    return new CnnSentenceDataSetIterator.Builder()
        .wordVectors(wordVectors)
        .tokenizerFactory(tokenizerFactory)
        .sentenceProvider(clsp)
        .minibatchSize(batchSize)
        .maxSentenceLength(truncateLength)
        .useNormalizedWordVectors(false)
        .sentencesAlongHeight(true)
        .stopwords(stopwords)
        .build();
  }

  @Override
  public void validate(Instances data) throws InvalidInputDataException {
    if (!getWordVectorLocation().isFile()) {
      throw new InvalidInputDataException("File not valid: " + getWordVectorLocation());
    }
  }


}
