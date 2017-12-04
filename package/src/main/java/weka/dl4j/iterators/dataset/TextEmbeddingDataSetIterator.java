package weka.dl4j.iterators.dataset;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import weka.core.Instances;

/**
 * A DataSetIterator implementation that reads text documents from an arff file and translates each
 * document to a sequence of wordvectors, given a wordvector model.
 *
 * @author Steven Lang
 */
@Slf4j
public class TextEmbeddingDataSetIterator implements DataSetIterator, Serializable {

  private static final long serialVersionUID = 1682821361704251554L;
  private final WordVectors wordVectors;
  private final int batchSize;
  private final int vectorSize;
  private final int truncateLength;

  protected int cursor = 0;
  protected final Instances data;
  private final TokenizerFactory tokenizerFactory;

  /**
   * @param data Instances with documents and labels
   * @param wordVectors WordVectors object
   * @param batchSize Size of each minibatch for training
   * @param truncateLength If reviews exceed
   */
  public TextEmbeddingDataSetIterator(
      Instances data, WordVectors wordVectors, int batchSize, int truncateLength)
      throws IOException {
    this.batchSize = batchSize;
    this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    this.data = data;

    this.wordVectors = wordVectors;
    this.truncateLength = truncateLength;

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
  }

  @Override
  public DataSet next(int num) {
    if (cursor >= data.numInstances()) throw new NoSuchElementException();
    try {
      return nextDataSet(num);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private DataSet nextDataSet(int num) throws IOException {
    // First: load reviews to String. Alternate positive and negative reviews
    List<String> reviews = new ArrayList<>(num);
    List<Double> lbls = new ArrayList<>(num);

    for (int i = 0; i < num && cursor < totalExamples(); i++) {
      final String document = data.get(cursor).stringValue(0);
      final double label = data.get(cursor).value(1);
      reviews.add(document);
      lbls.add(label);
      cursor++;
    }

    // Second: tokenize reviews and filter out unknown words
    final int numDocuments = reviews.size();
    List<List<String>> allTokens = new ArrayList<>(numDocuments);
    int maxLength = 0;
    for (String s : reviews) {
      List<String> tokens = tokenizerFactory.create(s).getTokens();
      List<String> tokensFiltered = new ArrayList<>();
      for (String t : tokens) {
        if (wordVectors.hasWord(t)) tokensFiltered.add(t);
      }
      allTokens.add(tokensFiltered);
      maxLength = Math.max(maxLength, tokensFiltered.size());
    }

    // If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
    if (maxLength > truncateLength) maxLength = truncateLength;

    // Create data for training
    // Here: we have reviews.size() examples of varying lengths
    INDArray features = Nd4j.create(new int[] {numDocuments, vectorSize, maxLength}, 'f');
    INDArray labels =
        Nd4j.create(
            new int[] {numDocuments, data.numClasses(), maxLength},
            'f'); // Two labels: positive or negative
    // Because we are dealing with reviews of different lengths and only one output at the final
    // time step: use padding arrays
    // Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is
    // just padding
    INDArray featuresMask = Nd4j.zeros(numDocuments, maxLength);
    INDArray labelsMask = Nd4j.zeros(numDocuments, maxLength);

    /*
     Vectorized version
    */
    for (int i = 0; i < numDocuments; i++) {
      List<String> tokens = allTokens.get(i);

      // Check for empty document
      if (tokens.isEmpty()) {
        continue;
      }

      // Get the sequence length of document (i)
      int lastIdx = Math.min(tokens.size(), maxLength);

      // Get all wordvectors in batch
      final INDArray vectors = wordVectors.getWordVectors(tokens.subList(0, lastIdx)).transpose();

      // Put wordvectors into features array: instead of putting one vector at position (j) we put
      // an array of vectors in the interval of [0, lastIdx)
      features.put(
          new INDArrayIndex[] {
            NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, lastIdx)
          },
          vectors);

      // Assign "1" to each position where a feature is present, that is, in the interval of
      // [0, lastIdx)
      featuresMask
          .get(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.interval(0, lastIdx)})
          .assign(1);

      /*
       Put the labels in the labels and labelsMask arrays
      */

      // Differ between classification and regression task
      if (data.numClasses() == 1) { // Regression
        double val = lbls.get(i);
        labels.putScalar(new int[] {i, 0, lastIdx - 1}, val);
      } else if (data.numClasses() > 1) { // Classification
        // One-Hot-Encoded class
        int idx = lbls.get(i).intValue();
        // Set label
        labels.putScalar(new int[] {i, idx, lastIdx - 1}, 1.0);
      } else {
        throw new RuntimeException("Could not detect classification or regression task.");
      }

      // Specify that an output exists at the final time step for this example
      labelsMask.putScalar(new int[] {i, lastIdx - 1}, 1.0);
    }

    // Cache the dataset
    return new DataSet(features, labels, featuresMask, labelsMask);
  }

  @Override
  public int totalExamples() {
    return data.numInstances();
  }

  @Override
  public int inputColumns() {
    return vectorSize;
  }

  @Override
  public int totalOutcomes() {
    return data.numClasses();
  }

  @Override
  public void reset() {
    cursor = 0;
  }

  public boolean resetSupported() {
    return true;
  }

  @Override
  public boolean asyncSupported() {
    return true;
  }

  @Override
  public int batch() {
    return batchSize;
  }

  @Override
  public int cursor() {
    return cursor;
  }

  @Override
  public int numExamples() {
    return totalExamples();
  }

  @Override
  public void setPreProcessor(DataSetPreProcessor preProcessor) {
    throw new UnsupportedOperationException();
  }

  @Override
  public List<String> getLabels() {
    return IntStream.range(0, data.numClasses())
        .boxed()
        .map(i -> data.classAttribute().value(i))
        .collect(Collectors.toList());
  }

  @Override
  public boolean hasNext() {
    return cursor < numExamples();
  }

  @Override
  public DataSet next() {
    return next(batchSize);
  }

  @Override
  public void remove() {}

  @Override
  public DataSetPreProcessor getPreProcessor() {
    throw new UnsupportedOperationException("Not implemented");
  }
}
