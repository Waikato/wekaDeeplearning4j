package weka.dl4j.iterators.dataset.sequence.text;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.DataSet;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.stopwords.AbstractStopwords;

/**
 * A DataSetIterator implementation that reads text documents from text files in a given directory
 * and translates each document to a sequence of wordvectors, given a wordvector model.
 *
 * @author Steven Lang
 */
@Slf4j
public class TextFilesEmbeddingDataSetIterator extends TextEmbeddingDataSetIterator {

  private static final long serialVersionUID = 2569158554412509023L;
  /** Location of the text files */
  private File textsLocation;

  /**
   * @param data Instances with documents and labels
   * @param wordVectors WordVectors object
   * @param tokenFact Tokenizer factory
   * @param tpp Token pre processor
   * @param stopWords Stop word object
   * @param batchSize Size of each minibatch for training
   * @param truncateLength If reviews exceed
   * @param textsLocation Location of the text files
   */
  public TextFilesEmbeddingDataSetIterator(
      Instances data,
      WordVectors wordVectors,
      TokenizerFactory tokenFact,
      TokenPreProcess tpp,
      AbstractStopwords stopWords,
      int batchSize,
      int truncateLength,
      File textsLocation)
      throws IOException {
    super(data, wordVectors, tokenFact, tpp, stopWords, batchSize, truncateLength);
    this.textsLocation = textsLocation;
  }

  @Override
  public DataSet next(int num) {
    Instances copy = new Instances(data, num);

    final int clsIndex = data.classIndex();

    for (int i = cursor; i < cursor + num && i < totalExamples(); i++) {
      final Instance row = data.get(i);

      final String pathname = row.stringValue(1 - clsIndex);
      try {
        String content =
            FileUtils.readFileToString(
                Paths.get(textsLocation.getAbsolutePath(), pathname).toFile());
        Instance newRow = new DenseInstance(2);
        newRow.setDataset(copy);
        newRow.setClassValue(row.classValue());
        newRow.setValue(1 - clsIndex, content);
        copy.add(newRow);
      } catch (IOException e) {
        final String s = "Could not read content of file \"" + pathname + "\"";
        throw new RuntimeException(s, e);
      }
    }
    cursor+=num;
    return super.nextDataSet(copy);
  }
}
