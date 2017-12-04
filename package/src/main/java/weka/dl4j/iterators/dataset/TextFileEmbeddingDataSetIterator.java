package weka.dl4j.iterators.dataset;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.dataset.DataSet;
import weka.core.Instances;

/**
 * A DataSetIterator implementation that reads text documents from text files in a given directory
 * and translates each document to a sequence of wordvectors, given a wordvector model.
 *
 * @author Steven Lang
 */
@Slf4j
public class TextFileEmbeddingDataSetIterator extends TextEmbeddingDataSetIterator {

  private static final long serialVersionUID = 2569158554412509023L;
  /** Location of the text files */
  private File textsLocation;
  /** Collect already loaded documents */
  private Set<Integer> alreadyLoaded;

  /**
   * @param data Instances with documents and labels
   * @param wordVectors WordVectors object
   * @param batchSize Size of each minibatch for training
   * @param truncateLength If reviews exceed
   * @param textsLocation Location of the text files
   */
  public TextFileEmbeddingDataSetIterator(Instances data,
      WordVectors wordVectors, int batchSize, int truncateLength, File textsLocation) throws IOException {
    super(new Instances(data), wordVectors, batchSize, truncateLength);
    this.textsLocation = textsLocation;
    this.alreadyLoaded = new HashSet<>();
  }

  @Override
  public DataSet next(int num) {
    for(int i = cursor; i < cursor + num && i < totalExamples(); i++){
      final String pathname = data.get(i).stringValue(0);
      try {
        if (!alreadyLoaded.contains(i)){
          String content = FileUtils.readFileToString(Paths.get(textsLocation.getAbsolutePath(), pathname).toFile());
          data.get(i).setValue(0, content);
          alreadyLoaded.add(i);
        }
      } catch (IOException e) {
        final String s = "Could not read content of file \"" + pathname + "\"";
        throw new RuntimeException(s, e);
      }
    }

    return super.next(num);
  }
}
