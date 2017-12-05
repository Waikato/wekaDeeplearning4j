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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.UUID;
import java.util.zip.GZIPInputStream;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.core.OptionMetadata;
import weka.core.converters.CSVSaver;
import weka.dl4j.iterators.dataset.TextEmbeddingDataSetIterator;
import weka.dl4j.text.stopwords.Dl4jAbstractStopwords;
import weka.dl4j.text.stopwords.Dl4jRainbow;
import weka.dl4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import weka.dl4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import weka.gui.ProgrammaticProperty;

/**
 * Converts the given Instances object into a DataSet and then constructs and returns a
 * TextEmbeddingInstanceIterator.
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
public class TextEmbeddingInstanceIterator extends AbstractInstanceIterator {

  /** The ID used to serialize this class */
  private static final long serialVersionUID = 1316260988724548474L;

  /** Word vector file location */
  protected File wordVectorLocation = new File(System.getProperty("user.dir"));

  /** Token pre processor */
  protected TokenPreProcess tokenPreProcess = new CommonPreprocessor();

  /** Tokenizer factory */
  protected TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

  /** Stop words */
  protected Dl4jAbstractStopwords stopwords = new Dl4jRainbow();

  /** Loaded word vectors */
  public transient WordVectors wordVectors;

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
    return new TextEmbeddingDataSetIterator(data, wordVectors, tokenizerFactory, tokenPreProcess, stopwords, batchSize, truncateLength);
  }


  /** Initialize the word vectors from the given file */
  protected void initWordVectors() {
    log.debug("Loading word vector model");

    final String path = wordVectorLocation.getAbsolutePath();
    final String pathLower = path.toLowerCase();
    if (pathLower.endsWith(".arff")) {
      loadEmbeddingFromArff(path);
    } else if (pathLower.endsWith(".csv")) {
      // Check if file is CSV
      boolean success = loadEmbeddingFromCSV(wordVectorLocation);
      if (!success) {
        throw new RuntimeException("Could not load the word vector file.");
      }
    } else if (pathLower.endsWith(".csv.gz")) {
      // If the file is gzipped, try the dl4j gzipped format
      try {
        wordVectors = WordVectorSerializer.loadStaticModel(wordVectorLocation);
      } catch (RuntimeException re) {
        // Dl4j format not found, continue with decompression by hand
        try {
          GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(wordVectorLocation));
          File tmpFile =
              Paths.get(System.getProperty("java.io.tmpdir"), "wordmodel-tmp.csv").toFile();
          tmpFile.delete();
          FileOutputStream fos = new FileOutputStream(tmpFile);
          int length;
          byte[] buffer = new byte[1024];
          while ((length = gzis.read(buffer)) > 0) {
            fos.write(buffer, 0, length);
          }
          fos.close();
          gzis.close();

          // Try loading decompressed CSV file
          boolean success = loadEmbeddingFromCSV(tmpFile);
          if (!success) {
            throw new RuntimeException("Could not load the word vector file.");
          }
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    } else {
      // If no file extension was caught before, try loading as is
      wordVectors = WordVectorSerializer.loadStaticModel(wordVectorLocation);
    }
  }

  /**
   * Load the embedding from a given arff file. First converts the ARFF to a temporary CSV file and
   * continues the loading mechanism with the CSV file afterwards
   *
   * @param path Path to the ARFF file
   */
  private void loadEmbeddingFromArff(String path) {
    // Try loading ARFF file
    try {
      Instances insts = new Instances(new FileReader(path));
      CSVSaver saver = new CSVSaver();
      saver.setFieldSeparator(" ");
      saver.setInstances(insts);
      final File tmpFile =
          Paths.get(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString(), ".csv")
              .toFile();
      saver.setFile(tmpFile);
      saver.setNoHeaderRow(true);
      saver.writeBatch();
      loadEmbeddingFromCSV(tmpFile);
      tmpFile.delete();
    } catch (Exception e) {
      throw new RuntimeException(
          "ARFF file could not be read (" + wordVectorLocation.getAbsolutePath() + ")", e);
    }
  }

  /**
   * Try to load the embedding from a CSV file. This iterates over different separators.
   *
   * @param f CSV file
   * @return True if loading was successful
   */
  private boolean loadEmbeddingFromCSV(File f) {
    // Try different separators
    for (String sep : new String[] {" ", ";", ",", "\t", "\\w", ":"}) {
      boolean success = loadEmbeddingFromCSV(sep, f);
      if (success) {
        return true;
      }
    }
    return false;
  }

  /**
   * Try to load the embedding from a CSV file.
   *
   * @param separator Separator for CSV interpretation
   * @param file CSV file
   * @return
   */
  private boolean loadEmbeddingFromCSV(String separator, File file) {
    try {
      File augmentedCSVFile = Paths.get(file.getAbsolutePath() + ".aug").toFile();
      augmentedCSVFile.delete();
      BufferedReader br = new BufferedReader(new FileReader(file));
      BufferedWriter bw = new BufferedWriter(new FileWriter(augmentedCSVFile));
      String line;
      // Replace separator with whitespace in each line for dl4j parsing
      while ((line = br.readLine()) != null) {
        if (!line.contains(separator)) return false; // Continue to the next separator
        bw.write(line.replaceAll(separator, " "));
        bw.newLine();
      }
      br.close();
      bw.close();

      // First try to load it as is
      try {
        wordVectors = WordVectorSerializer.loadStaticModel(augmentedCSVFile);
        if (wordVectors.vocab().words().size() != FileUtils.readLines(augmentedCSVFile).size()) {
          throw new Exception("Something went wrong"); // Continue in catch clause
        }
        augmentedCSVFile.delete();
        return true;
      } catch (Exception e) {
        // Failed: this might be due to DL4J expecting the word at pos [0] and the CSV file having
        // the word at the last position

        // Try to put the last token to the first position and reload the model
        File augmentedCSVFileSwitched = Paths.get(file.getAbsolutePath() + ".aug2").toFile();
        augmentedCSVFileSwitched.delete();
        br = new BufferedReader(new FileReader(augmentedCSVFile));
        bw = new BufferedWriter(new FileWriter(augmentedCSVFileSwitched));

        // Replace separator with whitespace in each line for dl4j parsing
        while ((line = br.readLine()) != null) {
          String[] parts = line.split(" ");
          // Join the parts, starting with the last one
          StringBuilder sb = new StringBuilder();
          sb.append(parts[parts.length - 1]);
          for (int i = 0; i < parts.length - 1; i++) {
            if (i < parts.length - 2) {
              sb.append(" ");
            }
            sb.append(parts[i]);
          }
          bw.write(sb.toString());
          bw.newLine();
        }
        br.close();
        bw.close();

        try {
          wordVectors = WordVectorSerializer.loadStaticModel(augmentedCSVFileSwitched);
          // Success
          return true;
        } catch (Exception ex) {
          return false;
        } finally {
          augmentedCSVFile.delete();
          augmentedCSVFileSwitched.delete();
        }
      } finally {
        augmentedCSVFile.delete();
      }
    } catch (IOException e) {
      // Not successful
      return false;
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


  @OptionMetadata(
      displayName = "location of word vectors",
      description = "The word vectors location.",
      commandLineParamName = "wordVectorLocation",
      commandLineParamSynopsis = "-wordVectorLocation <string>",
      displayOrder = 3
  )
  public File getWordVectorLocation() {
    return wordVectorLocation;
  }

  /**
   * Set the word vector location and try to initialize it
   *
   * @param file Word vector location
   */
  public void setWordVectorLocation(File file) {
    if (file != null && !file.equals(wordVectorLocation)) {
      this.wordVectorLocation = file;
      initWordVectors();
    }
  }

  @OptionMetadata(
      displayName = "token pre processor",
      description = "The token pre processor.",
      commandLineParamName = "tokenPreProcessor",
      commandLineParamSynopsis = "-tokenPreProcessor <string>",
      displayOrder = 4
  )

  public TokenPreProcess getTokenPreProcess() {
    return tokenPreProcess;
  }

  public void setTokenPreProcess(
      TokenPreProcess tokenPreProcess) {
    this.tokenPreProcess = tokenPreProcess;
  }

  @OptionMetadata(
      displayName = "tokenizer factory",
      description = "The tokenizer factory.",
      commandLineParamName = "tokenizerFactory",
      commandLineParamSynopsis = "-tokenizerFactory <string>",
      displayOrder = 5
  )
  public TokenizerFactory getTokenizerFactory() {
    return tokenizerFactory;
  }

  public void setTokenizerFactory(
      TokenizerFactory tokenizerFactory) {
    this.tokenizerFactory = tokenizerFactory;
  }

  @OptionMetadata(
      displayName = "stop words",
      description = "The stop words to use.",
      commandLineParamName = "stopWords",
      commandLineParamSynopsis = "-stopWords <string>",
      displayOrder = 5
  )
  public Dl4jAbstractStopwords getStopwords() {
    return stopwords;
  }

  public void setStopwords(Dl4jAbstractStopwords stopwords) {
    this.stopwords = stopwords;
  }

  @ProgrammaticProperty
  public WordVectors getWordVectors() {
    return wordVectors;
  }

  @ProgrammaticProperty
  public void setWordVectors(WordVectors wordVectors) {
    this.wordVectors = wordVectors;
  }

  public String globalInfo() {
    return "Instance iterator reads rows from the given ARFF file. This "
        + "iterator is not compatible with convolution layers. See also: "
        + "ConvolutionInstanceIterator/ImageInstanceIterator.";
  }
}
