/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * CnnSentenceDataSetIterator.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.iterators.dataset.sequence.text.cnn;

import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import weka.core.stopwords.AbstractStopwords;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

import static weka.classifiers.functions.dl4j.Utils.*;

/**
 * CnnSentenceDataSetIterator extension to Deeplearning4j implementation. This class extends its
 * parent {@link org.deeplearning4j.iterator.CnnSentenceDataSetIterator#next(int)} method to support
 * regression with numeric values.
 * <p>
 * Since {@link org.deeplearning4j.iterator.CnnSentenceDataSetIterator} only has private access to
 * its fields and most of its methods, it is necessary to make use of
 * <p>
 * - {@link weka.classifiers.functions.dl4j.Utils#getFieldValue(Object, String)},
 * - {@link weka.classifiers.functions.dl4j.Utils#setFieldValue(Object, String, Object)}
 * - {@link weka.classifiers.functions.dl4j.Utils#invokeMethod(Object, String, Object...)}
 * <p>
 * until field/method visibility is changed upstream.
 *
 * @author Steven Lang
 */
public class CnnSentenceDataSetIterator extends org.deeplearning4j.iterator.CnnSentenceDataSetIterator {

    private static final long serialVersionUID = 685006779872000151L;

    /**
     * Stop words.
     */
    private AbstractStopwords stopwords;

    /**
     * Constructor that uses {@link Builder} extended with stopwords.
     *
     * @param builder Builder
     */
    protected CnnSentenceDataSetIterator(CnnSentenceDataSetIterator.Builder builder) {
        super(builder);
        this.stopwords = builder.stopwords;
    }


    @Override
    public boolean hasNext() {
        LabeledSentenceProvider sentenceProvider = getSentenceProvider();
        if (sentenceProvider == null) {
            throw new RuntimeException("Sentenceprovider was null");
        }
        return sentenceProvider.hasNext();
    }


    @Override
    public DataSet next(int num) {
        if (!hasNext()) {
            throw new NoSuchElementException("No next element available");
        }
        LabeledSentenceProvider sentenceProvider = getSentenceProvider();
        int maxSentenceLength = getMaxSentenceLength();
        int numClasses = getNumClasses();
        String unknownWordSentinel = getUnknownWordSentinel();
        Map<String, Integer> labelClassMap = getLabelClassMap();

        List<Pair<List<String>, String>> tokenizedSentences = new ArrayList<>(num);
        int maxLength = -1;
        int minLength = Integer.MAX_VALUE;
        for (int i = tokenizedSentences.size(); i < num && sentenceProvider.hasNext(); i++) {
            Pair<String, String> p = sentenceProvider.nextSentence();
            String sentence = p.getFirst();
            String label = p.getSecond();
            List<String> tokens = tokenizeSentence(sentence);

            // Skip stopwords
            tokens = tokens.stream()
                    .filter(stopwords::isStopword)
                    .collect(Collectors.toList());

            // If tokens are empty, add at least the UNKNOWN_WORD_SENTINEL
            if (tokens.isEmpty()) {
                tokens.add(unknownWordSentinel);
            }

            // Update max/min token lengths
            maxLength = Math.max(maxLength, tokens.size());
            minLength = Math.min(minLength, tokens.size());
            tokenizedSentences.add(new Pair<>(tokens, label));
        }

        // Limit max length
        if (maxSentenceLength > 0) {
            maxLength = Math.min(maxLength, maxSentenceLength);
        }

        int currMinibatchSize = tokenizedSentences.size();

        INDArray labels = Nd4j.create(currMinibatchSize, numClasses);
        final boolean isClassification = numClasses > 1;
        if (isClassification) {
            for (int i = 0; i < currMinibatchSize; i++) {
                String labelStr = tokenizedSentences.get(i).getSecond();
                if (!labelClassMap.containsKey(labelStr)) {
                    throw new IllegalStateException(
                            "Got label \""
                                    + labelStr
                                    + "\" that is not present in list of LabeledSentenceProvider labels");
                }

                int labelIdx = labelClassMap.get(labelStr);
                labels.putScalar(i, labelIdx, 1.0);
            }
        } else if (numClasses == 1) {
            // Fix labels array for Regression task
            for (int i = 0; i < currMinibatchSize; i++) {
                final String labelStr = tokenizedSentences.get(i).getSecond();
                double lbl = Double.parseDouble(labelStr);
                labels.putScalar(i, lbl);
            }
        } else {
            throw new IllegalStateException("Number of classes must be >= 1.");
        }

        // Determine feature shape
        int[] featuresShape = new int[4];
        featuresShape[0] = currMinibatchSize;
        featuresShape[1] = 1;
        int wordVectorSize = getWordVectorSize();
        boolean sentencesAlongHeight = isSentencesAlongHeight();
        if (sentencesAlongHeight) {
            featuresShape[2] = maxLength;
            featuresShape[3] = wordVectorSize;
        } else {
            featuresShape[2] = wordVectorSize;
            featuresShape[3] = maxLength;
        }

        // Create features from tokens
        INDArray features = Nd4j.create(featuresShape);
        for (int i = 0; i < currMinibatchSize; i++) {
            List<String> currSentence = tokenizedSentences.get(i).getFirst();

            for (int j = 0; j < currSentence.size() && j < maxSentenceLength; j++) {
                INDArray vector = getVector(currSentence.get(j));

                INDArrayIndex[] indices = new INDArrayIndex[4];
                indices[0] = NDArrayIndex.point(i);
                indices[1] = NDArrayIndex.point(0);
                if (sentencesAlongHeight) {
                    indices[2] = NDArrayIndex.point(j);
                    indices[3] = NDArrayIndex.all();
                } else {
                    indices[2] = NDArrayIndex.all();
                    indices[3] = NDArrayIndex.point(j);
                }

                features.put(indices, vector);
            }
        }

        INDArray featuresMask = null;

        // Create feature mask
        if (minLength != maxLength) {
            int idxSeq;
            if (sentencesAlongHeight) {
                featuresMask = Nd4j.create(currMinibatchSize, 1, maxLength, 1);
                idxSeq = 2;
            } else {
                featuresMask = Nd4j.create(currMinibatchSize, 1, 1, maxLength);
                idxSeq = 3;
            }

            INDArrayIndex[] idxs = new INDArrayIndex[4];
            idxs[1] = NDArrayIndex.all();
            idxs[2] = NDArrayIndex.all();   //One of [2] and [3] will get replaced, depending on sentencesAlongHeight
            idxs[3] = NDArrayIndex.all();
            for (int i = 0; i < currMinibatchSize; i++) {
                idxs[0] = NDArrayIndex.point(i);
                int sentenceLength = tokenizedSentences.get(i).getFirst().size();
                if (sentenceLength >= maxLength) {
                    idxs[idxSeq] = NDArrayIndex.all();
                } else {
                    idxs[idxSeq] = NDArrayIndex.interval(0, sentenceLength);
                }
                featuresMask.get(idxs).assign(1.0);
            }
        }

        // Create DataSet
        DataSet ds = new DataSet(features, labels, featuresMask, null);
        DataSetPreProcessor dataSetPreProcessor = getDataSetPreProcessor();
        if (dataSetPreProcessor != null) {
            dataSetPreProcessor.preProcess(ds);
        }

        // Increment the cursor
        incrementCursor(ds.numExamples());
        return ds;
    }


    public static String getUnknownWordSentinel() {
        return "UNKNOWN_WORD_SENTINEL";
    }

    public LabeledSentenceProvider getSentenceProvider() {
        return getFieldValue(this, "sentenceProvider");
    }

    public int getMaxSentenceLength() {
        return getFieldValue(this, "maxSentenceLength");
    }

    public boolean isSentencesAlongHeight() {
        return getFieldValue(this, "sentencesAlongHeight");
    }

    public DataSetPreProcessor getDataSetPreProcessor() {
        return getFieldValue(this, "dataSetPreProcessor");
    }

    public int getWordVectorSize() {
        return getFieldValue(this, "wordVectorSize");
    }

    public int getNumClasses() {
        return getFieldValue(this, "numClasses");
    }

    public INDArray getUnknown() {
        return getFieldValue(this, "unknown");
    }

    public int getCursor() {
        return getFieldValue(this, "cursor");
    }

    public void setCursor(int value) {
        setFieldValue(this, "cursor", value);
    }

    protected void incrementCursor(int n) {
        int oldCurser = getCursor();
        setCursor(oldCurser + n);
    }

    public AbstractStopwords getStopwords() {
        return stopwords;
    }

    protected INDArray getVector(String s) {
        return invokeMethod(this, "getVector", s);
    }

    protected List<String> tokenizeSentence(String sentence) {
        return invokeMethod(this, "tokenizeSentence", sentence);
    }

    /**
     * CnnSentenceDataSetIterator.Builder implementation that supports stopwords.
     *
     * @author Steven Lang
     */
    public static class Builder extends org.deeplearning4j.iterator.CnnSentenceDataSetIterator.Builder {

        /**
         * Stopwords
         */
        AbstractStopwords stopwords;

        /**
         * Set stopwords.
         *
         * @param stopwords Stopwords
         * @return Builder instance
         */
        public Builder stopwords(AbstractStopwords stopwords) {
            this.stopwords = stopwords;
            return this;
        }

        /**
         * Build the iterator.
         *
         * @return {@link CnnSentenceDataSetIterator} instance
         */
        public CnnSentenceDataSetIterator build() {
            return new CnnSentenceDataSetIterator(this);
        }
    }
}
