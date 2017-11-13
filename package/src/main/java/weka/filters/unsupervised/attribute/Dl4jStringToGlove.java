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
 *    Dl4jStringToGlove.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 *
 *
 */

package weka.filters.unsupervised.attribute;

import java.util.Enumeration;

import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Type;
import weka.dl4j.text.sentenceiterator.WekaInstanceSentenceIterator;

/**
 *  <!-- globalinfo-start --> An attribute filter that calculates word embeddings on a String attribute using the Glove implementation provided by
 *   DeepLearning4j.
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;@Article{Glove,
 *  Title                    = {Glove: Global Vectors for Word Representation.},
 *  Author                   = {Pennington, Jeffrey and Socher, Richard and Manning, Christopher D},
 *  Booktitle                = {EMNLP},
 *  Year                     = {2014}
 * }
 *
 *
 *
 * </pre>
 <!-- technical-bibtex-end -->
 *
 *
 * @author Felipe Bravo-Marquez (fjb11@students.waikato.ac.nz)
 *
 */
public class Dl4jStringToGlove extends Dl4jStringToWordEmbeddings {


	/** For serialization  */
	private static final long serialVersionUID = -1767367935663656698L;
	
	
	/** Parameters specifying, if cooccurrences list should be build into both directions from any current word. */
	protected boolean m_symetric = true;


	/** Parameter specifying, if cooccurrences list should be shuffled between training epochs. */
	protected boolean m_shuffle = true;
	
	/** Parameter specifying cutoff in weighting function; default 100.0 */
	protected double m_xMax= 100.0;
	
	/** Parameter in exponent of weighting function */
	protected double m_alpha = 0.75;
	
	/** The learning rate  */
	protected double m_learningRate = 0.05;

	/** The minimum learning rate */
	protected double m_minLearningRate = 0.0001;

	/** True for using adaptive gradients  */
	protected boolean m_useAdaGrad = false;

	/** This parameter specifies batch size for each thread. Also, if shuffle == TRUE, this batch will be shuffled before processing. */
	protected int m_batchSize = 1000;
	
	
	/**
	 * Returns a string describing this filter.
	 *
	 * @return a description of the filter suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	@Override
	public String globalInfo() {
		return "Calculates word embeddings on a string attribute using the Glove method.\n"
				+ "More info at: https://nlp.stanford.edu/projects/glove/ .\n"
				+ getTechnicalInformation().toString();
	}


	/**
	 * Returns an instance of a TechnicalInformation object, containing
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 *
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;
		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(TechnicalInformation.Field.AUTHOR, "Pennington, Jeffrey and Socher, Richard and Manning, Christopher D");
		result.setValue(TechnicalInformation.Field.TITLE, "Glove: Global Vectors for Word Representation");
		result.setValue(TechnicalInformation.Field.YEAR, "2014");
		result.setValue(TechnicalInformation.Field.BOOKTITLE, "EMNLP.");

		return result;
	}

	@Override
	public Enumeration<Option> listOptions() {
		//this.getClass().getSuperclass()
		return Option.listOptionsForClassHierarchy(this.getClass(), this.getClass().getSuperclass()).elements();
	}


	/* (non-Javadoc)
	 * @see weka.filters.Filter#getOptions()
	 */
	@Override
	public String[] getOptions() {
		return Option.getOptionsForHierarchy(this, this.getClass().getSuperclass());
		
		//return Option.getOptions(this, this.getClass());
	}


	/**
	 * Parses the options for this object.
	 *
	 *
	 * @param options
	 *            the options to use
	 * @throws Exception
	 *             if setting of options fails
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		Option.setOptionsForHierarchy(options, this, this.getClass().getSuperclass());
		// Option.setOptions(options, this, this.getClass());
	}

	/* (non-Javadoc)
	 * @see Dl4jStringToWordEmbeddings#initiliazeVectors(weka.core.Instances)
	 */
	@Override
	void initiliazeVectors(Instances instances) {
		SentenceIterator iter = new WekaInstanceSentenceIterator(instances,this.m_textIndex-1);

		// sets the tokenizer
		this.m_tokenizerFactory.setTokenPreProcessor(this.m_preprocessor);

		// initializes stopwords
		this.m_stopWordsHandler.initialize();


		// Building model
		this.vec = new Glove.Builder()
		.iterate(iter)
		.tokenizerFactory(this.m_tokenizerFactory)
		.alpha(this.m_alpha)
		.learningRate(this.m_learningRate)
		.epochs(this.m_epochs)
		.layerSize(this.m_layerSize)
		.minLearningRate(this.m_minLearningRate)
		.minWordFrequency(this.m_minWordFrequency)
		.stopWords(this.m_stopWordsHandler.getStopList())
		.useAdaGrad(this.m_useAdaGrad)
		.windowSize(this.m_windowSize)
		.workers(this.m_workers)
		.windowSize(this.m_windowSize)
		.xMax(this.m_xMax)
		.batchSize(this.m_batchSize)
		.shuffle(this.m_shuffle)
		.symmetric(this.m_symetric)
		.build();

		// fit model
		this.vec.fit();
		
	}
	

	



	@OptionMetadata(displayName = "learningRate",
			description = "The learning rate (default = 0.05).",
			commandLineParamName = "learningRate", commandLineParamSynopsis = "-learningRate <double>",
			displayOrder = 15)
	public double getLearningRate() {
		return m_learningRate;
	}
	public void setLearningRate(double m_learningRate) {
		this.m_learningRate = m_learningRate;
	}

	@OptionMetadata(displayName = "minLearningRate",
			description = "This method defines minimal learning rate value for training (default = 1.0E-4).",
			commandLineParamName = "minLearningRate", commandLineParamSynopsis = "-minLearningRate <double>",
			displayOrder = 16)
	public double getMinLearningRate() {
		return m_minLearningRate;
	}
	public void setMinLearningRate(double m_minLearningRate) {
		this.m_minLearningRate = m_minLearningRate;
	}


	@OptionMetadata(displayName = "symetric",
			description = "Parameters specifying, if cooccurrences list should be build into both directions from any current word (default = true).",
			commandLineParamName = "symetric", commandLineParamSynopsis = "-symetric", commandLineParamIsFlag = true,
			displayOrder = 17)
	public boolean isSymetric() {
		return m_symetric;
	}
	public void setSymetric(boolean m_symetric) {
		this.m_symetric = m_symetric;
	}


	@OptionMetadata(displayName = "shuffle",
			description = "Parameter specifying, if cooccurrences list should be shuffled between training epochs (default = true).",
			commandLineParamName = "shuffle", commandLineParamSynopsis = "-shuffle", commandLineParamIsFlag = true,
			displayOrder = 18)
	public boolean isShuffle() {
		return m_shuffle;
	}
	public void setShuffle(boolean m_shuffle) {
		this.m_shuffle = m_shuffle;
	}


	@OptionMetadata(displayName = "xMax",
			description = "Parameter specifying cutoff in weighting function (default = 100.0).",
			commandLineParamName = "xMax", commandLineParamSynopsis = "-xMax <double>",
			displayOrder = 19)
	public double getXMax() {
		return m_xMax;
	}
	public void setXMax(double m_xMax) {
		this.m_xMax = m_xMax;
	}


	@OptionMetadata(displayName = "alpha",
			description = "Parameter in exponent of weighting function (default = 0.75).",
			commandLineParamName = "alpha", commandLineParamSynopsis = "-alpha <double>",
			displayOrder = 20)
	public double getAlpha() {
		return m_alpha;
	}
	public void setAlpha(double m_alpha) {
		this.m_alpha = m_alpha;
	}
	
	@OptionMetadata(displayName = "useAdaGrad",
			description = "This method defines whether adaptive gradients should be used or not (default = false).",
			commandLineParamName = "useAdaGrad", commandLineParamSynopsis = "-useAdaGrad", commandLineParamIsFlag = true,
			displayOrder = 21)
	public boolean isUseAdaGrad() {
		return m_useAdaGrad;
	}
	public void setUseAdaGrad(boolean m_useAdaGrad) {
		this.m_useAdaGrad = m_useAdaGrad;
	}


	@OptionMetadata(displayName = "batchSize",
			description = "The mini-batch size (default = 1000).",
			commandLineParamName = "batchSize", commandLineParamSynopsis = "-batchSize <int>",
			displayOrder = 22)
	public int getBatchSize() {
		return m_batchSize;
	}
	public void setBatchSize(int m_batchSize) {
		this.m_batchSize = m_batchSize;
	}



}
