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
 *    Dl4jStringToWordEmbeddings.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 *
 *
 */




package weka.filters.unsupervised.attribute;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;

import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.core.Capabilities.Capability;
import weka.dl4j.text.stopwords.Dl4jAbstractStopwords;
import weka.dl4j.text.stopwords.Dl4jNull;
import weka.dl4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import weka.dl4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import weka.filters.SimpleBatchFilter;


/**
 *  <!-- globalinfo-start --> An abstract attribute filter that calculates word embeddings on a String attribute. 
 * <!-- globalinfo-end -->
 *  
 * 
 * @author Felipe Bravo-Marquez (fjb11@students.waikato.ac.nz)
 * @version $Revision: 1 $
 */


public abstract class Dl4jStringToWordEmbeddings extends SimpleBatchFilter {


	/** For serialization  */
	private static final long serialVersionUID = 3748678887246129719L;

	/** The object where word embeddings are stored */
	protected SequenceVectors<VocabWord> vec;

	


	/**  Possible actions to perform on the embeddings.  */
	protected enum Action {
		WORD_VECTOR,
		DOC_VECTOR_AVERAGE,
		DOC_VECTOR_ADD,
		DOC_VECTOR_CONCAT
	}
	
	
	/** Prefix for embedding attributes */
	protected String m_embedding_prefix="embedding-";
	

	/** Number of words (from left to right) of the tweet whose embeddings will be concatenated. */
	protected int m_concat_words = 15;


	/** Action to perform on the embeddings. This action will define whether word or document vectors are produced. */
	protected Action m_action = Action.WORD_VECTOR;


	/** The TokenPreProcess object  */
	protected TokenPreProcess m_preprocessor=new CommonPreprocessor();

	/** The TokenizerFactory object  */
	protected TokenizerFactory m_tokenizerFactory=new DefaultTokenizerFactory();

	/** The Stopwords class */
	protected Dl4jAbstractStopwords m_stopWordsHandler=new Dl4jNull();


	/** The number of epochs  */
	protected int m_epochs = 1;


	/** The maximum number of concurrent threads available for training.  */
	protected int m_workers=Runtime.getRuntime().availableProcessors();;


	/** the index of the string attribute to be processed. */
	protected int m_textIndex=1; 

	/** the minimum frequency of a word to be considered. */
	protected int m_minWordFrequency=5;

	/** the layer size. */
	protected int m_layerSize=100;

	/** the number of iterations */
	protected int m_iterations=1;

	/** the size of the window. */
	protected int m_windowSize=5;

	/** Random number seed  */
	protected int m_seed=1;




	/* (non-Javadoc)
	 * @see weka.filters.Filter#listOptions()
	 */
	@Override
	public Enumeration<Option> listOptions() {
		return Option.listOptionsForClass(this.getClass()).elements();
	}


	/* (non-Javadoc)
	 * @see weka.filters.Filter#getOptions()
	 */
	@Override
	public String[] getOptions() {		
		return Option.getOptions(this, this.getClass());
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
		Option.setOptions(options, this, this.getClass());
	}


	/* (non-Javadoc)
	 * @see weka.filters.Filter#getCapabilities()
	 */
	@Override
	public Capabilities getCapabilities() {

		Capabilities result = new Capabilities(this);
		result.disableAll();

		// attributes
		result.enableAllAttributes();
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enableAllClasses();
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NO_CLASS);

		result.setMinimumNumberInstances(0);

		return result;
	}



	/* (non-Javadoc)
	 * @see weka.filters.SimpleFilter#determineOutputFormat(weka.core.Instances)
	 */
	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {

		ArrayList<Attribute> att = new ArrayList<Attribute>();


		if(this.m_action.equals(Action.WORD_VECTOR)){
			// Add one attribute for each embedding dimension
			for(int i=0;i<this.m_layerSize;i++){
				att.add(new Attribute(m_embedding_prefix+i));
			}	

			att.add(new Attribute("word_id", (ArrayList<String>) null));


			Instances result = new Instances("Word vectors calculated from:"+inputFormat.relationName(), att, 0);

			return result;

		}


		// Represent doc vectors
		else {
			// Adds all attributes of the inputformat
			for (int i = 0; i < inputFormat.numAttributes(); i++) {
				att.add(inputFormat.attribute(i));
			}
			if(this.m_action.equals(Action.DOC_VECTOR_ADD)||this.m_action.equals(Action.DOC_VECTOR_AVERAGE))
				for(int i=0;i<this.m_layerSize;i++){
					att.add(new Attribute(m_embedding_prefix+i));
				}	
			else if(this.m_action.equals(Action.DOC_VECTOR_CONCAT))
				for(int i=0;i<this.m_concat_words;i++){
					for(int j=0;j<this.m_layerSize;j++){
						att.add(new Attribute(m_embedding_prefix+i+","+j));
					}			
				}


			Instances result = new Instances(inputFormat.relationName(), att, 0);

			// set the class index
			result.setClassIndex(inputFormat.classIndex());


			return result;
		}


	}



	/**
	 * Calculates word embeddings from Weka Instances
	 * 
	 * @param instances the Weka Instances object
	 */
	abstract void initiliazeVectors(Instances instances);



	/* (non-Javadoc)
	 * @see weka.filters.SimpleFilter#process(weka.core.Instances)
	 */
	@Override
	protected Instances process(Instances instances) throws Exception {

		Instances result = getOutputFormat();

		if(this.m_textIndex>instances.numAttributes())
			throw new IOException("Invalid attribute index.");

		if(!instances.attribute(this.m_textIndex-1).isString())
			throw new IOException("Given attribute is not String.");


		// create Embeddings in the first batch
		if(!isFirstBatchDone())		{	
			this.initiliazeVectors(instances);
		}


		// outputs the word vectors
		if(this.m_action.equals(Action.WORD_VECTOR)){

			String[] words= this.vec.getVocab().words().toArray(new String[0]);
			Arrays.sort(words);

			for (String word:words){
				double[] values = new double[result.numAttributes()];

				for(int i=0;i<this.vec.getWordVector(word).length;i++)
					values[i]=this.vec.getWordVector(word)[i];

				values[result.numAttributes()-1] = result.attribute("word_id").addStringValue(word);

				Instance inst = new DenseInstance(1, values);

				inst.setDataset(result);
				result.add(inst);

			}

		}

		// outputs doc vectors using the embeddings
		else {
			// reference to the content of the message, users index start from zero
			Attribute attrCont = instances.attribute(this.m_textIndex-1);

			// copy all previous attributes
			for (int i = 0; i < instances.numInstances(); i++) {
				double[] values = new double[result.numAttributes()];
				for (int n = 0; n < instances.numAttributes(); n++)
					values[n] = instances.instance(i).value(n);

				String content = instances.instance(i).stringValue(attrCont);
				List<String> words = this.m_tokenizerFactory.create(content).getTokens();

				int m=0;
				for(String word:words){
					if(this.vec.hasWord(word)){
						int j=0;						
						for(double embDimVal:this.vec.getWordVector(word)){						
							if(this.m_action==Action.DOC_VECTOR_AVERAGE){
								values[result.attribute(m_embedding_prefix+j).index()] += embDimVal/words.size();	
							}
							else if(this.m_action==Action.DOC_VECTOR_ADD){
								values[result.attribute(m_embedding_prefix+j).index()] += embDimVal;
							}
							else if(this.m_action==Action.DOC_VECTOR_CONCAT){
								if(m<this.m_concat_words){
									values[result.attribute(m_embedding_prefix+m+","+j).index()] += embDimVal;
								}
							}

							j++;
						}					
					}
					m++;

				}



				Instance inst = new DenseInstance(1, values);

				inst.setDataset(result);

				// copy possible strings, relational values...
				copyValues(inst, false, instances, result);

				result.add(inst);

			}
		}



		return result;


	}



	public int getM_concat_words() {
		return m_concat_words;
	}


	public void setM_concat_words(int m_concat_words) {
		this.m_concat_words = m_concat_words;
	}


	@OptionMetadata(
			displayName = "action",
			description = "The action to perform on the embeddings: 1) report embeddings (WORD_VECTOR), "
					+ "2) Average embeddings of the input string (DOC_VECTOR_AVERAGE),"
					+ "3) Add embeddings of the input string (DOC_VECTOR_ADD), "
					+ "4) Concatenate the first *concat_words* embeddings of  the input string (DOC_VECTOR_CONCAT), (default WORD_VECTOR).",
			commandLineParamName = "action", commandLineParamSynopsis = "-level <speficiation>",
			displayOrder = 1)	
	public Action getAction() {
		return m_action;
	}
	public void setAction(Action action) {
		this.m_action = action;
	}
	
	
	
	@OptionMetadata(displayName = "concat_words",
			description = "Number of words (from left to right) of the tweet whose embeddings will be concatenated."
					+ "This parameter only applies if action=DOC_VECTOR_CONCAT (default = 15).",
			commandLineParamName = "concat_words", commandLineParamSynopsis = "-concat_words <int>",
			displayOrder = 2)		
	public int getConcat_words() {
		return m_concat_words;
	}
	public void setConcat_words(int m_concat_words) {
		this.m_concat_words = m_concat_words;
	}
	
	
	
	@OptionMetadata(displayName = "stopWordsHandler",
			description = "The stopWordsHandler. Dl4j Null means no stop words are used.",
			commandLineParamName = "stopWordsHandler", commandLineParamSynopsis = "-stopWordsHandler <String>",
			displayOrder = 3)		
	public Dl4jAbstractStopwords getStopWordsHandler() {
		return m_stopWordsHandler;
	}

	public void setStopWordsHandler(Dl4jAbstractStopwords m_stopWordsHandler) {
		this.m_stopWordsHandler = m_stopWordsHandler;
	}


	@OptionMetadata(displayName = "tokenizerFactory",
			description = "The tokenizer factory to use on the strings. Default: DefaultTokenizer.",
			commandLineParamName = "tokenizerFactory", commandLineParamSynopsis = "-tokenizerFactory <String>",
			displayOrder = 4)	
	public TokenizerFactory getTokenizerFactory() {
		return m_tokenizerFactory;
	}
	public void setTokenizerFactory(TokenizerFactory m_tokenizerFactory) {
		this.m_tokenizerFactory = m_tokenizerFactory;
	}



	@OptionMetadata(displayName = "preprocessor",
			description = "The token Preprocessor for preprocessing the Strings. Default: CommonPreprocessor.",
			commandLineParamName = "preprocessor", commandLineParamSynopsis = "-preprocessor <String>",
			displayOrder = 5)	
	/**
	 * Gets the action for the selected preprocessor.
	 * 
	 * @return the current action.
	 */
	public TokenPreProcess getPreProcessor() {
		return this.m_preprocessor;
	}

	/**
	 * Sets the preprocessor action.
	 * 
	 * @param value the action type
	 * 
	 */
	public void setPreProcessor(TokenPreProcess value) {
		this.m_preprocessor=value;
	}


	@OptionMetadata(displayName = "attribute string index",
			description = "The attribute string index (starting from 1) to process (default = 1).",
			commandLineParamName = "index", commandLineParamSynopsis = "-index <int>",
			displayOrder = 6)
	/**
	 * Get the position of the target string.
	 * 
	 * @return the index of the target string
	 */	
	public int getTextIndex() {
		return m_textIndex;
	}
	/**
	 * Set the attribute's index with the string to process.
	 * 
	 * @param textIndex the index value name
	 */
	public void setTextIndex(int textIndex) {
		this.m_textIndex = textIndex;
	}



	@OptionMetadata(displayName = "minWordFrequency",
			description = "The minimum word frequency (default = 5).",
			commandLineParamName = "minWordFrequency", commandLineParamSynopsis = "-minWordFrequency <int>",
			displayOrder = 7)
	public int getMinWordFrequency() {
		return m_minWordFrequency;
	}
	public void setMinWordFrequency(int minWordFrequency) {
		this.m_minWordFrequency = minWordFrequency;
	}


	@OptionMetadata(displayName = "layerSize",
			description = "The size of the word vectors (default = 100).",
			commandLineParamName = "layerSize", commandLineParamSynopsis = "-layerSize <int>",
			displayOrder = 8)
	public int getLayerSize() {
		return m_layerSize;
	}
	public void setLayerSize(int layerSize) {
		this.m_layerSize = layerSize;
	}


	@OptionMetadata(displayName = "iterations",
			description = "The number of iterations (default = 1).",
			commandLineParamName = "iterations", commandLineParamSynopsis = "-iterations <int>",
			displayOrder = 9)	
	public int getIterations() {
		return m_iterations;
	}
	public void setIterations(int iterations) {
		this.m_iterations = iterations;
	}



	@OptionMetadata(displayName = "windowSize",
			description = "The size of the window (default = 5).",
			commandLineParamName = "windowSize", commandLineParamSynopsis = "-windowSize <int>",
			displayOrder = 10)
	public int getWindowSize() {
		return m_windowSize;
	}
	public void setWindowSize(int windowSize) {
		this.m_windowSize = windowSize;
	}




	@OptionMetadata(displayName = "epochs", 
			description = "The number of epochs (iterations over whole training corpus) for training (default = 1).",
			commandLineParamName = "epochs", commandLineParamSynopsis = "-epochs <int>",
			displayOrder = 11)		
	public int getEpochs() {
		return m_epochs;
	}
	public void setEpochs(int m_epochs) {
		this.m_epochs = m_epochs;
	}


	@OptionMetadata(displayName = "workers", 
			description = "The maximum number of concurrent threads available for training.",
			commandLineParamName = "workers", commandLineParamSynopsis = "-workers <int>",
			displayOrder = 12)		
	public int getWorkers() {
		return m_workers;
	}
	public void setWorkers(int m_workers) {
		this.m_workers = m_workers;
	}



	@OptionMetadata(displayName = "seed",
			description = "The random number seed to be used. (default = 1).",
			commandLineParamName = "seed", commandLineParamSynopsis = "-seed <int>",
			displayOrder = 13)	
	public int getSeed() {
		return m_seed;
	}
	public void setSeed(int m_seed) {
		this.m_seed = m_seed;
	}

	

	@OptionMetadata(displayName = "embedding_prefix",
			description = "The prefix for each embedding attribute. Default: \"embedding-\".",
			commandLineParamName = "embedding_prefix", commandLineParamSynopsis = "-embedding_prefix <String>",
			displayOrder = 14)		
	public String getEmbedding_prefix() {
		return m_embedding_prefix;
	}
	public void setEmbedding_prefix(String m_embedding_prefix) {
		this.m_embedding_prefix = m_embedding_prefix;
	}





}
