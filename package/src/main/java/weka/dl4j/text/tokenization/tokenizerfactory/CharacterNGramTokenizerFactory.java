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
 *    TweetNLPTokenizerFactory.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 */


package weka.dl4j.text.tokenization.tokenizerfactory;

import java.io.InputStream;
import java.io.Serializable;
import java.util.Enumeration;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.tokenizers.CharacterNGramTokenizer;



/**
 * A DeepLearning4j's TokenizerFactory interface for Weka core tokenizers.
 *
 * @author Felipe Bravo-Marquez
 *
 * @version $Revision: 1 $
 */
public class CharacterNGramTokenizerFactory implements TokenizerFactory, Serializable, OptionHandler {

	/** For Serialization */
	private static final long serialVersionUID = 4694868790645893109L;

	/** The TokenPreProcess object */
	private TokenPreProcess tokenPreProcess;

	private CharacterNGramTokenizer wekaTokenizer; 

	/** the maximum number of N */
	protected int m_NMax = 3;

	/** the minimum number of N */
	protected int m_NMin = 1;


	/* (non-Javadoc)
	 * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#create(java.lang.String)
	 */
	@Override
	public Tokenizer create(String toTokenize) {

		this.wekaTokenizer = new CharacterNGramTokenizer();
		this.wekaTokenizer.setNGramMinSize(this.m_NMin);
		this.wekaTokenizer.setNGramMaxSize(this.m_NMax);

		WekaTokenizer t=new WekaTokenizer(toTokenize, wekaTokenizer);
		t.setTokenPreProcessor(tokenPreProcess);
		return t;
	}



	/* (non-Javadoc)
	 * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#create(java.io.InputStream)
	 */
	@Override
	public Tokenizer create(InputStream toTokenize) {
		return null;
	}

	/* (non-Javadoc)
	 * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#setTokenPreProcessor(org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess)
	 */
	@Override
	public void setTokenPreProcessor(TokenPreProcess preProcessor) {
		this.tokenPreProcess = preProcessor;
	}


	/* (non-Javadoc)
	 * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#getTokenPreProcessor()
	 */
	@Override
	public TokenPreProcess getTokenPreProcessor() {
		return tokenPreProcess;
	}

	/**
	 * Returns a string describing this object.
	 * 
	 * @return a description of the object suitable for displaying in the
	 *         explorer/experimenter gui
	 */	
	public String globalInfo() {	
		return "Splits a string into all character n-grams it contains based on the given maximum and minimum for n.";
	}


	/* (non-Javadoc)
	 * @see weka.core.OptionHandler#listOptions()
	 */
	@Override
	public Enumeration<Option> listOptions() {
		return Option.listOptionsForClass(this.getClass()).elements();
	}


	/* (non-Javadoc)
	 * @see weka.core.OptionHandler#setOptions(java.lang.String[])
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		Option.setOptions(options, this, this.getClass());

	}


	/* (non-Javadoc)
	 * @see weka.core.OptionHandler#getOptions()
	 */
	@Override
	public String[] getOptions() {
		return Option.getOptions(this, this.getClass());
	}


	@OptionMetadata(displayName = "NMax", 
			description = "NGram max size.",
			commandLineParamName = "NMax", commandLineParamSynopsis = "-NMax <int>",
			displayOrder = 0)		
	public int getNMax() {
		return m_NMax;
	}
	public void setNMax(int m_NMax) {
		this.m_NMax = m_NMax;
	}


	@OptionMetadata(displayName = "NMin", 
			description = "NGram min size.",
			commandLineParamName = "NMin", commandLineParamSynopsis = "-NMin <int>",
			displayOrder = 1)		
	public int getNMin() {
		return m_NMin;
	}
	public void setNMin(int m_NMin) {
		this.m_NMin = m_NMin;
	}



}
