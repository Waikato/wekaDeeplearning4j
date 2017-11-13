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
 *    StemmingPreProcessor.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 */


package weka.dl4j.text.tokenization.tokenizer.preprocessor;

import java.io.Serializable;
import java.util.Enumeration;

import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.stemmers.NullStemmer;
import weka.core.stemmers.Stemmer;

/**
 * Implements basic cleaning inherited from CommonPreprocessor + does stemming using a Weka Stemmer.
 *
 * @author Felipe Bravo-Marquez
 *
 *
 */
public class StemmingPreprocessor extends org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor implements  Serializable, OptionHandler{

	
	/** For serialization */
	private static final long serialVersionUID = 436336311776463684L;

	
	/** A Weka stemmer objet */	
	private Stemmer m_stemmer= new NullStemmer();


	/**
	 * Returns a string describing this object.
	 * 
	 * @return a description of the object suitable for displaying in the
	 *         explorer/experimenter gui
	 */	
	public String globalInfo() {
		return "This tokenizer preprocessor implements basic cleaning inherited from CommonPreprocessor + does stemming using a Weka Stemmer.\n";

	}	


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
	 * @see org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor#preProcess(java.lang.String)
	 */
	@Override
	public String preProcess(String token) {
		String prep = super.preProcess(token);    
		return m_stemmer.stem(prep);
	}


	@OptionMetadata(displayName = "stemmer",
			description = "The Weka stemmer to use.",
			commandLineParamName = "stemmer", commandLineParamSynopsis = "-stemmer <String>",
			displayOrder = 0)		
	public Stemmer getStemmer() {
		return m_stemmer;
	}
	public void setStemmer(Stemmer m_stemmer) {
		this.m_stemmer = m_stemmer;
	}	




}
