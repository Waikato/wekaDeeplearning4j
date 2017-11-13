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
 *    DefaultTokenizerFactory.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 */



package weka.dl4j.text.tokenization.tokenizerfactory;

import java.io.Serializable;
import java.util.Enumeration;

import weka.core.Option;
import weka.core.OptionHandler;


/**
 * A serializable version of DeepLearning4j's DefaultTokenizerFactory.
 *
 * @author Felipe Bravo-Marquez
 *
 * @version $Revision: 1 $
 */
public class DefaultTokenizerFactory extends org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory implements Serializable, OptionHandler {

	/** For Serialization */
	private static final long serialVersionUID = 762619182220458182L;

	/**
	 * Returns a string describing this object.
	 * 
	 * @return a description of the object suitable for displaying in the
	 *         explorer/experimenter gui
	 */	
	public String globalInfo() {	
		return "Default tokenizer based on string tokenizer.";
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


}
