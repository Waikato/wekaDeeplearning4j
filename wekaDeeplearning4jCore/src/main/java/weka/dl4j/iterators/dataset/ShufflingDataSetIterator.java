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
 *    ShufflingDataSetIterator.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.dl4j.iterators.dataset;

import java.util.List;
import java.util.Random;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * An nd4j mini-batch iterator that shuffles the data whenever it is reset.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 * @author Steven Lang
 */
public class ShufflingDataSetIterator extends DefaultDataSetIterator {

	/** The ID used to serialize this class */
	private static final long serialVersionUID = 5571114918884888578L;

	/** The random number generator used for shuffling the data */
	protected Random random = null;

	/**
	 * Constructs a new shuffling iterator.
	 *
	 * @param data the dataset to operate on
	 * @param batchSize the mini batch size
	 * @param seed the seed for the random number generator
	 */
	public ShufflingDataSetIterator(DataSet data, int batchSize, int seed) {
		super(data, batchSize);
		random = new Random(seed);
	}

	/**
	 * Resets the cursor. Also shuffles the data again.
	 */
	@Override
	public void reset() {
		cursor = 0;
		data.shuffle(random.nextLong());
	}
}