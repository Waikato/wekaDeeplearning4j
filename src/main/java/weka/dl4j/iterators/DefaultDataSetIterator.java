package weka.dl4j.iterators;

import java.util.Enumeration;
import java.util.Vector;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.DataSet;

import weka.core.Option;
import weka.dl4j.Constants;
import weka.dl4j.ShufflingDataSetIterator;

public class DefaultDataSetIterator extends AbstractDataSetIterator {

	@Override
	public DataSetIterator getIterator(DataSet dataset, int seed) {
		MultipleEpochsIterator iter = new MultipleEpochsIterator(
				getNumIterations()-1, 
				new ShufflingDataSetIterator(dataset, getTrainBatchSize(), seed));
		return iter;
	}

}
