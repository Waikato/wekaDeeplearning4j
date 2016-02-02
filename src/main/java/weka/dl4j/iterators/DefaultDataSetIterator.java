package weka.dl4j.iterators;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.DataSet;

import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.dl4j.ShufflingDataSetIterator;

public class DefaultDataSetIterator extends AbstractDataSetIterator {

	private static final long serialVersionUID = 1316260988724548474L;
	
	@Override
	public int getNumAttributes(Instances data) {
		return data.numAttributes()-1;
	}
	
	@Override
	public DataSetIterator getTestIterator(Instances data, int seed, int testBatchSize) {
		DataSet dataset = Utils.instancesToDataSet(data);
		return new ShufflingDataSetIterator(dataset, testBatchSize);
	}

	@Override
	public DataSetIterator getIterator(Instances data, int seed) {
		// convert the dataset
		DataSet dataset = Utils.instancesToDataSet(data);
		MultipleEpochsIterator iter = new MultipleEpochsIterator(
				getNumIterations()-1, 
				new ShufflingDataSetIterator(dataset, getTrainBatchSize(), seed));
		return iter;
	}

}
