package weka.dl4j.iterators;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

import weka.core.Instances;

public class MnistDebugIterator extends AbstractDataSetIterator {

	private static final long serialVersionUID = -8490189565204476744L;

	@Override
	public int getNumAttributes(Instances data) throws Exception {
		return data.numAttributes()-1;
	}

	@Override
	public DataSetIterator getTrainIterator(Instances data, int seed)
			throws Exception {
		DataSetIterator mnistTrain = new MnistDataSetIterator( getTrainBatchSize(), true, seed);
		return mnistTrain;	
	}

	@Override
	public DataSetIterator getTestIterator(Instances data, int seed,
			int testBatchSize) throws Exception {
		DataSetIterator mnistTest = new MnistDataSetIterator( testBatchSize, false, seed);
		return mnistTest;
	}	

}
