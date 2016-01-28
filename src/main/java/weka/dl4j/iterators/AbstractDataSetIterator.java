package weka.dl4j.iterators;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.Constants;

public abstract class AbstractDataSetIterator implements OptionHandler, Serializable {
	
	private static final long serialVersionUID = 7440584973810993954L;

	public abstract DataSetIterator getIterator(Instances data, int seed) throws Exception;
	
	private int m_trainBatchSize = 1;
	
	public void setTrainBatchSize(int trainBatchSize) {
		m_trainBatchSize = trainBatchSize;
	}
	
	@OptionMetadata(description = "Batch size for SGD", displayName = "trainBatchSize", displayOrder = 1)
	public int getTrainBatchSize() {
		return m_trainBatchSize;
	}
	
	private int m_numIterations = 10;

	public void setNumIterations(int numIterations) {
		m_numIterations = numIterations;
	}

	@OptionMetadata(description = "Number of iterations/epochs", displayName = "numIterations", displayOrder = 1)
	public int getNumIterations() {
		return m_numIterations;
	}
	
	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = weka.core.Utils.getOption(Constants.NUM_ITERATIONS, options);
		if(!tmp.equals("")) setNumIterations( Integer.parseInt(tmp) );
		// train batch size
		tmp = weka.core.Utils.getOption(Constants.TRAIN_BATCH_SIZE, options);
		if(!tmp.equals("")) setTrainBatchSize( Integer.parseInt(tmp) );
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		//String[] options = super.getOptions();
		//for (int i = 0; i < options.length; i++) {
		//	result.add(options[i]);
		//}
		// train batch size
		result.add("-" + Constants.TRAIN_BATCH_SIZE);
		result.add("" + getTrainBatchSize());
		// num iterations
		result.add("-" + Constants.NUM_ITERATIONS);
		result.add("" + getNumIterations());
		return result.toArray(new String[result.size()]);
	}

}
