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

	/**
	 * Get the number of input attributes for this iterator.
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public abstract int getNumAttributes(Instances data) throws Exception;
	
	public abstract DataSetIterator getTrainIterator(Instances data, int seed) throws Exception;
	
	public abstract DataSetIterator getTestIterator(Instances data, int seed, int testBatchSize) throws Exception;
	
	private int m_trainBatchSize = 1;
	
	public void setTrainBatchSize(int trainBatchSize) {
		m_trainBatchSize = trainBatchSize;
	}
	
	@OptionMetadata(description = "Batch size for SGD", displayName = "trainBatchSize", displayOrder = 1)
	public int getTrainBatchSize() {
		return m_trainBatchSize;
	}
	
	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// train batch size
		String tmp = weka.core.Utils.getOption(Constants.TRAIN_BATCH_SIZE, options);
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
		return result.toArray(new String[result.size()]);
	}

}
