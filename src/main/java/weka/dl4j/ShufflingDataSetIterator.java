package weka.dl4j;

import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public class ShufflingDataSetIterator implements DataSetIterator {
	
	private static final long serialVersionUID = 5571114918884888578L;
	
	private DataSet m_data = null;
	private int m_batchSize = 0;
	private int m_counter = 0;
	private Random m_random = null;
	
	public ShufflingDataSetIterator(DataSet data, int batchSize, int seed) {
		m_data = data;
		m_batchSize = batchSize;
		m_random = new Random(seed);
	}

	@Override
	public boolean hasNext() {
		if( m_counter * m_batchSize >= m_data.numExamples() ) {
			return false;
		}
		return true;
	}

	@Override
	public DataSet next() {
		DataSet thisBatch = (DataSet) m_data.getRange(m_counter*m_batchSize, (m_counter+1)*m_batchSize);
		m_counter++;
		return thisBatch;
	}

	@Override
	public DataSet next(int num) {
		return (DataSet) m_data.getRange(num*m_batchSize, (num+1)*m_batchSize);
	}

	@Override
	public int totalExamples() {
		return m_data.numExamples();
	}

	@Override
	public int inputColumns() {
		return m_data.get(0).getFeatureMatrix().columns();
	}

	@Override
	public int totalOutcomes() {
		return m_data.get(0).getLabels().columns();
	}

	@Override
	public void reset() {
		m_counter = 0;
		//m_data.shuffle();
		long next = m_random.nextLong();
        Nd4j.shuffle(m_data.getFeatureMatrix(), new Random(next), 1);
        Nd4j.shuffle(m_data.getLabels(), new Random(next), 1);
	}

	@Override
	public int batch() {
		return m_batchSize;
	}

	@Override
	public int cursor() {
		return m_counter;
	}

	@Override
	public int numExamples() {
		return m_data.numExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return null;
	}

}
