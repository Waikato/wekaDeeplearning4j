package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.nio.file.Files;
import java.util.List;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.dl4j.ShufflingDataSetIterator;
import weka.dl4j.iterators.DefaultDataSetIterator;

public class IrisTest {
	
	public Instances loadIris() throws Exception {
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		return data;
	}
	
	@Test
	public void test() throws Exception {
		

		Instances data = loadIris();
		
		DataSet dataset = Utils.instancesToDataSet(data);
		
		// we expect 3 minibatches per epoch (since iris has 150 instances),
		// but since we're doing 10 epochs, we expect 3 * 10 = 30 minibatches
		// in total
		MultipleEpochsIterator iter = new MultipleEpochsIterator(
				10, 
				new ShufflingDataSetIterator(dataset, 50, 0));
		iter.reset();
		int numMinibatches = 0;
		while(iter.hasNext()) {
			numMinibatches += 1;
			iter.next();
		}
		
		// SHOULD BE EQUAL TO 30, BUT IS EQUAL TO 28
		System.out.println("number of minibatches: " + numMinibatches);
		
		
		// we expect 3 mini-batches per epoch (since iris has 150 instances)
		DataSetIterator iter2 = new ShufflingDataSetIterator(dataset, 50, 0);
		iter2.reset();
		numMinibatches = 0;
		while(iter2.hasNext()) {
			numMinibatches += 1;
			iter2.next();
		}
		
		// SHOULD BE EQUAL TO 3
		System.out.println("number of minibatches: " + numMinibatches);
		
		
	}

}
