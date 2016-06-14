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
		
		/*
		MultipleEpochsIterator iter = new MultipleEpochsIterator(
				10, 
				new ShufflingDataSetIterator(dataset, 50, 0));
		iter.reset();
		while(iter.hasNext()) {
			System.out.println("stuff");
			iter.next();
		}
		*/
		
		DataSetIterator iter = new ShufflingDataSetIterator(dataset, 50, 0);
		iter.reset();
		while(iter.hasNext()) {
			System.out.println("stuff");
			iter.next();
		}
		
	}

}
