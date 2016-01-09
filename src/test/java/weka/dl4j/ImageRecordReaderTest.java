package weka.dl4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;




public class ImageRecordReaderTest {
	
	public static void main(String[] args) throws Exception {
		
		int numClasses = 10;
		
		String labeledPath = "/Users/cjb60/github/wekaDeeplearning4j/mnist-data/";
		
		List<String> labels = new ArrayList<>();
		for(File f : new File(labeledPath).listFiles()) {
			String filename = f.getName();
			labels.add(filename);
		}
		
		/*
		
		ArrayList<DataSet> imgs = new ArrayList<DataSet>();
		for(int x = 0; x < labels.size(); x++) {
			System.out.println("iter: " + x);
			ImageLoader loader = new ImageLoader(28, 28);
			// load the image as an int matrix
			int[][] imgMatrix = loader.fromFile( new File(labeledPath + "/" + labels.get(x)) );
			// convert to double matrix
			double[][] imgFloat = ArrayUtil.toDouble(imgMatrix);
			// convert to indarray
			INDArray imgFloat2 = Nd4j.create(imgFloat);
			//System.out.println(imgFloat2.columns());
			//System.out.println(imgFloat2.rows());
			DataSet d = new DataSet(imgFloat2, FeatureUtil.toOutcomeVector(1, numClasses));
			imgs.add(d);
		}
		
		System.out.println(imgs.size());
		
		*/
		
		System.out.println(labels);
		
		ImageRecordReader reader = new ImageRecordReader(28, 28, true, labels);
		reader.initialize(new FileSplit(new File(labeledPath)));
		
		DataSetIterator iter = new RecordReaderDataSetIterator(reader, 784, labels.size());
		DataSet d = iter.next();
		
		System.out.println(d.get(0));
		
		
	}

}
