package weka.dl4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.loader.ImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.vectorizer.ImageVectorizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.util.FeatureUtil;

public class ImageRecordReaderTest {
	
	public static void main(String[] args) throws Exception {
		
		int numClasses = 10;
		
		String labeledPath = "/Users/cjb60/github/wekaDeeplearning4j/mnist-data/0/";
		
		List<String> labels = new ArrayList<>();
		for(File f : new File(labeledPath).listFiles()) {
			String filename = f.getName();
			labels.add(filename);
		}
		
		System.out.println( labeledPath + "/" + labels.get(0) );
		
		ImageLoader loader = new ImageLoader(28, 28);
		INDArray imgMatrix = loader.asMatrix( new File(labeledPath) );
		INDArray label = FeatureUtil.toOutcomeVector(0, 10);
		
		DataSet d = new DataSet(imgMatrix, label);
		
		System.out.println(d);
	}

}
