package weka.dl4j;

import java.io.File;

import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;

public class ImageRecordReaderTest {
	
	public static void main(String[] args) throws Exception {
		
		ImageRecordReader reader = new ImageRecordReader(28, 28);
		reader.initialize( new FileSplit(new File("mnist-data")) );
		
		
		
	}

}
