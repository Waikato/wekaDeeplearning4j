package weka.dl4j;

import java.net.URI;
import java.util.List;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.util.MathArrays;
import org.canova.image.recordreader.ImageRecordReader;

public class ShufflingImageRecordReader extends ImageRecordReader {
	
	private static final long serialVersionUID = -417326936766115928L;
	
    public ShufflingImageRecordReader(int width, int height, int channels, boolean appendLabel, List<String> labels) {
        super(width, height, channels, appendLabel, labels);
    }
	
	/**
	 * This is a bit hacky but is necessary if we want to do proper SGD (i.e. shuffling per epoch)
	 * with images.
	 */
	@Override
	public void reset() {
		
		// HACKY
		
		URI[] locations = this.inputSplit.locations();
		//for(URI tmp : locations) {
		//	System.out.println(tmp);
		//}
		
		//System.exit(0);
		
		// create copy of locations
		URI[] copy = new URI[locations.length];
		for(int x = 0; x < locations.length; x++) {
			copy[x] = locations[x];
		}
		
		// shuffle an array of ints
		int[] indices = new int[ locations.length ];
		for(int x = 0; x < locations.length; x++) {
			indices[x] = x;
		}
		
		JDKRandomGenerator rnd = new JDKRandomGenerator();
		rnd.setSeed(0);
		MathArrays.shuffle(indices, rnd);
		
		// transfer
		for(int x = 0; x < locations.length; x++) {
			locations[x] = copy[ indices[x] ];
		}
		
		super.reset();
	}


}
