package weka.dl4j.iterators;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;

import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;

import weka.core.Instances;
import weka.dl4j.Constants;
import weka.dl4j.EasyImageRecordReader;
import weka.dl4j.SpecifiableFolderSplit;
import weka.dl4j.ScaleImagePixelsPreProcessor;
import weka.dl4j.ShufflingImageRecordReader;

public class ImageDataSetIterator extends AbstractDataSetIterator {

	private static final long serialVersionUID = -3701309032945158130L;
	
	private String m_imagesLocation = "";
	
	public String getImagesLocation() {
		return m_imagesLocation;
	}
	
	public void setImagesLocation(String imagesLocation) {
		m_imagesLocation = imagesLocation;
	}
	
	private int m_width = 0;
	
	public int getWidth() {
		return m_width;
	}
	
	public void setWidth(int width) {
		m_width = width;
	}
	
	private int m_height = 0;
	
	public int getHeight() {
		return m_height;
	}
	
	public void setHeight(int height) {
		m_height = height;
	}
	
	private int m_numChannels = 0;
	
	public int getNumChannels() {
		return m_numChannels;
	}
	
	public void setNumChannels(int numChannels) {
		m_numChannels = numChannels;
	}
	
	public void validate(Instances data) throws Exception {
		if( ! new File(getImagesLocation()).isDirectory() ) {
			throw new Exception("Directory not valid: " + getImagesLocation());
		}
		if( ! ( data.attribute(0).isString() && data.classIndex() == 1) ) {
			throw new Exception("An ARFF is required with a string attribute and a class attribute");
		}
	}
	
	@Override
	public int getNumAttributes(Instances data) {
		return getNumChannels() * getHeight() * getWidth();
	}
	
	private EasyImageRecordReader getImageRecordReader(Instances data) throws Exception {
        URI[] locations = new URI[ data.numInstances() ];
        int len = 0;
        ArrayList<File> filenames = new ArrayList<File>();
        ArrayList<String> classes = new ArrayList<String>();
        for(int x = 0; x < data.numInstances(); x++) {
        	String location = data.attribute(0).value( (int) data.get(x).value(0) );
        	filenames.add( new File(getImagesLocation() + File.separator + location) );
        	classes.add( String.valueOf(data.get(x).classValue() ) );
        	
        	File f = new File( getImagesLocation() + File.separator + location );
        	len += f.length();
        }
        EasyImageRecordReader reader = new EasyImageRecordReader(getWidth(), getHeight(), getNumChannels(), filenames, classes);
        SpecifiableFolderSplit fs = new SpecifiableFolderSplit();
        fs.setFiles(locations);
        fs.setLength(len);
        reader.initialize(fs); 
        return reader;
	}
	
	@Override
	public DataSetIterator getTestIterator(Instances data, int seed, int testBatchSize) throws Exception {
		validate(data);
        EasyImageRecordReader reader = getImageRecordReader(data);
        // we don't want to shuffle, nor do we want to do multiple epochs
        //reader.setDontShuffle(true); // TODO: "hacky"
        DataSetIterator tmpIter = new RecordReaderDataSetIterator(
        		reader, getTrainBatchSize(), -1, data.numClasses());
        tmpIter.setPreProcessor(new ScaleImagePixelsPreProcessor());
		return tmpIter;
	}

	@Override
	public DataSetIterator getTrainIterator(Instances data, int seed) throws Exception {
		validate(data);      
		EasyImageRecordReader reader = getImageRecordReader(data);
        DataSetIterator tmpIter = new RecordReaderDataSetIterator(
        		reader, getTrainBatchSize(), -1, data.numClasses());
        tmpIter.setPreProcessor(new ScaleImagePixelsPreProcessor());
		//MultipleEpochsIterator iter = new MultipleEpochsIterator(
		//		getNumIterations(), tmpIter);		
		//return iter;
        return tmpIter;
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		String tmp = weka.core.Utils.getOption(Constants.WIDTH, options);
		if(!tmp.equals("")) setWidth( Integer.parseInt(tmp) );
		tmp = weka.core.Utils.getOption(Constants.HEIGHT, options);
		if(!tmp.equals("")) setHeight( Integer.parseInt(tmp) );
		tmp = weka.core.Utils.getOption(Constants.NUM_CHANNELS, options);
		if(!tmp.equals("")) setNumChannels( Integer.parseInt(tmp) );
		tmp = weka.core.Utils.getOption(Constants.IMAGES_LOCATION, options);
		if(!tmp.equals("")) setImagesLocation(tmp);		
	}
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		String[] options = super.getOptions();
		for (int i = 0; i < options.length; i++) {
			result.add(options[i]);
		}
		// width
		result.add("-" + Constants.WIDTH);
		result.add("" + getWidth());
		// height
		result.add("-" + Constants.HEIGHT);
		result.add("" + getHeight());
		// channels
		result.add("-" + Constants.NUM_CHANNELS);
		result.add("" + getNumChannels());
		// images location
		result.add("-" + Constants.IMAGES_LOCATION);
		result.add("" + getImagesLocation());
		return result.toArray(new String[result.size()]);
	}

}
