package weka.dl4j.layers;

import java.util.Enumeration;
import java.util.Vector;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;

import weka.core.Option;
import weka.core.Utils;
import weka.dl4j.Activation;
import weka.dl4j.Constants;

public class Conv2DLayer extends Layer {
	
	private static final long serialVersionUID = 6905344091980568487L;
	
	private int m_filterSizeX = 0;
	
	public int getFilterSizeX() {
		return m_filterSizeX;
	}
	
	public void setFilterSizeX(int filterSizeX) {
		m_filterSizeX = filterSizeX;
	}
	
	private int m_filterSizeY = 0;
	
	public int getFilterSizeY() {
		return m_filterSizeY;
	}
	
	public void setFilterSizeY(int filterSizeY) {
		m_filterSizeY = filterSizeY;
	}
	
	private int m_numFilters = 0;
	
	public int getNumFilters() {
		return m_numFilters;
	}
	
	public void setNumFilters(int numFilters) {
		m_numFilters = numFilters;
	}
	
	private int m_strideX = 1;
	
	public int getStrideX() {
		return m_strideX;
	}
	
	public void setStrideX(int strideX) {
		m_strideX = strideX;
	}
	
	private int m_strideY = 1;
	
	public int getStrideY() {
		return m_strideY;
	}
	
	public void setStrideY(int strideY) {
		m_strideY = strideY;
	}
	
	protected Activation m_activation = Activation.RELU;
	
	public Activation getActivation() {
		return m_activation;
	}
	
	public void setActivation(Activation activation) {
		m_activation = activation;
	}
	
	protected WeightInit m_weightInit = WeightInit.XAVIER;
	
	public WeightInit getWeightInit() {
		return m_weightInit;
	}
	
	public void setWeightInit(WeightInit weightInit) {
		m_weightInit = weightInit;
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// num filters
		String tmp = Utils.getOption(Constants.NUM_FILTERS, options);
		if(!tmp.equals("")) setNumFilters( Integer.parseInt(tmp) );
		// filter size x
		tmp = Utils.getOption(Constants.FILTER_SIZE_X, options);
		if(!tmp.equals("")) setFilterSizeX(Integer.parseInt(tmp));
		// filter size y
		tmp = Utils.getOption(Constants.FILTER_SIZE_Y, options);
		if(!tmp.equals("")) setFilterSizeY(Integer.parseInt(tmp));
		// stride x
		tmp = Utils.getOption(Constants.STRIDE_X, options);
		if(!tmp.equals("")) setStrideX(Integer.parseInt(tmp));
		// stride y
		tmp = Utils.getOption(Constants.STRIDE_Y, options);
		if(!tmp.equals("")) setStrideY(Integer.parseInt(tmp));
		// activation
		tmp = Utils.getOption(Constants.ACTIVATION, options);
		if(!tmp.equals("")) setActivation( Activation.valueOf(tmp.toUpperCase()) );
		// weight init
		tmp = Utils.getOption(Constants.WEIGHT_INIT, options);
		setWeightInit( WeightInit.valueOf(tmp) );
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		// num filters
		result.add("-" + Constants.NUM_FILTERS);
		result.add( "" + getNumFilters() );
		// filter size x
		result.add("-" + Constants.FILTER_SIZE_X);
		result.add( "" + getFilterSizeX() );
		// filter size y
		result.add("-" + Constants.FILTER_SIZE_Y);
		result.add( "" + getFilterSizeY() );
		// stride x
		result.add( "-" + Constants.STRIDE_X);
		result.add( "" + getStrideX() );
		// stride y
		result.add( "-" + Constants.STRIDE_Y);
		result.add( "" + getStrideY() );
		// activation
		result.add( "-" + Constants.ACTIVATION);
		result.add( "" + getActivation().name().toLowerCase() );
		// weight init
		result.add("-" + Constants.WEIGHT_INIT);
		result.add( "" + getWeightInit().name() );
	    return result.toArray(new String[result.size()]);
	}

	@Override
	public org.deeplearning4j.nn.conf.layers.Layer getLayer() {
    	ConvolutionLayer layer = new ConvolutionLayer.Builder( getFilterSizeX(), getFilterSizeY() )
			.stride( getStrideX(), getStrideY() )
			.nIn(m_numIncoming)
			.nOut( getNumFilters() )
			.activation( getActivation().name().toLowerCase() )
			.weightInit( getWeightInit() )
			.build();
    	return layer;
	}

}
