package weka.dl4j.layers;

import java.util.Enumeration;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;

import weka.core.Option;
import weka.dl4j.Activation;

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
		// TODO Auto-generated method stub
		
	}

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		return null;
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
