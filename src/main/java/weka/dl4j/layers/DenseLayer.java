package weka.dl4j.layers;

import org.deeplearning4j.nn.weights.WeightInit;

public class DenseLayer implements Layer {

	private String m_activation = "tanh";
	
	public String getActivation() {
		return m_activation;
	}
	
	public void setActivation(String activation) {
		m_activation = activation;
	}
	
	private int m_numUnits = 1;
	
	public int getNumUnits() {
		return m_numUnits;
	}
	
	public void setNumUnits(int numUnits) {
		m_numUnits = numUnits;
	}
	
	@Override
	public org.deeplearning4j.nn.conf.layers.Layer getLayer(int layerNumber, int numInputs) {	
		org.deeplearning4j.nn.conf.layers.DenseLayer layer = new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
			.nIn(numInputs)
			.nOut( getNumUnits() )
			.activation( getActivation() )
			.weightInit(WeightInit.XAVIER)
			.build();
		return layer;
	}
	
	private WeightInit m_weightInit = WeightInit.XAVIER;
	
	public WeightInit getWeightInit() {
		return m_weightInit;
	}
	
	public void setWeightInit(WeightInit weightInit) {
		m_weightInit = weightInit;
	}

}
