package weka.dl4j.layers;

import java.io.Serializable;

import weka.core.OptionHandler;

public abstract class Layer implements Serializable, OptionHandler {
	
	private static final long serialVersionUID = -7317125157063984845L;

	public abstract org.deeplearning4j.nn.conf.layers.Layer getLayer();
	
	protected int m_numIncoming = 0;
	
	/**
	 * Set the number of units coming into this layer. This seems
	 * to be a weird technicality that DL4J requires to be addressed.
	 * For convolution layers, this is the number of feature maps
	 * (i.e. the number of input channels).
	 * @param numIncoming
	 */
	public void setNumIncoming(int numIncoming) {
		m_numIncoming = numIncoming;
	}
	
	protected int m_numOutgoing = 0;
	
	/**
	 * Set the number of units coming out this layer. This seems
	 * to be a weird technicality that DL4J requires to be addressed.
	 * @param numOutgoing
	 */
	public void setNumOutgoing(int numOutgoing) {
		m_numOutgoing = numOutgoing;
	}

}
