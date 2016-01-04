package weka.dl4j.layers;

import java.io.Serializable;

import weka.core.OptionHandler;

public interface Layer extends Serializable, OptionHandler {
	
	public org.deeplearning4j.nn.conf.layers.Layer getLayer(int layerNumber, int numInputs);

}
