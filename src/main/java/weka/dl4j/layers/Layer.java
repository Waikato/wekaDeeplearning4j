package weka.dl4j.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;

public interface Layer {
	
	public org.deeplearning4j.nn.conf.layers.Layer getLayer(int layerNumber, int numInputs);

}
