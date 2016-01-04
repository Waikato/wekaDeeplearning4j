package weka.dl4j.layers;

import java.util.Vector;
import weka.dl4j.Constants;
import weka.dl4j.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class OutputLayer extends DenseLayer {
	
	private static final long serialVersionUID = 139321786136127207L;
	
	public OutputLayer() {
		super();
		m_activation = Activation.SOFTMAX;
	}
	
	private LossFunction m_lossFunction = LossFunction.MCXENT;

	public LossFunction getLossFunction() {
		return m_lossFunction;
	}

	public void setLossFunction(LossFunction lossFunction) {
		m_lossFunction = lossFunction;
	}
	
	@Override
	public org.deeplearning4j.nn.conf.layers.Layer getLayer() {	
		org.deeplearning4j.nn.conf.layers.OutputLayer layer = new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
			.nIn(m_numIncoming)
			.nOut(m_numOutgoing)
			.activation( getActivation().name().toLowerCase() )
			.weightInit( getWeightInit() )
			.dropOut( getDropoutP() )
			.l1( getL1() )
			.l2( getL2() )
			.lossFunction( getLossFunction() )
			.build();
		return layer;
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		String[] options = super.getOptions();
		for (int i = 0; i < options.length; i++) {
			result.add(options[i]);
		}
		// loss function
		result.add("-" + Constants.LOSS_FUNCTION);
		result.add("" + getLossFunction().name());
		return result.toArray(new String[result.size()]);
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		// loss function
		String tmp = weka.core.Utils.getOption(Constants.LOSS_FUNCTION, options);
		if(!tmp.equals("")) setLossFunction( LossFunction.valueOf(tmp) );
	}

}
