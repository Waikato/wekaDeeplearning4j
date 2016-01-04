package weka.dl4j.layers;

import java.util.Enumeration;
import java.util.Vector;
import org.deeplearning4j.nn.weights.WeightInit;
import weka.core.Option;
import weka.core.Utils;

public class DenseLayer extends Layer {

	protected static final long serialVersionUID = -6905917800811990400L;
	
	protected String m_activation = "tanh";
	
	public String getActivation() {
		return m_activation;
	}
	
	public void setActivation(String activation) {
		m_activation = activation;
	}
	
	protected int m_numUnits = 1;
	
	public int getNumUnits() {
		return m_numUnits;
	}
	
	public void setNumUnits(int numUnits) {
		m_numUnits = numUnits;
	}
	
	protected double m_dropoutP = 0.0;
	
	public double getDropoutP() {
		return m_dropoutP;
	}
	
	public void setDropoutP(double dropoutP) {
		m_dropoutP = dropoutP;
	}
	
	protected double m_l1 = 0.0;
	
	public double getL1() {
		return m_l1;
	}
	
	public void setL1(double l1) {
		m_l1 = l1;
	}
	
	protected double m_l2 = 0.0;
	
	public double getL2() {
		return m_l2;
	}
	
	public void setL2(double l2) {
		m_l2 = l2;
	}
	
	@Override
	public org.deeplearning4j.nn.conf.layers.Layer getLayer() {	
		org.deeplearning4j.nn.conf.layers.DenseLayer layer = new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
			.nIn(m_numIncoming)
			.nOut( getNumUnits() )
			.activation( getActivation() )
			.weightInit( getWeightInit() )
			.dropOut( getDropoutP() )
			.l1( getL1() )
			.l2( getL2() )
			.build();
		return layer;
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
		String tmp = Utils.getOption(Constants.NUM_UNITS, options);
		// num units
		if(!tmp.equals("")) setNumUnits( Integer.parseInt(tmp) );
		// activation
		tmp = Utils.getOption(Constants.ACTIVATION, options);
		if(!tmp.equals("")) setActivation(tmp);
		// weight init
		tmp = Utils.getOption(Constants.WEIGHT_INIT, options);
		setWeightInit( WeightInit.valueOf(tmp) );
		// dropout
		tmp = Utils.getOption(Constants.DROPOUT_P, options);
		if(!tmp.equals("")) setDropoutP( Double.parseDouble(tmp) );
		// l1
		tmp = Utils.getOption(Constants.L1, options);
		if(!tmp.equals("")) setL1( Double.parseDouble(tmp) );		
		// l2
		tmp = Utils.getOption(Constants.L2, options);
		if(!tmp.equals("")) setL2( Double.parseDouble(tmp) );
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		// num units
		result.add("-" + Constants.NUM_UNITS);
		result.add( "" + getNumUnits() );
		// activation
		result.add( "-" + Constants.ACTIVATION);
		result.add( "" + getActivation() );
		// weight init
		result.add("-" + Constants.WEIGHT_INIT);
		result.add( "" + getWeightInit().name() );
		// dropout p
		result.add("-" + Constants.DROPOUT_P);
		result.add( "" + getDropoutP() );
		// l1
		result.add("-" + Constants.L1);
		result.add( "" + getL1() );
		// l2
		result.add("-" + Constants.L2);
		result.add( "" + getL2() );
	    return result.toArray(new String[result.size()]);
	}

}
