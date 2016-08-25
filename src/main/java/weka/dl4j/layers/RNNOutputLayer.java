package weka.dl4j.layers;

import org.nd4j.linalg.lossfunctions.LossFunctions;
import weka.dl4j.Activation;
import weka.dl4j.Constants;

import java.util.Vector;

/**
 * Created by pedro on 18-07-2016.
 */
public class RNNOutputLayer extends LSTM {

    private static final long serialVersionUID = 139321786136127207L;

    public RNNOutputLayer() {
        super();
        m_activation = Activation.SOFTMAX;
    }

    private LossFunctions.LossFunction m_lossFunction = LossFunctions.LossFunction.MCXENT;

    public LossFunctions.LossFunction getLossFunction() {
        return m_lossFunction;
    }

    public void setLossFunction(LossFunctions.LossFunction lossFunction) {
        m_lossFunction = lossFunction;
    }

    @Override
    public org.deeplearning4j.nn.conf.layers.Layer getLayer() {
        org.deeplearning4j.nn.conf.layers.RnnOutputLayer layer = new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder()
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
        if(!tmp.equals("")) setLossFunction( LossFunctions.LossFunction.valueOf(tmp) );
    }

}
