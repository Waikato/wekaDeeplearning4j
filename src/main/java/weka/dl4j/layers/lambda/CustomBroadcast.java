package weka.dl4j.layers.lambda;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;

import java.util.Collection;

public class CustomBroadcast extends SameDiffLambdaLayer {

    private long width;

    public CustomBroadcast(long width) {
        this.width = width;
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable x) {
        long[] thisShape = x.shape().getShape();
        long minibatchSize = thisShape[0];
        long channels = thisShape[1];
        INDArray broadcasted = x.getArr().broadcast(minibatchSize, channels, this.width, this.width);
        System.err.println("CALLED");
        return x.setArray(broadcasted);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        long[] shape = inputType.getShape(false);
        long channels = shape[2];
        if (channels == 1) {
            System.err.println("Got input type in wrong order: " + inputType.toString());
            channels = shape[0];
        }
        assert channels != 1;
        return InputType.convolutional(width, width, channels, CNN2DFormat.NHWC);
//        return InputType.convolutional(width, width, channels);
    }
}
