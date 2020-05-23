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
        long channels = Math.max(inputType.getShape()[0], inputType.getShape()[2]);
        return InputType.convolutional(channels, width, width, CNN2DFormat.NCHW);
    }
}
