package weka.dl4j.preprocessors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

public class ImageNetPreprocessor implements DataSetPreProcessor {
    /**
     * Pre process a dataset
     * Taken from the Keras imagenet_utils.py file:
     * https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/applications/imagenet_utils.py#L181
     *
     * @param toPreProcess the data set to pre process
     */
    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray features = toPreProcess.getFeatures();
        features.divi(127.5);
        features.subi(1.0);
    }
}
