package weka.dl4j.interpretability;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

// TODO document
public abstract class AbstractSaliencyMapGenerator {

    protected ComputationGraph model;

    public AbstractSaliencyMapGenerator(ComputationGraph model) {
        this.model = model;
    }

    public abstract void generateForImage(File imageFile, int targetClassID);
}
