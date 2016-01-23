package weka.classifiers.functions;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;

public class Dl4jMlpClassifierAbstractTest extends AbstractClassifierTest {

	public Dl4jMlpClassifierAbstractTest(String name) {
		super(name);
	}

	@Override
	public Classifier getClassifier() {
		Dl4jMlpClassifier mlp = new Dl4jMlpClassifier();
		mlp.setLayers(new Layer[] { new DenseLayer(), new OutputLayer() } );
		mlp.getDataSetIterator().setNumIterations(1);
		return mlp;
	}

}
