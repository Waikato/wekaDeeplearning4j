package weka.dl4j;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

public class ScaleImagePixelsPreProcessor implements DataSetPreProcessor {

	@Override
	public void preProcess(DataSet data) {
		data.divideBy(255);
	}

}