package weka.classifiers.functions;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.dl4j.Activation;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RNNOutputLayer;
import weka.filters.supervised.attribute.TSLagMaker;

import java.util.List;

/**
 * Created by pedrofale on 31-08-2016 based on the WekaForecasterTest class from the TS package.
 */
public class RNNForecasterTest extends TestCase {
    public RNNForecasterTest(String name) {
        super(name);
    }

    private String predsToString(List<List<NumericPrediction>> preds, int steps) {
        StringBuffer b = new StringBuffer();

        for (int i = 0; i < steps; i++) {
            List<NumericPrediction> predsForTargetsAtStep =
                    preds.get(i);

            for (int j = 0; j < predsForTargetsAtStep.size(); j++) {
                NumericPrediction p = predsForTargetsAtStep.get(j);
                double[][] limits = p.predictionIntervals();
                b.append(p.predicted() + " ");
                if (limits != null && limits.length > 0) {
                    b.append(limits[0][0] + " " + limits[0][1] + " ");
                }
            }
            b.append("\n");
        }

        return b.toString();
    }

    public Instances getWineData() throws Exception {
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource("datasets-numeric/wine_date.arff");
        Instances data = ds.getDataSet();
        return data;
    }

    public Classifier configureRNN() {
        Dl4jRNNForecaster cls = new Dl4jRNNForecaster();
        LSTM lstm = new weka.dl4j.layers.LSTM();
        lstm.setActivation(Activation.TANH);
        RNNOutputLayer out = new weka.dl4j.layers.RNNOutputLayer();
        out.setActivation(Activation.IDENTITY);
        out.setLossFunction(LossFunctions.LossFunction.MSE);
        cls.setLayers(new weka.dl4j.layers.Layer[] { lstm, out } );
        cls.setDebugFile("/tmp/debug.txt");
        cls.setLearningRate(0.01);
        cls.setNumEpochs(10);
        cls.setIterations(5);
        return cls;
    }

    public void testForecastTwoTargetsConfidenceIntervals() throws Exception {

        boolean success = false;
        Instances wine = getWineData();

        WekaForecaster forecaster = new WekaForecaster();
        TSLagMaker lagMaker = forecaster.getTSLagMaker();

        try {
            forecaster.setBaseForecaster(configureRNN());
            forecaster.setFieldsToForecast("Fortified,Dry-white");
            forecaster.setCalculateConfIntervalsForForecasts(12);
            lagMaker.setTimeStampField("Date");
            lagMaker.setMinLag(1);
            lagMaker.setMaxLag(12);
            lagMaker.setAddMonthOfYear(true);
            lagMaker.setAddQuarterOfYear(true);
            forecaster.buildForecaster(wine, System.out);
            forecaster.primeForecaster(wine);

            int numStepsToForecast = 12;
            List<List<NumericPrediction>> forecast =
                    forecaster.forecast(numStepsToForecast, System.out);

            String forecastString = predsToString(forecast, numStepsToForecast);
            success = true;
            System.out.println(forecastString);
        } catch (Exception ex) {
            ex.printStackTrace();
            String msg = ex.getMessage().toLowerCase();
            if (msg.indexOf("not in classpath") > -1) {
                return;
            }
        }

        if (!success) {
            fail("Problem during regression testing: no successful predictions generated");
        }
    }

    public static Test suite() {
        return new TestSuite(weka.classifiers.functions.RNNForecasterTest.class);
    }

    public static void main(String[] args) {
        junit.textui.TestRunner.run(suite());
    }
}
