package weka.dl4j.interpretability;

import junit.framework.TestCase;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Utils;
import weka.dl4j.zoo.AbstractZooModel;
import weka.zoo.ZooModelTest;

import javax.imageio.ImageIO;
import java.io.File;
import java.util.List;

public class ScoreCAMTest extends TestCase {

    public void testAllKerasModels() {
        List<AbstractZooModel> kerasZooModels = ZooModelTest.createKerasModels();

        boolean tmpDir = new File("tmp").mkdir();

        for (AbstractZooModel pretrainedModel : kerasZooModels) {
            ComputationGraph computationGraph = pretrainedModel.getDefaultGraph();

            ScoreCAM scoreCAM = new ScoreCAM();
            scoreCAM.setBatchSize(8);
            scoreCAM.setComputationGraph(computationGraph);
            scoreCAM.setModelInputShape(Utils.decodeCNNShape(pretrainedModel.getShape()[0]));
            scoreCAM.setImageChannelsLast(pretrainedModel.getChannelsLast());
            scoreCAM.setImagePreProcessingScaler(pretrainedModel.getImagePreprocessingScaler());
            scoreCAM.processImage(new File("src/test/resources/images/dog.jpg"));

            String modelName = pretrainedModel.getPrettyName();

            var predictionClasses = new int[] { -1 };
            var classMap = new String[] {"Test Class"};

            try {
                ImageIO.write(scoreCAM.generateHeatmapToImage(predictionClasses, classMap, true), "png", new File(String.format("tmp/%s_composite.png", modelName)));
            } catch (OutOfMemoryError error) {
                error.printStackTrace();
            } catch (Exception ex) {
                fail();
            }
        }
    }
}