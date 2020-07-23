package weka.dl4j.playground;

import junit.framework.TestCase;
import lombok.extern.log4j.Log4j2;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.AbstractZooModel;
import weka.dl4j.zoo.Dl4jVGG;
import weka.dl4j.zoo.KerasResNet;
import weka.dl4j.zoo.keras.VGG;
import weka.zoo.ZooModelTest;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

@Log4j2
public class Dl4jCNNExplorerTest extends TestCase {

    private final static int BEN_STILLER_ID = 201;

    private final static String BEN_STILLER_PATH = "src/test/resources/images/ben_stiller.jpg";

    private final static int DOG_ID = 235;

    private final static String DOG_PATH = "src/test/resources/images/dog.jpg";

    /**
     * Test the explorer with VGG16 pretrained on VGGFACE, and a photo of Ben Stiller
     * @throws Exception
     */
    public void testBenStillerVGGFACE() throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jVGG zooModel = new Dl4jVGG();
        zooModel.setVariation(VGG.VARIATION.VGG16);
        zooModel.setPretrainedType(PretrainedType.VGGFACE);
        explorer.setZooModelType(zooModel);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ModelOutputDecoder.ClassmapType.VGGFACE);
        explorer.setModelOutputDecoder(decoder);

        checkPredictionInTopN(explorer, BEN_STILLER_PATH, BEN_STILLER_ID);
    }

    /**
     * Test the explorer with a pretrained ResNet on ImageNet and a photo of a trombone
     * @throws Exception
     */
    public void testDogResNet() throws Exception {
        checkImageNetModel(new KerasResNet(), DOG_PATH, DOG_ID);
    }

    /**
     * Tests the Keras models with a simple dog picture - checks to see which models output correct predictions
     * @throws Exception
     */
    private void checkValidKerasModels() throws Exception {
        List<AbstractZooModel> kerasModels = ZooModelTest.createKerasModels();

        testModelList(kerasModels);
    }

    private void testModelList(List<AbstractZooModel> zooModels) throws Exception {
        List<AbstractZooModel> failedModels = new ArrayList<>();

        for (AbstractZooModel zooModel : zooModels) {
            try {
                checkImageNetModel(zooModel, DOG_PATH, DOG_ID);
            } catch (AssertionError ex) {
                failedModels.add(zooModel);
            }
        }

        // Ideally no models failed the check
        if (failedModels.size() == 0) {
            assertTrue(true);
            return;
        }

        System.err.println(String.format("%d/%d models failed the check:", failedModels.size(), zooModels.size()));
        for (AbstractZooModel zooModel : failedModels) {
            System.err.println(zooModel.getPrettyName());
        }
        fail();
    }

    /**
     * Simply tests a pretrained (on IMAGENET) zoo model
     * @param zooModel
     */
    private void checkImageNetModel(AbstractZooModel zooModel, String imagePath, int expectedClassID) throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        explorer.setZooModelType(zooModel);
        checkPredictionInTopN(explorer, imagePath, expectedClassID);
    }

    private void checkPredictionInTopN(Dl4jCNNExplorer explorer, String imagePath, int expectedClassID) throws Exception {
        try {
            explorer.init();
            explorer.makePrediction(new File(imagePath));

            TopNPredictions topNPredictions = explorer.getCurrentPredictions();

            for (Prediction p : topNPredictions.getTopPredictions()) {
                if (p.getClassID() == expectedClassID) {
                    assertTrue(true);
                    return;
                }
            }
            log.error("Couldn't find correct prediction id in predictions:");
            log.error(topNPredictions.toSummaryString());
            fail();
        } catch (OutOfMemoryError error) {
            log.warn("OutOfMemoryError encountered - please run this test individually to ensure there are no other errors");
            assertTrue(true);
        }
    }
}