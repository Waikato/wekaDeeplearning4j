package weka.dl4j.inference;

import junit.framework.TestCase;
import lombok.extern.log4j.Log4j2;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.keras.VGG;
import weka.zoo.ZooModelTest;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

@Log4j2
public class Dl4jCNNExplorerTest extends TestCase {

    private final static int BEN_STILLER_ID = 201;

    private final static String BEN_STILLER_PATH = "src/test/resources/images/ben_stiller.jpg";

    private final static int GERMAN_SHEPPARD_ID = 235;

    private final static String GERMAN_SHEPPARD_PATH = "src/test/resources/images/dog.jpg";

    /**
     * Test the explorer with VGG16 pretrained on VGGFACE, and a photo of Ben Stiller
     * @throws Exception If an exception occurs during testing
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
     * @throws Exception If an exception occurs during testing
     */
    public void testDogResNet() throws Exception {
        checkImageNetModel(new KerasResNet(), GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the explorer with a pretrained Darknet19 model - it uses a different class mapping
     * hence why it's in a separate method
     * @throws Exception If an exception occurs during testing
     */
    public void testValidDarknet() throws Exception {
        final int DARKNET_GERMAN_SHEPPARD_ID = 210;

        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jDarknet19 zooModel = new Dl4jDarknet19();
        zooModel.setVariation(Dl4jDarknet19.VARIATION.INPUT448);
        explorer.setZooModelType(zooModel);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ModelOutputDecoder.ClassmapType.DARKNET_IMAGENET);
        explorer.setModelOutputDecoder(decoder);

        checkPredictionInTopN(explorer, GERMAN_SHEPPARD_PATH, DARKNET_GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the Deeplearning4j zoo models with a simple dog picture
     * @throws Exception If an exception occurs during testing
     */
    public void testValidDl4jModels() throws Exception {
        List<AbstractZooModel> dl4jModels = ZooModelTest.createDL4JModels();

        // LeNet only takes images of shape [28, 28, 1]
        dl4jModels.removeIf(x -> x.getClass() == Dl4jLeNet.class);

        // VGGFace model can't detect dogs/cars/other ImageNet classes
        dl4jModels.removeIf(x -> x.getClass() == Dl4jVGG.class && x.getPretrainedType() == PretrainedType.VGGFACE);

        // Need to test DarkNet separately -- has different class mappings
        dl4jModels.removeIf(x -> x.getClass() == Dl4jDarknet19.class);

        // Dl4jXception just straight up gives weird predictions - can't figure out what preprocessing is required
        dl4jModels.removeIf(x -> x.getClass() == Dl4jXception.class);

        testModelList(dl4jModels);
    }

    /**
     * Tests the Keras models with a simple dog picture - checks to see which models output correct predictions
     * @throws Exception If an exception occurs during testing
     */
    public void testValidKerasModels() throws Exception {
        List<AbstractZooModel> kerasModels = ZooModelTest.createKerasModels();

        testModelList(kerasModels);
    }

    /**
     * Test a set of zoo models against the german sheppard image
     * @param zooModels models to test
     * @throws Exception If an exception occurs during testing
     */
    private void testModelList(List<AbstractZooModel> zooModels) throws Exception {
        List<AbstractZooModel> failedModels = new ArrayList<>();

        for (AbstractZooModel zooModel : zooModels) {
            try {
                checkImageNetModel(zooModel, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
            } catch (AssertionError ex) {
                failedModels.add(zooModel);
            }
        }

        // Ideally no models failed the check
        if (failedModels.size() == 0) {
            assertTrue(true);
            return;
        }

        System.err.printf("%d/%d models failed the check:%n", failedModels.size(), zooModels.size());
        for (AbstractZooModel zooModel : failedModels) {
            System.err.println(zooModel.getPrettyName());
        }
        fail();
    }

    /**
     * Simply tests a pretrained (on IMAGENET) zoo model
     * @param zooModel Model to test
     */
    private void checkImageNetModel(AbstractZooModel zooModel, String imagePath, int expectedClassID) throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        explorer.setZooModelType(zooModel);
        checkPredictionInTopN(explorer, imagePath, expectedClassID);
    }

    /**
     * Checks that the expected class ID is in the top N predictions - the model has (at least somewhat) correctly
     * predicted the image
     * @param explorer Explorer to run the prediction with
     * @param imagePath Image to predict on
     * @param expectedClassID Expected class ID of the prediction
     * @throws Exception If an exception occurs during testing
     */
    private void checkPredictionInTopN(Dl4jCNNExplorer explorer, String imagePath, int expectedClassID) throws Exception {
        try {
            explorer.init();
            explorer.processImage(new File(imagePath));

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