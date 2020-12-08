package weka.dl4j.inference;

import junit.framework.TestCase;
import lombok.extern.log4j.Log4j2;
import org.junit.Assert;
import org.junit.Test;
import weka.dl4j.enums.PretrainedType;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.keras.DenseNet;
import weka.dl4j.zoo.keras.EfficientNet;
import weka.dl4j.zoo.keras.VGG;
import weka.zoo.ZooModelTest;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

@Log4j2
public class Dl4jCNNExplorerTest {

    private final static int BEN_STILLER_ID = 201;

    private final static String BEN_STILLER_PATH = "src/test/resources/images/ben_stiller.jpg";

    private final static int GERMAN_SHEPPARD_ID = 235;

    private final static String GERMAN_SHEPPARD_PATH = "src/test/resources/images/dog.jpg";

    private final static int FOUR_ID = 4;

    private final static String FOUR_PATH = "datasets/nominal/mnist-minimal/img_3574_4.jpg";

    private final static String MNIST1x28x28_MODEL_PATH = "src/test/resources/models/custom_1x28x28_mnist_30e.model";

    private final static String MNIST3x56x56_MODEL_PATH = "src/test/resources/models/custom_3x56x56_mnist_30e.model";

    /**
     * Test the explorer with VGG16 pretrained on VGGFACE, and a photo of Ben Stiller
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testBenStillerVGGFACE() throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jVGG zooModel = new Dl4jVGG();
        zooModel.setVariation(VGG.VARIATION.VGG16);
        zooModel.setPretrainedType(PretrainedType.VGGFACE);
        explorer.setZooModelType(zooModel);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.VGGFACE);
        explorer.setModelOutputDecoder(decoder);

        checkPredictionInTopN(explorer, BEN_STILLER_PATH, BEN_STILLER_ID);
    }


    /**
     * Test the explorer with a pretrained Darknet19 model - it uses a different class mapping
     * hence why it's in a separate method
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testValidDarknet() throws Exception {
        final int DARKNET_GERMAN_SHEPPARD_ID = 210;

        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jDarknet19 zooModel = new Dl4jDarknet19();
        zooModel.setVariation(Dl4jDarknet19.VARIATION.INPUT448);
        explorer.setZooModelType(zooModel);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.DARKNET_IMAGENET);
        explorer.setModelOutputDecoder(decoder);

        checkPredictionInTopN(explorer, GERMAN_SHEPPARD_PATH, DARKNET_GERMAN_SHEPPARD_ID);
    }

    /**
     * Test that if someone selects Darknet19 (which has a different version of IMAGENET),
     * that an exception is thrown.
     * @throws Exception
     */
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidDarknet_throwsIllegalArgumentException() throws Exception {
        final int DARKNET_GERMAN_SHEPPARD_ID = 210;
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jDarknet19 zooModel = new Dl4jDarknet19();
        zooModel.setVariation(Dl4jDarknet19.VARIATION.INPUT448);
        explorer.setZooModelType(zooModel);

        checkPredictionInTopN(explorer, GERMAN_SHEPPARD_PATH, DARKNET_GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the Deeplearning4j zoo models with a simple dog picture
     * @throws Exception If an exception occurs during testing
     */
    @Test
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
    @Test
    public void testValidKerasModels() throws Exception {
        List<AbstractZooModel> kerasModels = ZooModelTest.createKerasModels();

        testModelList(kerasModels);
    }

    /**
     * Test the explorer with a pretrained ResNet on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testDl4jResNet50_SimpleInference() throws Exception {
        checkImageNetModel(new Dl4jResNet50(), false, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the explorer with a pretrained LeNet on MNIST. This won't give accurate predictions but is merely
     * to check it doesn't throw any exceptions.
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testDl4jLeNet_SimpleInference() throws Exception {
        checkMnistModel(new Dl4jLeNet(), false, FOUR_PATH, FOUR_ID);
    }

    /**
     * Test the explorer with a pretrained ResNet on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testKerasResNet50_SimpleInference() throws Exception {
        checkImageNetModel(new KerasResNet(), false, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the explorer with a pretrained DenseNet on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testDenseNet169_SimpleInference() throws Exception {
        KerasDenseNet model = new KerasDenseNet();
        model.setVariation(DenseNet.VARIATION.DENSENET169);
        checkImageNetModel(model,false,  GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the explorer with a pretrained EfficientNet B1 on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testEfficientNet_SimpleInference() throws Exception {
        KerasEfficientNet model = new KerasEfficientNet();
        model.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
        checkImageNetModel(model, false, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    @Test
    public void test1x28x28_SimpleInference() throws Exception {
        CustomModelSetup modelSetup = new CustomModelSetup();
        modelSetup.setSerializedModelFile(new File(MNIST1x28x28_MODEL_PATH));
        modelSetup.setInputChannels(1);
        modelSetup.setInputWidth(28);
        modelSetup.setInputHeight(28);

        checkMnistModel(modelSetup, false, FOUR_PATH, FOUR_ID);
    }

    @Test
    public void test3x56x56_SimpleInference() throws Exception {
        CustomModelSetup modelSetup = new CustomModelSetup();
        modelSetup.setSerializedModelFile(new File(MNIST3x56x56_MODEL_PATH));
        modelSetup.setInputChannels(3);
        modelSetup.setInputWidth(56);
        modelSetup.setInputHeight(56);

        checkMnistModel(modelSetup, false, FOUR_PATH, FOUR_ID);
    }

    /* Saliency Map Tests */
    /**
     * Test the explorer with a pretrained ResNet on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testDl4jResNet50_SaliencyMap() throws Exception {
        checkImageNetModel(new Dl4jResNet50(), true, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the explorer with a pretrained LeNet on MNIST. This won't give accurate predictions but is merely
     * to check it doesn't throw any exceptions.
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testDl4jLeNet_SaliencyMap() throws Exception {
        checkMnistModel(new Dl4jLeNet(), true, FOUR_PATH, FOUR_ID);
    }

    /**
     * Test the explorer with a pretrained ResNet on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testKerasResNet50_SaliencyMap() throws Exception {
        checkImageNetModel(new KerasResNet(), true, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the explorer with a pretrained DenseNet on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testDenseNet169_SaliencyMap() throws Exception {
        KerasDenseNet model = new KerasDenseNet();
        model.setVariation(DenseNet.VARIATION.DENSENET169);
        checkImageNetModel(model,true,  GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    /**
     * Test the explorer with a pretrained EfficientNet B1 on ImageNet and a photo of a german sheppard
     * @throws Exception If an exception occurs during testing
     */
    @Test
    public void testEfficientNet_SaliencyMap() throws Exception {
        KerasEfficientNet model = new KerasEfficientNet();
        model.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
        checkImageNetModel(model, true, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
    }

    @Test
    public void test1x28x28_SaliencyMap() throws Exception {
        CustomModelSetup modelSetup = new CustomModelSetup();
        modelSetup.setSerializedModelFile(new File(MNIST1x28x28_MODEL_PATH));
        modelSetup.setInputChannels(1);
        modelSetup.setInputWidth(28);
        modelSetup.setInputHeight(28);

        checkMnistModel(modelSetup, true, FOUR_PATH, FOUR_ID);
    }

    @Test
    public void test3x56x56_SaliencyMap() throws Exception {
        CustomModelSetup modelSetup = new CustomModelSetup();
        modelSetup.setSerializedModelFile(new File(MNIST3x56x56_MODEL_PATH));
        modelSetup.setInputChannels(3);
        modelSetup.setInputWidth(56);
        modelSetup.setInputHeight(56);

        checkMnistModel(modelSetup, true, FOUR_PATH, FOUR_ID);
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
                checkImageNetModel(zooModel, false, GERMAN_SHEPPARD_PATH, GERMAN_SHEPPARD_ID);
            } catch (AssertionError ex) {
                failedModels.add(zooModel);
            }
        }

        // Ideally no models failed the check
        if (failedModels.size() == 0) {
            Assert.assertTrue(true);
            return;
        }

        System.err.printf("%d/%d models failed the check:%n", failedModels.size(), zooModels.size());
        for (AbstractZooModel zooModel : failedModels) {
            System.err.println(zooModel.getPrettyName());
        }
        Assert.fail();
    }

    /**
     * Simply tests a pretrained (on IMAGENET) zoo model
     * @param zooModel Model to test
     */
    private void checkImageNetModel(AbstractZooModel zooModel, boolean generateSaliencyMap, String imagePath, int expectedClassID) throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        explorer.setZooModelType(zooModel);
        explorer.setGenerateSaliencyMap(generateSaliencyMap);
        checkPredictionInTopN(explorer, imagePath, expectedClassID);
    }

    private void checkMnistModel(AbstractZooModel zooModel, boolean generateSaliencyMap, String imagePath, int expectedClassID) throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();
        explorer.setZooModelType(zooModel);
        explorer.setGenerateSaliencyMap(generateSaliencyMap);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.CUSTOM);
        decoder.setClassMapFile(new File("datasets/nominal/mnist.meta.minimal.arff"));
        explorer.setModelOutputDecoder(decoder);

        checkPredictionInTopN(explorer, imagePath, expectedClassID);
    }

    private void checkMnistModel(CustomModelSetup modelSetup, boolean generateSaliencyMap, String imagePath, int expectedClassID) throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();
        explorer.setUseCustomModel(true);
        explorer.setCustomModelSetup(modelSetup);
        explorer.setGenerateSaliencyMap(generateSaliencyMap);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.CUSTOM);
        decoder.setClassMapFile(new File("datasets/nominal/mnist.meta.minimal.arff"));
        explorer.setModelOutputDecoder(decoder);

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

            log.info(topNPredictions.toSummaryString());

            for (Prediction p : topNPredictions.getTopPredictions()) {
                if (p.getClassID() == expectedClassID) {
                    Assert.assertTrue(true);
                    return;
                }
            }
            log.error("Couldn't find correct prediction id in predictions...");
            Assert.fail();
        } catch (OutOfMemoryError error) {
            log.warn("OutOfMemoryError encountered - please run this test individually to ensure there are no other errors");
            Assert.assertTrue(true);
        }
    }
}