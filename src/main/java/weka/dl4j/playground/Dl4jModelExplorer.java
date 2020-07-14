package weka.dl4j.playground;

import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.*;
import weka.dl4j.PretrainedType;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.zoo.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static weka.classifiers.functions.Dl4jMlpClassifier.TAGS_FILTER;

public class Dl4jModelExplorer {

    class Prediction {
        public String classValue;
        public double probability;
    }

    public File imageFile;
    /**
     * The classifier model this filter is based on.
     */
    protected File serializedModelFile = new File(WekaPackageManager.getPackageHome().toURI());

    /**
     * The zoo model to use, if we're not loading from the serialized model file
     */
    protected AbstractZooModel zooModelType = new Dl4jVGG();

    /**
     * Model used for feature extraction
     */
    protected Dl4jMlpClassifier model;

    public void init() throws Exception {
        // Create a single instance iterator
        ImageInstanceIterator iterator = new ImageInstanceIterator();
        iterator.setImagesLocation(imageFile.getParentFile());
        iterator.setWidth(300);
        iterator.setHeight(300);
        iterator.setNumChannels(3);

        String imageName = imageFile.getName();

        List<String> classes = getClasses();

        ArrayList<Attribute> atts = new ArrayList<>(2);
        atts.add(new Attribute("image_path", (List<String>) null));
        atts.add(new Attribute("class", classes));

        Instances data = new Instances("image_dataset_", atts, 0);
        data.setClassIndex(1);

        double[] newInst = new double[2];
        newInst[0] = data.attribute(0).addStringValue(imageName);
        newInst[1] = 1;
        data.add(new DenseInstance(1.0, newInst));

        newInst = new double[2];
        newInst[0] = data.attribute(0).addStringValue(imageName);
        newInst[1] = 2;
        data.add(new DenseInstance(1.0, newInst));

        zooModelType.setPretrainedType(PretrainedType.VGGFACE);

        model.setFilterType(new SelectedTag(, TAGS_FILTER));

        model = Utils.loadModel(data, serializedModelFile, zooModelType, iterator);
    }

    public List<String> getClasses() throws Exception {
        List<String> strings = new ArrayList<>();
        try (FileReader fr = new FileReader("/home/rhys/Documents/git/wdl4j_official/src/main/resources/class-maps/VGGFACE.txt")) {
            try (BufferedReader br = new BufferedReader(fr)) {
                for (String line = br.readLine(); line != null; line = br.readLine()) {
                    if (line.trim().length() == 0)
                        continue;

                    strings.add(line);
                }
            }
        }
        return strings;
    }

    public void makePrediction() throws Exception {
        String imageName = imageFile.getName();

        List<String> classes = getClasses();

        ArrayList<Attribute> atts = new ArrayList<>(2);
        atts.add(new Attribute("image_path", (List<String>) null));
        atts.add(new Attribute("class", classes));

        Instances data = new Instances("image_dataset_", atts, 0);
        data.setClassIndex(1);

        double[] newInst = new double[2];
        newInst[0] = data.attribute(0).addStringValue(imageName);
        newInst[1] = 1;
        data.add(new DenseInstance(1.0, newInst));

        double[][] predictions = model.distributionsForInstances(data);

        for (double[] prediction : predictions) {
            double sum = weka.core.Utils.sum(prediction);
            int maxIndex = weka.core.Utils.maxIndex(prediction);
            double maxValue = prediction[maxIndex];
            String predictedValue = data.classAttribute().value(maxIndex);
            System.out.println(String.format(
                    "Predicted class is %s (index %d) with probability %.3f", predictedValue, maxIndex, maxValue));
        }
    }

    public File getSerializedModelFile() {
        return serializedModelFile;
    }

    public void setSerializedModelFile(File serializedModelFile) {
        this.serializedModelFile = serializedModelFile;
    }

    public AbstractZooModel getZooModelType() {
        return zooModelType;
    }

    public void setZooModelType(AbstractZooModel zooModelType) {
        this.zooModelType = zooModelType;
    }
}
