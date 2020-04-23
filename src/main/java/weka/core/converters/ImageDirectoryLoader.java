package weka.core.converters;

import weka.core.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.stream.Collectors;

public class ImageDirectoryLoader extends AbstractLoader implements
        BatchConverter, IncrementalConverter, CommandlineRunnable, OptionHandler {

    private static final long serialVersionUID = 8956780190928290891L;

    protected String inputDirectory = "";

    protected String outputFile = "";

    @OptionMetadata(
            displayName = "Input directory",
            description = "Top level directory of the image dataset",
            commandLineParamName = "i",
            commandLineParamSynopsis = "-i <directory>"
    )
    public String getInputDirectory() { return inputDirectory; }

    public void setInputDirectory(String inputDirectory) { this.inputDirectory = inputDirectory; }

    @OptionMetadata(
            displayName = "Output File",
            description = "Output meta .arff file",
            commandLineParamName = "o",
            commandLineParamSynopsis = "-o <filepath> (default: inputDir/output.arff)"
    )
    public String getOutputFile() {
        return outputFile;
    }

    public void setOutputFile(String outputFile) {
        this.outputFile = outputFile;
    }

    public boolean isImage(String imgName) {
        return imgName.toLowerCase().endsWith(".jpg") && !imgName.toLowerCase().endsWith(".png");
    }

    /**
     * Appends the folder that the image is in, to the image path
     * @param folder
     * @return Image names (including the folder they're in)
     */
    public String[] fileListForFolder(File folder) {
        String[] imageNames = folder.list();
        String imgClass = folder.getName();
        for (int i = 0; i < imageNames.length; i++) {
            String imgName = imageNames[i];
            if (!isImage(imgName)) {
                System.err.println(String.format("Found non image file: %s, ignoring...", imgName));
                continue;
            }
            imageNames[i] = Paths.get(imgClass, imgName).toString();
        }
        return imageNames;
    }

    public Instances createDataset() throws Exception {
        File dir = new File(inputDirectory);
        File[] classArr = dir.listFiles();
        assert classArr != null;
        List<File> classes = Arrays.stream(classArr).filter(File::isDirectory).collect(Collectors.toList());
        List<String> classStrings = classes.stream().map(File::getName).collect(Collectors.toList());

        ArrayList<Attribute> atts = new ArrayList<>(2);
        atts.add(new Attribute("image_path", (List<String>) null));
        atts.add(new Attribute("class", classStrings));

        Instances data = new Instances("image_dataset__" + dir.getName(), atts, 0);

        for (File imgClass : classes) {
            String[] imgs = fileListForFolder(imgClass);
            for (String img : imgs) {
                double[] newInst = new double[2];
                newInst[0] = data.attribute(0).addStringValue(img);
                newInst[1] = classStrings.indexOf(imgClass.getName());
                data.add(new DenseInstance(1.0, newInst));
            }
        }
        return data;
    }

    private boolean unusedOptions(String[] options) {
        return Arrays.stream(options).anyMatch(x -> !x.equals(""));
    }

    @Override
    public void preExecution() throws Exception {}

    @Override
    public void run(Object toRun, String[] options) throws Exception {
        ImageDirectoryLoader loader = (ImageDirectoryLoader) toRun;
        if (options.length > 0) {
            try {
                loader.setOptions(options);
                Instances dataset = loader.createDataset();

                // Save the dataset
                ConverterUtils.DataSink.write(outputFile, dataset);

                System.out.println("------- SUCCESS -------");
                System.out.println("Output arff file written to: " + outputFile);
            } catch (Exception e) {
                e.printStackTrace();
                printInfo();
            }
        } else {
            printInfo();
        }
    }

    private void printInfo() {
        System.err.println("\nUsage:\n" + "\tImageDirectoryLoader [options]\n"
                + "\n" + "Options:\n");

        Enumeration<Option> enm =
                ((OptionHandler) new ImageDirectoryLoader()).listOptions();
        while (enm.hasMoreElements()) {
            Option option = enm.nextElement();
            System.err.println(option.synopsis());
            System.err.println(option.description());
        }
    }

    @Override
    public void postExecution() throws Exception {

    }

    @Override
    public Instances getStructure() throws IOException {
        return null;
    }

    @Override
    public Instances getDataSet() throws IOException {
        return null;
    }

    @Override
    public Instance getNextInstance(Instances structure) throws IOException {
        return null;
    }

    @Override
    public String getRevision() {
        return null;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {
        return Option.listOptionsForClassHierarchy(this.getClass(), AbstractFileLoader.class).elements();
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        return Option.getOptionsForHierarchy(this, AbstractFileLoader.class);
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        Option.setOptionsForHierarchy(options, this, AbstractFileLoader.class);

        if (outputFile.equals("")) {
            outputFile = Paths.get(inputDirectory, "output.arff").toString();
        }

        Utils.checkForRemainingOptions(options);
//        if (unusedOptions(options))
//            throw new Exception("Invalid arguments: " + Arrays.toString(options));
    }
}
