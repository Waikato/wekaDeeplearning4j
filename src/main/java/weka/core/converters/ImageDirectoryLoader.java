package weka.core.converters;

import lombok.extern.log4j.Log4j2;
import weka.core.*;
import weka.gui.ProgrammaticProperty;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.stream.Collectors;


/**
 * Loader for image datasets that are in a folder-organized fashion
 * i.e., image class for an instance is inferred from the folder name it resides in.
 *
 * This format cannot be natively imported into WEKA, so this loader creates the appropriate
 * Instances object which points to the images
 *
 * <!-- options-start -->
 * * Valid options are: <p>
 * *
 * * <pre> -i &lt;directory&gt;
 * *  Top level directory of the image dataset</pre>
 * *
 * * <pre> -name &lt;filename&gt; (default: output.arff)
 * *  Output meta .arff filename</pre>
 * *
 * <!-- options-end -->
 *
 * <!-- globalinfo-start -->
 * <!-- globalinfo-end -->
 */
@Log4j2
public class ImageDirectoryLoader extends AbstractLoader implements
        BatchConverter, IncrementalConverter, CommandlineRunnable, OptionHandler {

    private static final long serialVersionUID = 8956780190928290891L;

    protected File inputDirectory = new File(System.getProperty("user.dir"));

    protected String outputFileName = "";

    @ProgrammaticProperty
    @OptionMetadata(
            displayName = "Input directory",
            description = "Top level directory of the image dataset",
            commandLineParamName = "i",
            commandLineParamSynopsis = "-i <directory>",
            displayOrder = 1
    )
    public File getInputDirectory() { return inputDirectory; }

    public void setInputDirectory(File inputDirectory) { this.inputDirectory = inputDirectory; }

    @ProgrammaticProperty
    @OptionMetadata(
            displayName = "Output File",
            description = "Output meta .arff filename",
            commandLineParamName = "name",
            commandLineParamSynopsis = "-name <filename> (default: output.arff)",
            displayOrder = 2
    )
    public String getOutputFileName() {
        return outputFileName;
    }

    public void setOutputFileName(String outputFileName) {
        this.outputFileName = outputFileName;
    }

    public boolean isImage(String imgName) {
        String lowerName = imgName.toLowerCase();
        return  lowerName.endsWith(".jpg") ||
                lowerName.endsWith(".jpeg") ||
                lowerName.endsWith(".png");
    }

    private String[] removeNullImages(String[] images) {
        return Arrays.stream(images).filter(x -> !(x == null)).toArray(String[]::new);
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
                log.error(String.format("Found non image file: %s, ignoring...", imgName));
                imageNames[i] = null;
                continue;
            }
            imageNames[i] = Paths.get(imgClass, imgName).toString();
        }
        return removeNullImages(imageNames);
    }

    public Instances createDataset() {
        File[] classArr = inputDirectory.listFiles();
        assert classArr != null;
        List<File> classes = Arrays.stream(classArr).filter(File::isDirectory).collect(Collectors.toList());
        List<String> classStrings = classes.stream().map(File::getName).collect(Collectors.toList());

        ArrayList<Attribute> atts = new ArrayList<>(2);
        atts.add(new Attribute("image_path", (List<String>) null));
        atts.add(new Attribute("class", classStrings));

        Instances data = new Instances("image_dataset__" + inputDirectory.getName(), atts, 0);

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

    @Override
    public void preExecution() {}

    @Override
    public void run(Object toRun, String[] options) throws IllegalArgumentException {
        if (!(toRun instanceof ImageDirectoryLoader)) {
            throw new IllegalArgumentException("Object to execute is not a "
                    + "ImageDirectoryLoader!");
        }

        ImageDirectoryLoader loader = (ImageDirectoryLoader) toRun;
        if (options.length > 0) {
            try {
                loader.setOptions(options);

                if (outputFileName.equals("")) {
                    outputFileName = Paths.get(inputDirectory.getAbsolutePath(), "output.arff").toString();
                } else{
                    outputFileName = Paths.get(inputDirectory.getAbsolutePath(), outputFileName).toString();
                }

                Instances dataset = getDataSet();

                // Save the dataset
                ConverterUtils.DataSink.write(outputFileName, dataset);

                log.info("------- SUCCESS -------");
                log.info("Output arff file written to: " + outputFileName);
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
    public void postExecution() {

    }

    @Override
    public Instances getStructure() {
        return null;
    }

    @Override
    public Instances getDataSet() {
        return createDataset();
    }

    @Override
    public Instance getNextInstance(Instances structure) {
        return null;
    }

    @Override
    public void setSource(File dir) throws IOException {
        if (dir == null) {
            throw new IOException("Source directory object is null!");
        }

        inputDirectory = dir;
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IOException("Directory '" + dir + "' not found");
        }
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

        Utils.checkForRemainingOptions(options);
    }

    public static void main(String[] args) {
        ImageDirectoryLoader loader = new ImageDirectoryLoader();
        loader.run(loader, args);
    }
}
