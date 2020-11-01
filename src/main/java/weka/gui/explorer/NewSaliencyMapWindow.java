package weka.gui.explorer;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FilenameUtils;
import weka.core.progress.ProgressManager;
import weka.dl4j.inference.Dl4jCNNExplorer;
import weka.dl4j.interpretability.AbstractCNNSaliencyMapWrapper;
import weka.gui.ExtensionFileFilter;
import weka.gui.WekaFileChooser;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

@Log4j2
public class NewSaliencyMapWindow extends JPanel {



    Dl4jCNNExplorer processedExplorer;
    /**
     * UI Elements
     */
    JFrame thisWindow = new JFrame("WekaDeeplearning4j - Saliency Map Viewer");

    GridBagConstraints gbc = new GridBagConstraints();

    JLabel saliencyImageLabel = new JLabel();
    Image saliencyImage;
    JCheckBox normalizeHeatmapCheckbox = new JCheckBox("Normalize heatmap");
    JButton addClassButton = new JButton("Add Class");
    JButton removeClassButton = new JButton("Remove Class");
    JButton generateButton = new JButton("Generate");
    JButton saveHeatmapButton = new JButton("Save...");
    private static final String DEFAULT_SALIENCY_IMAGE_PATH = "saliency2.png";
//    private static final String DEFAULT_SALIENCY_IMAGE_PATH = "src/main/resources/placeholderSaliencyMap.png";
//    private static final String DEFAULT_SALIENCY_IMAGE_PATH = "output.png";

    protected FileFilter m_ImageFilter = new ExtensionFileFilter(ExplorerDl4jInference.IMAGE_FILE_EXTENSIONS, "Image files");

    /** The file chooser for saving the image. */
    protected WekaFileChooser m_FileChooser = new WekaFileChooser(new File(System.getProperty("user.dir")));

    ArrayList<NewClassSelector> classSelectors = new ArrayList<>();

    int buttonsRow = 0;
    int targetClassRow = 1;
    int numTargetClasses = 0;
    int imageRow = 20;

    public NewSaliencyMapWindow() {
        oneTimeSetup();
    }

    private void addClassSelector() {
        if (classSelectors.size() == 5) {
            // Limit the size to 10
            return;
        }
        var classSelector = new NewClassSelector(this, targetClassRow + numTargetClasses++);
        classSelectors.add(classSelector);
        packWindow();
    }

    private void removeClassSelector() {
        if (classSelectors.size() == 1) {
            // Don't go below one class selector
            return;
        }
        NewClassSelector lastClassSelector = classSelectors.get(classSelectors.size() - 1);
        lastClassSelector.removeFromParent();
        classSelectors.remove(lastClassSelector);
        packWindow();
    }

    private void packWindow() {
//        var originalDimension = thisWindow.getSize();
        thisWindow.pack();
//        thisWindow.setSize(originalDimension);
    }

    private void setupButtonListeners() {
        generateButton.addActionListener(e -> generateSaliencyMap());
        saveHeatmapButton.addActionListener(e -> saveHeatmap());
        normalizeHeatmapCheckbox.addActionListener(e -> generateSaliencyMap());
        addClassButton.addActionListener(e -> addClassSelector());
        removeClassButton.addActionListener(e -> removeClassSelector());
    }

    private void addControlButtons() {
        gbc.gridy = buttonsRow;

        gbc.gridx = 0;
        gbc.gridwidth = 5;
        var buttonPanel = new JPanel(new GridLayout(1, 5, 30, 5));
        buttonPanel.add(addClassButton);
        buttonPanel.add(removeClassButton);
        buttonPanel.add(normalizeHeatmapCheckbox);
        buttonPanel.add(generateButton);
        buttonPanel.add(saveHeatmapButton);
        add(buttonPanel, gbc);
    }

    private void addScrollableImage() {
        gbc.gridx = 0;
        gbc.gridy = imageRow;
        gbc.weighty = 1;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.gridwidth = 5;
        gbc.gridheight = 1;
        var imageLabel = new JLabel(new ImageIcon(DEFAULT_SALIENCY_IMAGE_PATH));
        var scrollPane = new JScrollPane(imageLabel);
        add(scrollPane, gbc);
    }

    private void oneTimeSetup() {
        setupButtonListeners();

        setLayout(new GridBagLayout());
        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Saliency Map Viewer"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5)));


        gbc.insets = new Insets(5, 5, 5, 5);

        addControlButtons();

        addScrollableImage();
    }

    public String[] getCurrentClassMap() {
        if (processedExplorer == null) {
            return new String[0];
        } else {
            return processedExplorer.getModelOutputDecoder().getClasses();
        }
    }

    public void open(Dl4jCNNExplorer explorer) {
        this.processedExplorer = explorer;

        // Reset the image
        try {
            setSaliencyImage(ImageIO.read(new File(DEFAULT_SALIENCY_IMAGE_PATH)));
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        // Start with a fresh class selector
        addClassSelector();

        // Set the default class ID in the window
        normalizeHeatmapCheckbox.setSelected(true);

        thisWindow.add(this);
        packWindow();
        thisWindow.setLocationRelativeTo(null);
        thisWindow.setVisible(true);
    }

    public int getDefaultClassID() {
        if (processedExplorer != null)
            return processedExplorer.getCurrentPredictions().getTopPrediction().getClassID();
        else
            return -1;
    }

    private int[] getTargetClassIDs() {
        return classSelectors.stream().mapToInt(NewClassSelector::getTargetClass).toArray();
    }

    private void generateSaliencyMap() {
        SwingWorker<Image, Void> worker = new SwingWorker<Image, Void>() {
            @Override
            protected Image doInBackground() {
                ProgressManager manager = new ProgressManager(-1, "Generating saliency map...");
                manager.start();
                int targetClassID = classSelectors.get(0).getTargetClass();
                boolean normalize = normalizeHeatmapCheckbox.isSelected();
                log.info("Generating for class = " + targetClassID);

                AbstractCNNSaliencyMapWrapper wrapper = processedExplorer.getSaliencyMapWrapper();
                wrapper.setTargetClassIDsAsInt(getTargetClassIDs());
                wrapper.setNormalizeHeatmap(normalize);

                processedExplorer.setSaliencyMapWrapper(wrapper);
                Image outputMap = processedExplorer.generateOutputMap();

                manager.finish();

                return outputMap;
            }

            @SneakyThrows
            @Override
            protected void done() {
                super.done();
                setSaliencyImage(get());

                // Pack
                packWindow();
            }
        };
        worker.execute();
    }

    private void setSaliencyImage(Image image) {
        saliencyImage = image;
        ImageIcon icon = new ImageIcon(saliencyImage);
        saliencyImageLabel.setIcon(icon);
        saliencyImageLabel.invalidate();
    }

    private void saveHeatmap() {
        // Prompt the user for a place to save the image to
        m_FileChooser.setFileFilter(m_ImageFilter);

        int returnCode = m_FileChooser.showSaveDialog(this);

        if (returnCode == 1) {
            log.error("User did not select a new image");
            return;
        }
        // Get the image
        File outputFile = m_FileChooser.getSelectedFile();

        String extension = FilenameUtils.getExtension(outputFile.getName());

        try {
            ImageIO.write((RenderedImage) saliencyImage, extension, outputFile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
