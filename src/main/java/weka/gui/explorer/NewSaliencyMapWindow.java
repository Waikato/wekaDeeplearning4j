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

    JLabel targetClassIDLabel = new JLabel("Target Class ID:");
    JTextField targetClassIDInput = new JTextField(3);
    JLabel classNameLabel = new JLabel("  Class Name:");
    JTextField classNameInput = new JTextField();
    JButton patternButton = new JButton("Pattern");

    JLabel saliencyImageLabel = new JLabel();
    JScrollPane scrollPane;
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

    /** The file chooser for selecting model files. */
    protected WekaFileChooser m_FileChooser = new WekaFileChooser(new File(System.getProperty("user.dir")));

    JPanel classSelectorPanel;
    ArrayList<ClassSelector> classSelectors = new ArrayList<>();

    public NewSaliencyMapWindow() {
        setup();
    }

    private void addClassSelector() {
        if (classSelectors.size() == 10) {
            // Limit the size to 10
            return;
        }
        ClassSelector classSelector = new ClassSelector(getCurrentClassMap(), getDefaultClassID());
        classSelectors.add(classSelector);
        classSelectorPanel.add(classSelector);
        thisWindow.pack();
    }

    private void removeClassSelector() {
        if (classSelectors.size() == 1) {
            // Don't go below one class selector
            return;
        }
        ClassSelector lastClassSelector = classSelectors.get(classSelectors.size() - 1);
        classSelectorPanel.remove(lastClassSelector);
        classSelectors.remove(lastClassSelector);
        thisWindow.pack();
    }

    private void clearClassSelectors() {
        classSelectors.clear();
        classSelectorPanel.removeAll();
    }

    private void setup() {
        setLayout(new GridBagLayout());
        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Saliency Map Viewer"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5)));

        gbc.insets = new Insets(5, 5, 5, 5);
        
        gbc.gridx = 0;
        gbc.gridy = 0;
        add(targetClassIDLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 0;
        targetClassIDInput.setMinimumSize(new Dimension(50, 28));
        add(targetClassIDInput, gbc);

        gbc.gridx = 2;
        gbc.gridy = 0;
        add(classNameLabel, gbc);

        gbc.gridx = 3;
        gbc.gridy = 0;
        gbc.weightx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        add(classNameInput, gbc);

        gbc.weightx = 0;
        gbc.fill = GridBagConstraints.NONE;
        gbc.gridx = 4;
        gbc.gridy = 0;
        add(patternButton, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.weighty = 1;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.gridwidth = 5;
        gbc.gridheight = 1;
        var imageLabel = new JLabel(new ImageIcon(DEFAULT_SALIENCY_IMAGE_PATH));
        var scrollPane = new JScrollPane(imageLabel);
//        var panel = new JPanel();
//        panel.setBackground(Color.green);
        add(scrollPane, gbc);

//        gbc.weightx = 1;
//        gbc.fill = GridBagConstraints.HORIZONTAL;
//        gbc.gridx = 0;
//        gbc.gridy = 1;
//        add(addClassButton, gbc);
//
//        gbc.gridx = 1;
//        gbc.gridy = 1;
//        add(removeClassButton, gbc);
//
//        gbc.gridx = 2;
//        gbc.gridy = 1;
//        add(normalizeHeatmapCheckbox, gbc);
//
//        gbc.gridx = 3;
//        gbc.gridy = 1;
//        add(generateButton, gbc);
//
//        gbc.gridx = 4;
//        gbc.gridy = 1;
//        add(saveHeatmapButton, gbc);
//        // Saliency Map Window
//        generateButton.addActionListener(e -> generateSaliencyMap());
//        saveHeatmapButton.addActionListener(e -> saveHeatmap());
//        normalizeHeatmapCheckbox.addActionListener(e -> generateSaliencyMap());
//        addClassButton.addActionListener(e -> addClassSelector());
//        removeClassButton.addActionListener(e -> removeClassSelector());
//
//        // Panel to define the layout. We are using GridBagLayout
//        JPanel mainPanel = new JPanel();
//        GridBagLayout mainLayout = new GridBagLayout();
//        mainPanel.setLayout(mainLayout);
//        mainPanel.setBorder(BorderFactory.createCompoundBorder(
//                BorderFactory.createTitledBorder("Saliency Map Viewer"),
//                BorderFactory.createEmptyBorder(5, 5, 5, 5)));
//
//        // Setup top row in saliency window - class selector UI
//        classSelectorPanel = new JPanel();
//        BoxLayout classSelectorLayout = new BoxLayout(classSelectorPanel, BoxLayout.Y_AXIS);
//        classSelectorPanel.setLayout(classSelectorLayout);
//
//        GridBagConstraints gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.CENTER;
//        gbC.gridx = 0;
//        gbC.gridy = 0;
//        mainLayout.setConstraints(classSelectorPanel, gbC);
//        mainPanel.add(classSelectorPanel);
//
//        // Setup second row - Normalize checkbox and Generate button
//        JPanel secondRow = new JPanel(new GridLayout(1, 5, 30, 5));
//        secondRow.add(addClassButton);
//        secondRow.add(removeClassButton);
//        secondRow.add(normalizeHeatmapCheckbox);
//        secondRow.add(generateButton);
//        secondRow.add(saveHeatmapButton);
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.CENTER;
//        gbC.gridx = 0;
//        gbC.gridy = 1;
//        gbC.insets = new Insets(5, 0, 20, 0);
//        mainLayout.setConstraints(secondRow, gbC);
//        mainPanel.add(secondRow);
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.CENTER;
//        gbC.gridx = 0;
//        gbC.gridy = 2;
//        scrollPane = new JScrollPane(saliencyImageLabel);
//        scrollPane.getVerticalScrollBar().setUnitIncrement(20);
//        scrollPane.setPreferredSize(new Dimension(800, 400));
//        mainLayout.setConstraints(scrollPane, gbC);
//        mainPanel.add(scrollPane);
//
//        // Add panel to frame
//        thisWindow.add(mainPanel);
    }

    private String[] getCurrentClassMap() {
        if (processedExplorer == null) {
            return new String[0];
        } else {
            return processedExplorer.getModelOutputDecoder().getClasses();
        }
    }

    public void open(Dl4jCNNExplorer explorer) {
        var window = new NewSaliencyMapWindow();
        var jf = new JFrame();

        jf.setTitle("WekaDeeplearning4j - Saliency Map Viewer");
        jf.setSize(1000, 700);
        jf.setLocationRelativeTo(null);
        jf.setVisible(true);
        jf.add(window);

        System.err.println(classNameInput.getHeight());
//        this.processedExplorer = explorer;
//
//        // Reset the image
//        try {
//            setSaliencyImage(ImageIO.read(new File(DEFAULT_SALIENCY_IMAGE_PATH)));
//        } catch (IOException ex) {
//            ex.printStackTrace();
//        }
//
//        // Start with a fresh class selector
//        clearClassSelectors();
//        addClassSelector();
//
//        // Set the default class ID in the window
//        normalizeHeatmapCheckbox.setSelected(true);
//
//        thisWindow.pack();
//        thisWindow.setLocationRelativeTo(null);
//        thisWindow.setVisible(true);
    }

    private int getDefaultClassID() {
        if (processedExplorer != null)
            return processedExplorer.getCurrentPredictions().getTopPrediction().getClassID();
        else
            return -1;
    }

    private int[] getTargetClassIDs() {
        return classSelectors.stream().mapToInt(ClassSelector::getTargetClass).toArray();
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
                thisWindow.pack();
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
