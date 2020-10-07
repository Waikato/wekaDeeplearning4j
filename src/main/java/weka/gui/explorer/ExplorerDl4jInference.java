package weka.gui.explorer;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import weka.core.*;

import weka.core.progress.ProgressManager;
import weka.dl4j.inference.Dl4jCNNExplorer;
import weka.dl4j.interpretability.AbstractCNNSaliencyMapWrapper;
import weka.gui.*;
import weka.gui.explorer.Explorer.ExplorerPanel;
import weka.gui.explorer.Explorer.LogHandler;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Explorer panel for the Dl4j Model Inference Window
 * @author - Rhys Compton
 */
@Log4j2
public class ExplorerDl4jInference extends JPanel implements ExplorerPanel, LogHandler {

    /** the parent frame */
    protected Explorer m_Explorer = null;

    /**
     * The loaded instances
     */
    protected Instances m_Instances = null;

    /**
     * The system logger
     */
    protected Logger m_Logger = new SysErrLog();

    /**
     * File path of the currently displayed image
     */
    protected String m_currentlyDisplayedImage = "";

    /*
     * UI Components
     */

    /** Lets the user configure the classifier. */
    protected GenericObjectEditor m_CNNExplorerEditor = new GenericObjectEditor();

    /** The panel showing the current classifier selection. */
    protected PropertyPanel m_ExplorerPropertiesPanel = new PropertyPanel(m_CNNExplorerEditor);

    /** The filename extension that should be used for model files. */
    public static String MODEL_FILE_EXTENSION = ".model";

    /** The filename extension that should be used for PMML xml files. */
    public static String PMML_FILE_EXTENSION = ".xml";

    public static String[] IMAGE_FILE_EXTENSIONS = new String[] {".jpg", ".jpeg", ".png"};

    /** The output area for classification results. */
    protected JTextArea m_OutText = new JTextArea(10, 40);

    /** A panel controlling results viewing. */
    protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);

    /** Button for further output/visualize options. */
    protected JButton m_OpenImageButton = new JButton("Open Image...");

    /** Click to start running the classifier. */
    protected JButton m_startButton = new JButton("Start");

    /** Click to stop running the classifier. */
    protected JButton m_stopButton = new JButton("Stop");

    /** Click to view Saliency Map */
    protected JButton m_saliencyMapButton = new JButton("View Saliency Map...");

    /** A thread that classification runs in. */
    protected Thread m_RunThread;

    /** Filter to ensure only model files are selected. */
    protected FileFilter m_ModelFilter = new ExtensionFileFilter(
            MODEL_FILE_EXTENSION, "Model object files");

    protected FileFilter m_PMMLModelFilter = new ExtensionFileFilter(
            PMML_FILE_EXTENSION, "PMML model files");

    protected FileFilter m_ImageFilter = new ExtensionFileFilter(
            IMAGE_FILE_EXTENSIONS, "Image files");

    /** The file chooser for selecting model files. */
    protected WekaFileChooser m_FileChooser = new WekaFileChooser(new File(
            System.getProperty("user.dir")));

    /**
     * Label used to display the image
     */
    JLabel imageLabel;

    /**
     * Panel the imageLabel is displayed on
     */
    JPanel imagePanel;

    /* Register the property editors we need */
    static {
        GenericObjectEditor.registerEditors();
    }

    protected Dl4jCNNExplorer processedExplorer;

    /**
     * Saliency map UI
     */
    JFrame saliencyMapWindow = new JFrame("WekaDeeplearning4j - Saliency Map Viewer");
    JLabel targetClassIDLabel = new JLabel("Target Class ID:");
    JTextField targetClassIDInput = new JTextField("235");
    JLabel classNameLabel = new JLabel("  Class Name:");
    JTextField classNameInput = new JTextField("Doge");
    JButton generateButton = new JButton("Generate");
    JLabel saliencyImageLabel = new JLabel();
    JCheckBox normalizeHeatmapCheckbox = new JCheckBox("Normalize heatmap");


    /**
     * Sets the Explorer to use as parent frame (used for sending notifications
     * about changes in the data)
     *
     * @param parent the parent frame
     */
    @Override
    public void setExplorer(Explorer parent) {
        m_Explorer = parent;
    }

    /**
     * returns the parent Explorer frame
     *
     * @return the parent
     */
    @Override
    public Explorer getExplorer() {
        return m_Explorer;
    }

    /**
     * Tells the panel to use a new set of instances.
     *
     * @param inst a set of Instances
     */
    @Override
    public void setInstances(Instances inst) {
        m_Instances = inst;
    }

    /**
     * Returns the title for the tab in the Explorer
     *
     * @return the title of this tab
     */
    @Override
    public String getTabTitle() {
        return "Dl4j Inference";
    }

    /**
     * Returns the tooltip for the tab in the Explorer
     *
     * @return the tooltip of this tab
     */
    @Override
    public String getTabTitleToolTip() {
        return "An explorer for trying different trained classification models on individual images.";
    }

    /**
     * Sets the Logger to receive informational messages
     *
     * @param newLog the Logger that will now get info messages
     */
    @Override
    public void setLog(Logger newLog) {
        m_Logger = newLog;
    }

    public ExplorerDl4jInference() {
        super();

        initGUI();
    }

    protected void initGUI() {
        setupCNNExplorerEditor();

        setupOutputText();

        JPanel historyPanel = setupHistoryPanel();

        setupToolTipText();

        setupFileChooser();

        setupButtonListeners();

        setupSaliencyMapWindow();

        JPanel optionsPanel = setupMainButtons();

        JPanel modelOutput = setupOutputPanel();

        JPanel imagePanel = setupImagePanel();

        setupMainLayout(optionsPanel, historyPanel, modelOutput, imagePanel);

    }

    /**
     * Setup the top bar in the window - the explorer editor
     */
    private void setupCNNExplorerEditor() {
        m_CNNExplorerEditor.setClassType(Dl4jCNNExplorer.class);
        m_CNNExplorerEditor.setValue(new Dl4jCNNExplorer());
    }

    /**
     * Setup the output text panel
     */
    private void setupOutputText() {
        // Connect / configure the components
        m_OutText.setEditable(false);
        m_OutText.setFont(new Font("Monospaced", Font.PLAIN, 12));
        m_OutText.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        m_OutText.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if ((e.getModifiers() & InputEvent.BUTTON1_MASK) != InputEvent.BUTTON1_MASK) {
                    m_OutText.selectAll();
                }
            }
        });
    }

    /**
     * Setup the Historical results panel
     * @return Loaded results panel
     */
    private JPanel setupHistoryPanel() {
        JPanel historyHolder = new JPanel(new BorderLayout());
        historyHolder.setBorder(BorderFactory.createTitledBorder("Result list (right-click for separate results panel)"));
        historyHolder.add(m_History, BorderLayout.CENTER);
        m_History.getList().setFixedCellWidth(250);
        m_History.setHandleRightClicks(true);

        // Show the associated image when a results item is clicked on the results panel
        // Showing the appropriate results output is already handled by the ResultHistoryPanel
        m_History.getList().addListSelectionListener(e -> {
            PredictionResult selectedResult = (PredictionResult) m_History.getSelectedObject();
            loadPredictionsFromHistory(selectedResult);
            refreshState();
        });
        return historyHolder;
    }

    /**
     * Setup all tooltip texts
     */
    private void setupToolTipText() {
        m_OpenImageButton.setToolTipText("Open an image for prediction");
        m_startButton.setToolTipText("Run prediction on the image");
        m_saliencyMapButton.setToolTipText("View the saliency map for this image and model");
    }

    /**
     * Setup the file chooser object
     */
    private void setupFileChooser() {
        m_FileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
    }

    /**
     * Setup listeners for the two buttons
     */
    private void setupButtonListeners() {
        m_OpenImageButton.addActionListener(actionEvent -> openNewImage());

        m_predictButton.setEnabled(false);
        m_predictButton.addActionListener(e -> predict());

        m_saliencyMapButton.setEnabled(false);
        m_saliencyMapButton.addActionListener(e -> openSaliencyMapWindow());
    }

    /**
     * Setup the layout for the two buttons
     * @return
     */
    private JPanel setupMainButtons() {
        JPanel optionsPanel = new JPanel();
        GridBagLayout gbL = new GridBagLayout();
        optionsPanel.setLayout(gbL);
        optionsPanel.setBorder(BorderFactory.createEmptyBorder(0, 5, 5, 5));

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 0;
        gbC.gridx = 0;
        gbC.weightx = 100;
        gbC.insets = new Insets(10, 10, 10, 10);
        gbL.setConstraints(m_OpenImageButton, gbC);
        optionsPanel.add(m_OpenImageButton);

        JPanel ssButs = new JPanel();
        ssButs.setBorder(BorderFactory.createEmptyBorder(5,0,5,0));
        ssButs.setLayout(new GridLayout(1, 2, 5, 5));
        ssButs.add(m_startButton);
        ssButs.add(m_stopButton);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 1;
        gbC.gridx = 0;
        gbC.weightx = 100;
        gbC.insets = new Insets(0, 10, 10, 10);
        gbL.setConstraints(ssButs, gbC);
        optionsPanel.add(ssButs);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 2;
        gbC.gridx = 0;
        gbC.weightx = 100;
        gbC.insets = new Insets(0, 10, 10, 10);
        gbL.setConstraints(m_saliencyMapButton, gbC);
        optionsPanel.add(m_saliencyMapButton);

        return optionsPanel;
    }

    /**
     * Set the layout for the text output panel
     * @return Loaded output panel
     */
    private JPanel setupOutputPanel() {
        JPanel outputPanel = new JPanel();
        outputPanel.setBorder(BorderFactory.createTitledBorder("Model output"));
        outputPanel.setLayout(new BorderLayout());
        final JScrollPane js = new JScrollPane(m_OutText);
        outputPanel.add(js, BorderLayout.CENTER);
        js.getViewport().addChangeListener(new ChangeListener() {
            private int lastHeight;

            @Override
            public void stateChanged(ChangeEvent e) {
                JViewport vp = (JViewport) e.getSource();
                int h = vp.getViewSize().height;
                if (h != lastHeight) { // i.e. an addition not just a user scrolling
                    lastHeight = h;
                    int x = h - vp.getExtentSize().height;
                    vp.setViewPosition(new Point(0, x));
                }
            }
        });
        return outputPanel;
    }

    /**
     * Setup the layout of the entire explorer window
     * @param optionsPanel Options panel
     * @param historyPanel History panel
     * @param outputPanel Text output panel
     * @param imagePanel Image display panel
     */
    private void setupMainLayout(JPanel optionsPanel, JPanel historyPanel, JPanel outputPanel, JPanel imagePanel) {
        JPanel mainPanel = new JPanel();
        GridBagLayout mainLayout = new GridBagLayout();
        mainPanel.setLayout(mainLayout);

        // Layout the GUI
        JPanel topPanel = new JPanel();
        topPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Model Settings"),
                BorderFactory.createEmptyBorder(0, 5, 5, 5)));
        topPanel.setLayout(new BorderLayout());
        topPanel.add(m_ExplorerPropertiesPanel, BorderLayout.NORTH);

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 0;
        gbC.gridx = 0;
        mainLayout.setConstraints(optionsPanel, gbC);
        mainPanel.add(optionsPanel);

        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 1;
        gbC.gridx = 0;
        gbC.weightx = 0;
        gbC.weighty = 100;
        mainLayout.setConstraints(historyPanel, gbC);
        mainPanel.add(historyPanel);

        // Setup second column
        JPanel rightPanel = new JPanel();
        GridBagLayout rightLayout = new GridBagLayout();
        rightPanel.setLayout(rightLayout);

        // Add image panel
        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridx = 1;
        gbC.gridy = 0;
        gbC.weightx = 100;
        gbC.weighty = 100;
        gbC.gridheight = 2;
        rightLayout.setConstraints(imagePanel, gbC);
        rightPanel.add(imagePanel);

        // Add output panel
        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 2;
        gbC.gridx = 1;
        gbC.gridheight = 1;
        rightLayout.setConstraints(outputPanel, gbC);
        rightPanel.add(outputPanel);

        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridx = 1;
        gbC.gridy = 0;
        gbC.gridheight = 3;
        gbC.weightx = 100;
        gbC.weighty = 100;
        mainLayout.setConstraints(rightPanel, gbC);
        mainPanel.add(rightPanel);

        setLayout(new BorderLayout());
        add(topPanel, BorderLayout.NORTH);
        add(mainPanel, BorderLayout.CENTER);
    }

    /**
     * Setup the objects for the image display panel
     * @return Loaded image panel
     */
    private JPanel setupImagePanel() {
        imagePanel = new JPanel();
        imagePanel.setBorder(BorderFactory.createTitledBorder("Currently Selected Image"));
        imageLabel = new JLabel("", JLabel.CENTER);
        imagePanel.add(imageLabel, BorderLayout.CENTER);

        return imagePanel;
    }

    /**
     * Launches the "Open Image" popup, saves the image path, and shows it in the image panel
     */
    protected void openNewImage() {
        m_FileChooser.setFileFilter(m_ImageFilter);

        int returnCode = m_FileChooser.showOpenDialog(this);

        if (returnCode == 1) {
            log.error("User did not select a new image");
            return;
        }

        File f = m_FileChooser.getSelectedFile();
        m_currentlyDisplayedImage = f.getAbsolutePath();
        refreshState();
    }

    private String getDefaultClassID() {
        if (processedExplorer != null)
            return "" + processedExplorer.getCurrentPredictions().getTopPrediction().getClassID();
        else
            return "" + -1;
    }

    private String getClassName(int classID) {
        String[] classMap;
        try {
            classMap = processedExplorer.getModelOutputDecoder().getClasses();
        } catch (Exception ex) {
            classMap = new String[]{};
        }
        try {
            return classMap[classID];
        } catch (IndexOutOfBoundsException ex) {
            return null;
        }
    }

    private void updateClassNameInput() {
        String targetClassIDText = targetClassIDInput.getText();
        if (targetClassIDText.isEmpty()) {
            return;
        }
        int classID;
        try {
            classID = Integer.parseInt(targetClassIDText);
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(null,
                    "Error: Please enter a valid integer value", "Error Message",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        String newClassName = getClassName(classID);
        classNameInput.setText(newClassName);
    }

    private void setupSaliencyMapWindow() {
        // Setup the button listeners
        generateButton.addActionListener(e -> generateSaliencyMap());
        targetClassIDInput.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                updateClassNameInput();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                updateClassNameInput();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                updateClassNameInput();
            }
        });

        // Define the UI elements
        targetClassIDInput.setColumns(5);
        targetClassIDInput.setToolTipText("-1 to use max probability class");
        classNameInput.setColumns(40);
        classNameInput.setEditable(false);

        // Panel to define the layout. We are using GridBagLayout
        JPanel mainPanel = new JPanel();
        GridBagLayout gbL = new GridBagLayout();
        mainPanel.setLayout(gbL);
        mainPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Saliency Map Viewer"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5)));

        JPanel configPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        configPanel.add(targetClassIDLabel);
        configPanel.add(targetClassIDInput);
        configPanel.add(classNameLabel);
        configPanel.add(classNameInput);

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridx = 0;
        gbC.gridy = 0;
        gbL.setConstraints(configPanel, gbC);
        mainPanel.add(configPanel);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridx = 0;
        gbC.gridy = 1;
        gbC.insets = new Insets(5, 0, 20, 0);
        gbL.setConstraints(generateButton, gbC);
        mainPanel.add(generateButton);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridx = 0;
        gbC.gridy = 2;
        gbL.setConstraints(saliencyImageLabel, gbC);
        mainPanel.add(saliencyImageLabel);
        
        // Add panel to frame
        saliencyMapWindow.add(mainPanel);
    }

    protected void openSaliencyMapWindow() {
        // Reset the image
        ImageIcon defaultImage = new ImageIcon("src/main/resources/placeholderSaliencyMap.png");
        saliencyImageLabel.setIcon(defaultImage);

        // Set the default class ID in the window
        targetClassIDInput.setText(getDefaultClassID());
        normalizeHeatmapCheckbox.setSelected(true);

        saliencyMapWindow.pack();
        saliencyMapWindow.setLocationRelativeTo(null);
        saliencyMapWindow.setVisible(true);
    }

    private void generateSaliencyMap() {
        SwingWorker<Image, Void> worker = new SwingWorker<Image, Void>() {
            @Override
            protected Image doInBackground() {
                ProgressManager manager = new ProgressManager(-1, "Generating saliency map...");
                manager.start();
                int targetClassID = Integer.parseInt(targetClassIDInput.getText());
                boolean normalize = normalizeHeatmapCheckbox.isSelected();
                log.info("Generating for class = " + targetClassID);

                AbstractCNNSaliencyMapWrapper wrapper = processedExplorer.getSaliencyMapGenerator();
                wrapper.setTargetClassID(targetClassID);
                wrapper.setNormalizeHeatmap(normalize);

                processedExplorer.setSaliencyMapGenerator(wrapper);
                Image outputMap = processedExplorer.generateOutmapToImage();

                manager.finish();

                return outputMap;
            }

            @SneakyThrows
            @Override
            protected void done() {
                super.done();
                Image imageFile = get();
                ImageIcon newImage = new ImageIcon(imageFile);
                saliencyImageLabel.setIcon(newImage);
                saliencyImageLabel.invalidate();

                // Pack
                saliencyMapWindow.pack();
            }
        };
        worker.execute();
    }

    /**
     * Refresh the image panel to show the currently selected image
     */
    protected void refreshState() {
        _refreshImagePanel();
        _refreshButtonsEnabled();
    }

    private void _refreshImagePanel() {
        if (m_currentlyDisplayedImage == null || m_currentlyDisplayedImage == "") {
            return;
        }

        ImageIcon imageIcon = new ImageIcon(m_currentlyDisplayedImage);

        // Does some weird resizing if you use the full width and height of the image panel
        int desiredWidth = imagePanel.getWidth()-100;
        int desiredHeight = imagePanel.getHeight()-100;

        ImageIcon scaledIcon = scaleImage(imageIcon, desiredWidth, desiredHeight);

        imageLabel.setIcon(scaledIcon);
    }

    private void _refreshButtonsEnabled() {
        boolean predictButtonEnabled = !m_currentlyDisplayedImage.equals("");
        m_startButton.setEnabled(predictButtonEnabled);

        boolean saliencyMapEnabled = processedExplorer != null && processedExplorer.getGenerateSaliencyMap();
        m_saliencyMapButton.setEnabled(saliencyMapEnabled);
    }

    /**
     * Scale the image to the desired width and height, maintaining aspect ratio
     * @param icon Raw image
     * @param desiredWidth Desired width to resize to
     * @param desiredHeight Desired height to resize to
     * @return Resized image icon
     */
    private ImageIcon scaleImage(ImageIcon icon, int desiredWidth, int desiredHeight)
    {
        int newWidth = icon.getIconWidth();
        int newHeight = icon.getIconHeight();

        if(newWidth > desiredWidth)
        {
            newWidth = desiredWidth;
            newHeight = (newWidth * icon.getIconHeight()) / icon.getIconWidth();
        }

        if(newHeight > desiredHeight)
        {
            newHeight = desiredHeight;
            newWidth = (icon.getIconWidth() * newHeight) / icon.getIconHeight();
        }

        return new ImageIcon(icon.getImage().getScaledInstance(newWidth, newHeight, Image.SCALE_DEFAULT));
    }

    /**
     * Save the results in the ResultHistoryPanel
     * @param name Run name
     * @param buffer String buffer containing the output
     */
    private void saveResults(String name, StringBuffer buffer) {
        m_History.addResult(name, buffer);
        m_History.addObject(name, savePredictionsForHistory());
    }

    private PredictionResult savePredictionsForHistory() {
        return new PredictionResult(m_currentlyDisplayedImage, processedExplorer);
    }

    private void loadPredictionsFromHistory(PredictionResult result) {
        m_currentlyDisplayedImage = result.imagePath;
        processedExplorer = result.processedExplorer;
    }

    /**
     * Main run method - loads the Dl4jCNNExplorer, runs it on the image, and displays the output
     */
    @SneakyThrows
    private void runInference() {
        ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
        try {
            synchronized (this) {
                m_startButton.setEnabled(false);
            }
            Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());

            m_Logger.statusMessage("Initializing...");
            if (m_Logger instanceof TaskLogger) {
                ((TaskLogger) m_Logger).taskStarted();
            }

            Dl4jCNNExplorer explorer = (Dl4jCNNExplorer) m_CNNExplorerEditor.getValue();
            try {
                explorer.init();
            } catch (Exception ex) {
                log.error("Couldn't initialise model");
                ex.printStackTrace();
                return;
            }

            m_Logger.statusMessage("Processing image");
            explorer.makePrediction(new File(m_currentlyDisplayedImage));
            // Get the predictions
            StringBuffer buffer = new StringBuffer(explorer.getCurrentPredictions().toSummaryString());

            String name = new SimpleDateFormat("HH:mm:ss - ").format(new Date());
            name += explorer.getModelName();
            processedExplorer = explorer;
            saveResults(name, buffer);

            // Show these results
            m_History.setSingle(name);

            synchronized (this) {
                m_Logger.statusMessage("OK");
                if (processedExplorer.getGenerateSaliencyMap())
                    openSaliencyMapWindow();
            }
        } catch (Exception ex) {
            m_Logger.statusMessage("Error occured");
            ex.printStackTrace();
        } finally {
            Thread.currentThread().setContextClassLoader(origLoader);

            synchronized (this) {
                m_startButton.setEnabled(true);
                m_RunThread = null;

                if (m_Logger instanceof TaskLogger) {
                    ((TaskLogger) m_Logger).taskFinished();
                }
            }
        }
    }

    /**
     * Using a separate thread, runs the model prediction
     */
    private void predict() {
        if (m_RunThread == null) {
            m_RunThread = new Thread(this::runInference);
            m_RunThread.setPriority(Thread.MIN_PRIORITY);
            m_RunThread.start();
        }
    }

    private class PredictionResult {
        private String imagePath;
        private Dl4jCNNExplorer processedExplorer;

        public PredictionResult(String imagePath, Dl4jCNNExplorer processedExplorer) {
            this.imagePath = imagePath;
            this.processedExplorer = processedExplorer;
        }
    }
}
