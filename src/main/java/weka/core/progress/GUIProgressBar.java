package weka.core.progress;

// TODO document

import javax.swing.*;
import java.awt.*;

public class GUIProgressBar extends AbstractProgressBar implements Runnable {

    protected JFrame currentFrame;

    protected JProgressBar progressBar;

    protected JLabel progressMessageLabel;

    protected JLabel etaLabel;

    public GUIProgressBar(double maxProgress, String progressMessage) {
        super(maxProgress, progressMessage);
        initGUIElements();
    }

    private void initGUIElements() {
        // Create frame with title Registration Demo
        currentFrame = new JFrame();
        currentFrame.setTitle("Weka Progress Meter");

        progressMessageLabel = new JLabel(getProgressMessage());

        etaLabel = new JLabel(getETAString());

        progressBar = new JProgressBar();
        progressBar.setIndeterminate(m_indeterminate);
        progressBar.setPreferredSize(new Dimension(400, 30));
        progressBar.setMaximum((int) getMaxProgress());

        // Panel to define the layout. We are using GridBagLayout
        JPanel mainPanel = new JPanel();
        GridBagLayout gbL = new GridBagLayout();
        mainPanel.setLayout(gbL);
        mainPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Progress"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5)));

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridx = 0;
        gbC.gridy = 0;
        gbL.setConstraints(progressMessageLabel, gbC);
        mainPanel.add(progressMessageLabel);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridy = 1;
        gbC.gridx = 0;
        gbC.insets = new Insets(0, 0, 20, 0);
        gbL.setConstraints(etaLabel, gbC);
        mainPanel.add(etaLabel);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridy = 2;
        gbC.gridx = 0;
        gbL.setConstraints(progressBar, gbC);
        mainPanel.add(progressBar);

        // Add panel to frame
        currentFrame.add(mainPanel);
    }

    @Override
    protected void onSetProgress() {
        progressBar.setValue((int) m_actualProgress);
    }

    @Override
    protected void onStart() {
        currentFrame.pack();
        currentFrame.setLocationRelativeTo(null);
        currentFrame.setVisible(true);
    }

    @Override
    public void finish() {
        System.err.println("Closing...");
        if (currentFrame != null) {
            currentFrame.dispose();
        }
    }

    @Override
    public void refreshDisplay() {
        SwingUtilities.invokeLater(() -> {
            etaLabel.setText(getETAString());
            currentFrame.pack();
        });
    }

    @Override
    public void run() {
        start();
    }
}
