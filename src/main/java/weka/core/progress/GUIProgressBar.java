package weka.core.progress;

// TODO document

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GUIProgressBar extends AbstractProgressBar implements Runnable {

    protected JFrame currentFrame;

    protected JProgressBar progressBar;

    public GUIProgressBar(double maxProgress, String progressMessage) {
        super(maxProgress, progressMessage);
        init();
    }

    private void init() {
        show();
//        currentFrame = new JFrame();
//        currentFrame.setTitle("WekaDeeplearning4j Notification");
    }

    @Override
    protected void onSetProgress() {
        progressBar.setValue((int) m_actualProgress);
    }

    @Override
    public void show() { // TODO refactor this
        // Create frame with title Registration Demo
        currentFrame = new JFrame();
        currentFrame.setTitle("WekaDeeplearning4j Notification");

        // Panel to define the layout. We are using GridBagLayout
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

        JPanel headingPanel = new JPanel();
        GridBagConstraints constr = new GridBagConstraints();
        constr.insets = new Insets(5, 5, 5, 5);

        JLabel headingLabel = new JLabel(m_progressMessage);
        headingPanel.add(headingLabel, constr);

        // Panel to define the layout. We are using GridBagLayout
        JPanel panel = new JPanel(new GridBagLayout());
        // Constraints for the layout
        constr = new GridBagConstraints();
        constr.insets = new Insets(5, 5, 5, 5);
        constr.anchor = GridBagConstraints.NORTH;

        // Set the initial grid values to 0,0
        constr.gridx=0;
        constr.gridy=0;

        progressBar = new JProgressBar();
//        progressBar.setIndeterminate(true);
        progressBar.setPreferredSize(new Dimension(300, 50));
        progressBar.setMaximum((int) getMax());

        panel.add(progressBar, constr);

        mainPanel.add(headingPanel);
        mainPanel.add(panel);

        // Add panel to frame
        currentFrame.add(mainPanel);
        currentFrame.pack();
        currentFrame.setSize(400, 200);
        currentFrame.setLocationRelativeTo(null);

        currentFrame.setVisible(true);
    }

    @Override
    public void finish() {
        if (currentFrame != null) {
            currentFrame.dispose();
        }
    }

    @Override
    public void refreshDisplay() {

    }

    @Override
    public void run() {
        show();
    }
}
