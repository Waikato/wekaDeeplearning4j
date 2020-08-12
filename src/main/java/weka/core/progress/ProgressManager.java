package weka.core.progress;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.NotImplementedException;

import java.util.Arrays;
// TODO document

@Log4j2
public class ProgressManager {

    protected AbstractProgressBar progressBar;

//    protected JFrame currentFrame;

    protected boolean runningInGUI = false;

    /**
     * Checks the stacktrace for a call to GenericObjectEditor.
     * If that exists, WEKA was started by the GUIChooser
     */
    public boolean checkIfRunByGUI() {
        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
        System.out.println(Arrays.toString(stack));
        boolean tmpRunningInGUI = false;
        for (StackTraceElement s : stack) {
            if (s.getClassName().contains("weka.gui.")) {
                tmpRunningInGUI = true;
                log.debug("Weka started by GUIChooser");
                break;
            }
        }
        return tmpRunningInGUI;
    }

    private void init(double maxProgress) {
        runningInGUI = checkIfRunByGUI();

        if (runningInGUI) {
            log.error("GUI PROGRESS BAR NOT IMPLEMENTED");
            System.exit(1);
        } else {
            // Create the text progress bar as we're running it from code/command line
            progressBar = new TextProgressBar(maxProgress);
        }
    }

    public ProgressManager(double maxProgress) {
        init(maxProgress);
    }

    public AbstractProgressBar getProgressBar() {
        return progressBar;
    }

    //    public void show() {
//        currentFrame.pack();
//        currentFrame.setVisible(true);
//    }

//    public void hide() {
//        currentFrame.setVisible(false);
//    }

//    private void createEmptyFrame() {
//        currentFrame = new JFrame("WekaDeeplearning4j Notification");
////        currentFrame.setLocation(100, 150);
////
////        JLabel labelM = new JLabel(titleText);
////        labelM.setBounds(50, 50, 200, 30);
////
////        currentFrame.add(labelM);
//
//        JTextField myTextField = new JTextField("Start");
//        // Add the label to the JFrame
//        currentFrame.add(myTextField);
//
////        currentFrame.pack();
////        currentFrame.setLocationRelativeTo(null);
////        currentFrame.setVisible(true);
////        currentFrame.setAutoRequestFocus(true);
////        currentFrame.setAlwaysOnTop(true);
//    }

//    /**
//     * Creates a JFrame with the loading message. To be used while loading the zoo model layer spec
//     * @return reference to JFrame, so it can be destroyed later
//     */
//    public void createLoadingFrame(String titleText, boolean isIndeterminate) {
//        runningInGUI = checkIfRunByGUI();
//
//        runningInGUI = true;
//
//        log.info(titleText);
//
//        if (runningInGUI) {
//            createEmptyFrame();
//
////            final JProgressBar pb = new JProgressBar();
////            pb.setIndeterminate(isIndeterminate);
////            currentFrame.getContentPane().add(pb);
//
//            show();
//        }
//    }

//    /**
//     * Destroy the loading JFrame
//     */
//    public void closePopup() {
//        if (currentFrame != null) {
//            currentFrame.dispose();
//        }
//    }

}
