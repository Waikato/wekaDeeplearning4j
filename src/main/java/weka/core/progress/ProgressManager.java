package weka.core.progress;

import lombok.extern.log4j.Log4j2;

import java.io.Serializable;

// TODO document
// Exposes a few methods in the progress bar implementation
@Log4j2
public class ProgressManager implements Serializable {

    protected AbstractProgressBar progressBar;

    protected boolean runningInGUI = false;

    public ProgressManager(double maxProgress, String progressMessage) {
        init(maxProgress, progressMessage);
    }

    public ProgressManager(String indeterminateProgressMessage) {
        init(-1, indeterminateProgressMessage);
    }

    public ProgressManager() {
        init(-1, "");
    }

    /**
     * Checks the stacktrace for a call to anything in the weka.gui package.
     * If that exists, WEKA was started by the GUIChooser
     */
    public boolean checkIfRunByGUI() {
        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
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

    private void init(double maxProgress, String progressMessage) {
        runningInGUI = checkIfRunByGUI();

        if (runningInGUI) {
            progressBar = new GUIProgressBar(maxProgress, progressMessage);
        } else {
            // Create the text progress bar as we're running it from code/command line
            progressBar = new TextProgressBar(maxProgress, progressMessage);
        }

        if (maxProgress == -1) {
            progressBar.setIndeterminate(true);
        }
    }

    /**
     * Progress Bar passthrough methods
     */
    public void start() {
        progressBar.start();
    }

    public void increment() {
        progressBar.increment();
    }

    public void setProgress(double progress) {
        progressBar.setProgress(progress);
    }

    public void finish() {
        progressBar.finish();
    }

    public void setMaxProgress(double maxProgress) {
        progressBar.setMaxProgress(maxProgress);
    }

    public void setProgressMessage(String progressMessage) {
        progressBar.setProgressMessage(progressMessage);
    }
}
