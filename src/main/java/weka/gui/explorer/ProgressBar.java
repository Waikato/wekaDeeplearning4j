package weka.gui.explorer;

import weka.gui.Logger;
import weka.gui.SysErrLog;

public class ProgressBar {

    public ProgressBar() {
        restart();
    }

    protected boolean isRunningInGUI = false;

    /**
     * The system logger
     */
    protected Logger m_Logger = new SysErrLog();

    protected double max = 1;

    protected int progressBarSize = 40;

    private double actualProgress = 0;
    // value between 0 and 1
    private double normalizedProgress = 0;

    private int numDots = 0;

    protected boolean indeterminate = false; // TODO implement

    public void restart() {
        normalizedProgress = 0;
        actualProgress = 0;
    }

    public void setProgress(double progress) {
        actualProgress = progress;
        normalizedProgress = actualProgress / getMax();
        numDots = (int) (normalizedProgress * progressBarSize);

//        System.out.printf("Updated: actualProgress = %.2f, normalizedProgress = %.2f, numDots = %d",
//                actualProgress, normalizedProgress, numDots);
        refresh();
    }

    public void increment() {
        setProgress(actualProgress + 1);
    }

    private void refresh() {
        m_Logger.statusMessage("Progress  |" + ".".repeat(numDots) + " ".repeat(progressBarSize - numDots) + "|");
    }

    public double getMax() {
        return max;
    }

    public void setMax(double max) {
        this.max = max;
    }
}
