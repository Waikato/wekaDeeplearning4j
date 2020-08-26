package weka.core.progress;

import lombok.extern.log4j.Log4j2;

// TODO document
// TODO implement indeterminate bar

@Log4j2
public class TextProgressBar extends AbstractProgressBar {

    public TextProgressBar(double maxProgress, String progressMessage) {
        super(maxProgress, progressMessage);
    }

    protected int progressBarSize = 60;

    private int currNumDots = 0;

    @Override
    protected void onSetProgress() {
        currNumDots = (int) (m_normalizedProgress * progressBarSize);
    }

    @Override
    public void refreshDisplay() {
        String progressChar = "=";
        String progressRemainingChar = " ";

        System.err.print(String.format("\r%s: [%s%s] %s",
                getProgressMessage(),
                progressChar.repeat(currNumDots),
                progressRemainingChar.repeat(progressBarSize - currNumDots),
                getETAString()));
    }

    @Override
    protected void onStart() {
        System.err.println("\n\n");
    }

    @Override
    public void finish() {
        System.err.println("\nDone!\n");
    }
}
