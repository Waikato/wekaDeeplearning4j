package weka.dl4j;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

public class FileIterationListener implements IterationListener {
	
	private static final long serialVersionUID = 1948578564961956518L;
	
	protected transient PrintWriter m_pw = null;
	
	private int m_numMiniBatches = 0;
	private ArrayList<Double> lossesPerEpoch = new ArrayList<Double>();
	
	public FileIterationListener(String filename, int numMiniBatches) throws Exception {
		super();
		File f = new File(filename);
		if(f.exists()) f.delete();
		System.err.println("Creating debug file at: " + filename);
		m_numMiniBatches = numMiniBatches;
		m_pw = new PrintWriter(new FileWriter(filename, false));
		m_pw.write("loss\n");
	}

	@Override
	public void invoke() { }

	@Override
	public boolean invoked() {
		return false;
	}

	@Override
	public void iterationDone(Model model, int epoch) {	
		lossesPerEpoch.add( model.score() );
		if(lossesPerEpoch.size() == m_numMiniBatches) {
			// calculate mean
			double mean = 0;
			for(double val : lossesPerEpoch) {
				mean += val;
			}
			mean = mean / lossesPerEpoch.size();
			m_pw.write(mean + "\n");
			m_pw.flush();
			lossesPerEpoch.clear();
		}
		//System.err.println(epoch);
	}	

}
