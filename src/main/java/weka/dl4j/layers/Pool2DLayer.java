package weka.dl4j.layers;

import java.util.Enumeration;
import java.util.Vector;

import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;

import weka.core.Option;
import weka.core.Utils;
import weka.dl4j.Activation;
import weka.dl4j.Constants;

public class Pool2DLayer extends Layer {

	private static final long serialVersionUID = -699034028619492301L;

	private PoolingType m_poolingType = PoolingType.MAX;

	public PoolingType getPoolingType() {
		return m_poolingType;
	}

	public void setPoolingType(PoolingType poolingType) {
		m_poolingType = poolingType;
	}

	private int m_strideX = 1;

	public int getStrideX() {
		return m_strideX;
	}

	public void setStrideX(int strideX) {
		m_strideX = strideX;
	}

	private int m_strideY = 1;

	public int getStrideY() {
		return m_strideY;
	}

	public void setStrideY(int strideY) {
		m_strideY = strideY;
	}

	private int m_poolSizeX = 0;

	public int getPoolSizeX() {
		return m_poolSizeX;
	}

	public void setPoolSizeX(int filterSizeX) {
		m_poolSizeX = filterSizeX;
	}

	private int m_poolSizeY = 0;

	public int getPoolSizeY() {
		return m_poolSizeY;
	}

	public void setPoolSizeY(int poolSizeY) {
		m_poolSizeY = poolSizeY;
	}

	@Override
	public org.deeplearning4j.nn.conf.layers.Layer getLayer() {
		SubsamplingLayer layer = new SubsamplingLayer.Builder(
				SubsamplingLayer.PoolingType.MAX, new int[] { getPoolSizeX(), getPoolSizeY() } )
				.stride( getStrideX(), getStrideY() )
				.padding(0, 0)
				.build();
		return layer;
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// pool size x
		String tmp = Utils.getOption(Constants.POOL_SIZE_X, options);
		if(!tmp.equals("")) setPoolSizeX(Integer.parseInt(tmp));
		// pool size y
		tmp = Utils.getOption(Constants.POOL_SIZE_Y, options);
		if(!tmp.equals("")) setPoolSizeY(Integer.parseInt(tmp));
		// stride x
		tmp = Utils.getOption(Constants.STRIDE_X, options);
		if(!tmp.equals("")) setStrideX(Integer.parseInt(tmp));
		// stride y
		tmp = Utils.getOption(Constants.STRIDE_Y, options);
		if(!tmp.equals("")) setStrideY(Integer.parseInt(tmp));
		// pool type
		tmp = Utils.getOption(Constants.POOL_TYPE, options);
		if(!tmp.equals("")) setPoolingType( PoolingType.valueOf(tmp.toUpperCase()) );
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		// pool size x
		result.add("-" + Constants.POOL_SIZE_X);
		result.add( "" + getPoolSizeX() );
		// pool size y
		result.add("-" + Constants.POOL_SIZE_Y);
		result.add( "" + getPoolSizeY() );
		// stride x
		result.add( "-" + Constants.STRIDE_X);
		result.add( "" + getStrideX() );
		// stride y
		result.add( "-" + Constants.STRIDE_Y);
		result.add( "" + getStrideY() );
		// mode
		result.add( "-" + Constants.POOL_TYPE);
		result.add( "" + getPoolingType().name().toLowerCase() );
		return result.toArray(new String[result.size()]);
	}

}