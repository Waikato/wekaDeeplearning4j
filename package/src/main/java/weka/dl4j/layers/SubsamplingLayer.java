/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    SubsamplingLayer.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.dl4j.layers;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * A version of DeepLearning4j's SubsamplingLayer that implements WEKA option handling.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 */
public class SubsamplingLayer extends org.deeplearning4j.nn.conf.layers.SubsamplingLayer implements OptionHandler, Serializable {

    /**
     * The ID used to serialize this class.
     */
    private static final long serialVersionUID = -699034028619492301L;

    /**
     * Constructor for setting some defaults.
     */
    public SubsamplingLayer() {
        setLayerName("Subsampling layer");
        setConvolutionMode(ConvolutionMode.Truncate);
        setKernelSize(new int[]{1, 1});
        setStride(new int[]{2, 2});
        setPadding(new int[]{0, 0});
        setPoolingType(org.deeplearning4j.nn.conf.layers.PoolingType.MAX);
        setEps(1e-8);
        setPnorm(1);
    }

    /**
     * Global info.
     *
     * @return string describing this class.
     */
    public String globalInfo() {
        return "A subsampling layer from DeepLearning4J.";
    }

    @OptionMetadata(
            displayName = "layer name",
            description = "The name of the layer (default = Convolutional Layer).",
            commandLineParamName = "name", commandLineParamSynopsis = "-name <string>",
            displayOrder = 0)
    public String getLayerName() {
        return this.layerName;
    }

    public void setLayerName(String layerName) {
        this.layerName = layerName;
    }

    @OptionMetadata(
            displayName = "convolution mode",
            description = "The convolution mode (default = Truncate).",
            commandLineParamName = "mode", commandLineParamSynopsis = "-mode <string>",
            displayOrder = 1)
    public ConvolutionMode getConvolutionMode() {
        return this.convolutionMode;
    }

    public void setConvolutionMode(ConvolutionMode convolutionMode) {
        this.convolutionMode = convolutionMode;
    }

    @OptionMetadata(
            displayName = "eps",
            description = "The value of the eps parameter (default = 1e-8).",
            commandLineParamName = "eps", commandLineParamSynopsis = "-eps <double>",
            displayOrder = 2)
    public double getEps() {
        return super.getEps();
    }

    public void setEps(double e) {
        super.setEps(e);
    }

    @OptionMetadata(
            displayName = "pnorm",
            description = "The value of the pnorm parameter (default = 1).",
            commandLineParamName = "pnorm", commandLineParamSynopsis = "-pnorm <int>",
            displayOrder = 3)
    public int getPnorm() {
        return super.getPnorm();
    }

    public void setPnorm(int p) {
        super.setPnorm(p);
    }

    @OptionMetadata(
            displayName = "number of columns in kernel",
            description = "The number of columns in the kernel (default = 5).",
            commandLineParamName = "kernelSizeX", commandLineParamSynopsis = "-kernelSizeX <int>",
            displayOrder = 4)
    public int getKernelSizeX() {
        return this.kernelSize[0];
    }

    public void setKernelSizeX(int kernelSize) {
        this.kernelSize[0] = kernelSize;
    }


    @OptionMetadata(
            displayName = "number of rows in kernel",
            description = "The number of rows in the kernel (default = 5).",
            commandLineParamName = "kernelSizeY", commandLineParamSynopsis = "-kernelSizeY <int>",
            displayOrder = 5)
    public int getKernelSizeY() {
        return this.kernelSize[1];
    }

    public void setKernelSizeY(int kernelSize) {
        this.kernelSize[1] = kernelSize;
    }

    @ProgrammaticProperty
    public int[] getKernelSize() {
        return this.kernelSize;
    }

    public void setKernelSize(int[] kernelSize) {
        this.kernelSize = kernelSize;
    }

    @OptionMetadata(
            displayName = "number of columns in stride",
            description = "The number of columns in the stride (default = 1).",
            commandLineParamName = "strideX", commandLineParamSynopsis = "-strideX <int>",
            displayOrder = 6)
    public int getStrideX() {
        return this.stride[0];
    }

    public void setStrideX(int stride) {
        this.stride[0] = stride;
    }

    @OptionMetadata(
            displayName = "number of rows in stride",
            description = "The number of rows in the stride (default = 1).",
            commandLineParamName = "strideY", commandLineParamSynopsis = "-strideY <int>",
            displayOrder = 7)
    public int getStrideY() {
        return this.stride[1];
    }

    public void setStrideY(int stride) {
        this.stride[1] = stride;
    }

    @ProgrammaticProperty
    public int[] getStride() {
        return this.stride;
    }

    public void setStride(int[] stride) {
        this.stride = stride;
    }

    @OptionMetadata(
            displayName = "number of columns in padding",
            description = "The number of columns in the padding (default = 0).",
            commandLineParamName = "paddingX", commandLineParamSynopsis = "-paddingX <int>",
            displayOrder = 8)
    public int getPaddingX() {
        return this.padding[0];
    }

    public void setPaddingX(int padding) {
        this.padding[0] = padding;
    }

    @OptionMetadata(
            displayName = "number of rows in padding",
            description = "The number of rows in the padding (default = 0).",
            commandLineParamName = "paddingY", commandLineParamSynopsis = "-paddingY <int>",
            displayOrder = 9)
    public int getPaddingY() {
        return this.padding[1];
    }

    public void setPaddingY(int padding) {
        this.padding[1] = padding;
    }

    @ProgrammaticProperty
    public int[] getPadding() {
        return this.padding;
    }

    public void setPadding(int[] padding) {
        this.padding = padding;
    }

    @OptionMetadata(
            displayName = "pooling type",
            description = "The type of pooling to use (default = MAX; options: MAX, AVG, SUM, NONE).",
            commandLineParamName = "poolingType", commandLineParamSynopsis = "-poolingType <string>",
            displayOrder = 10)
    public org.deeplearning4j.nn.conf.layers.PoolingType getPoolingType() {
        return this.poolingType;
    }

    public void setPoolingType(org.deeplearning4j.nn.conf.layers.PoolingType poolingType) {
        this.poolingType = poolingType;
    }


    @OptionMetadata(
            displayName = "dropout parameter",
            description = "The dropout parameter (default = 0).",
            commandLineParamName = "dropout", commandLineParamSynopsis = "-dropout <double>",
            displayOrder = 11)
    public double getDropOut() {
        return this.dropOut;
    }

    public void setDropOut(double dropOut) {
        this.dropOut = dropOut;
    }


    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        return Option.listOptionsForClass(this.getClass()).elements();
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {

        return Option.getOptions(this, this.getClass());
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        Option.setOptions(options, this, this.getClass());
    }
}
