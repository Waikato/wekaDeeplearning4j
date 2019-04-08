/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Dl4jStringToWord2VecTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.filters.unsupervised.attribute;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;

/**
 * JUnit tests for the Dl4jStringToWord2Vec Filter.
 *
 * @author Steven Lang
 */
public class Dl4jStringToWord2VecTest {

    //    @Test
    public void testReuters() throws Exception {
        final String arffPath = "datasets/text/ReutersCorn-train.arff";
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(arffPath);
        final Instances data = ds.getDataSet();
        Dl4jStringToWord2Vec dl4jw2v = new Dl4jStringToWord2Vec();
        dl4jw2v.setInputFormat(data);
        Instances d = Filter.useFilter(data, dl4jw2v);
    }
}
