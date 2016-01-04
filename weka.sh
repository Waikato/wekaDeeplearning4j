#!/bin/bash

WEKA_SVN=/Users/cjb60/github/weka-fresh

#java -Xmx5g -cp build/classes/:$WEKA_HOME/weka.jar:/Users/cjb60/github/weka-fresh/build/testcases/:lib/* weka.gui.GUIChooser
java -cp $WEKA_HOME/weka.jar:$WEKA_SVN/build/testcases -Xmx5g weka.gui.explorer.Explorer datasets/iris.arff
