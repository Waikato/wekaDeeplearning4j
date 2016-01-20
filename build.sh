#!/bin/bash

mvn clean
mvn install
ant clean
ant -f make_package -Dpackage=wekaDl4j
cd dist
java weka.core.WekaPackageManager -install-package wekaDl4j0.0.1.zip
