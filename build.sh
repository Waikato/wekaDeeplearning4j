#!/bin/bash

if [ -z $WEKA_HOME ]; then
    echo "make sure WEKA_HOME env variable is set!"
    exit 1
fi

export CLASSPATH=$WEKA_HOME/weka.jar
echo "classpath" $CLASSPATH

DL4J_BUILD_EXPERIMENTAL="0"

if [ $1 == "fresh" ]; then
    echo "removing contents in lib-full and lib"
    rm lib-full/*
    rm lib/*
    mvn clean
fi

# ok, compile all the source code
mvn -Dmaven.test.skip=true install
# clean-up
ant -f build_package.xml clean

if [ DL4J_BUILD_EXPERIMENTAL == "1" ]; then
    # copy only the *necessary* jars from lib-full to lib
    while read p; do
      if [ -e lib-full/$p ]; then
	  cp lib-full/$p lib/$p
      fi
      # we'll probably need all the core jars
      cp lib-full/deeplearning*.jar lib/
      # ...and apache
      cp lib-full/commons*.jar lib/
    done <scripts/find_jars/jars_unique.txt
else
    # just copy all the jars :(
    cp lib-full/*jar lib/
fi


# build the package
ant -f build_package.xml make_package -Dpackage=wekaDl4j

cd dist
java weka.core.WekaPackageManager -install-package wekaDl4j0.0.1.zip
