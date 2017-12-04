#!/bin/bash

cd package

# Disable global maven options
MAVEN_OPTS=""
mvn test -P GPU -Dmaven.test.skip=false
cd ..
