#!/bin/bash

cd package
mvn test -P CPU -Dmaven.test.skip=false
cd ..
