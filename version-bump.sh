#!/usr/bin/env bash

old=$1
new=$2

sed -i.bak -e "0,/${old}/ s/${old}/${new}/" wekaDeeplearning4jCore/pom.xml
sed -i.bak -e "0,/${old}/ s/${old}/${new}/" wekaDeeplearning4jCore/build_package.xml
