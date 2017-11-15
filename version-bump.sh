#!/usr/bin/env bash

old=$1
new=$2

sed -i.bak -e "0,/${old}/ s/${old}/${new}/" package/pom.xml
sed -i.bak -e "0,/${old}/ s/${old}/${new}/" package/build_package.xml

git add package/pom.xml -v
git add package/build_package.xml -v
git commit -m "Version bump v${old} to v${new}"
