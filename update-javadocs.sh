#!/bin/bash
set -x

./gradlew clean javaDoc
mkdir ../temp_jdoc
cp -r build/docs/javadoc/* ../temp_jdoc
git checkout gh-pages
rm -rf *
cp -r ../temp_jdoc/* .
git add .
git commit -m 'Update javadoc'
git push
git checkout master

rm -rf ../temp_jdoc

echo "JavaDocs update was successful!"
