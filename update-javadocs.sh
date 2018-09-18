#!/bin/bash
set -x

./gradlew clean javaDoc
git push origin --delete gh-pages
git branch -D gh-pages
git checkout --orphan gh-pages
git rm -rf .
git add build/docs/javadoc
git mv build/docs/javadoc/* ./
git commit -m 'Update javadoc'
git push --set-upstream origin gh-pages
git checkout master

echo "JavaDocs update was successful!"
