#!/usr/bin/env bash

old=$1
new=$2

sed -i.bak -e "0,/${old}/ s/${old}/${new}/" version
sed -i.bak -e "0,/${old}/ s/${old}/${new}/" cuda-scripts/install-cuda-libs.ps1
sed -i.bak -e "0,/${old}/ s/${old}/${new}/" cuda-scripts/install-cuda-libs.sh

git add version -v
git add cuda-scripts/install-cuda-libs.ps1 -v
git add cuda-scripts/install-cuda-libs.sh -v
git commit -m "Version bump v${old} to v${new}"
echo "Adding git tag: $2"
git tag $2
echo "Do not forget to push the new tag."
