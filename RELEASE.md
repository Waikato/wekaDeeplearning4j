## Release Instructions

1. Make sure, that all tests are successful:
```bash
$ ./gradlew test
```
2. Also test the package in the GUI by doing the test cases outlined in [GUI-TEST.md](./GUI-TEST.md)

3. Increase the version and tag the release with the new version. The `version-bump.sh <old-version-tag> <new-version-tag>` script might be helpful.

4. Add a blurb of changes to the `Changes` section in `Description.props`
 
5. Create ZIP files in `./release-files` for each package and generate their sha256 sums with:
```bash
$ ./make-release-files.sh
```
6. [Draft a new GitHub release](https://github.com/Waikato/wekaDeeplearning4j/releases/new), add a changelog and the sha256 sums that were generated during (4) in `release-files/sums.sha256`

7. Upload all ZIP files and sh/ps1 scripts in the `release-files` directory generated in step (3).