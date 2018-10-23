## Release Instructions

1. Make sure, that all tests are successful:
```bash
$ ./gradlew test
```

2. Increase the version and tag the release with the new version. The `version-bump.sh <old-version-tag> <new-version-tag>` script might be helpful.

3. Create ZIP files in `./release-files` for each package and generate their sha256 sums with:
```bash
$ ./make-release-files.sh
```

4. [Draft a new GitHub release](https://github.com/Waikato/wekaDeeplearning4j/releases/new), add a changelog and the sha256 sums that were generated during (4) in `release-files/sums.sha256`

5. Upload all ZIP files and sh/ps1 scripts in the `release-files` directory generated in step (3).