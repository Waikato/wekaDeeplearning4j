## Release Instructions

1. Make sure, that all tests are successful:
```bash
./run-tests.sh
```

2. Tag the release with a new version

3. Create ZIP files for each package and generate their sha256 sums:
```bash
./make-zips.sh
```

4. [Draft a new GitHub release](https://github.com/Waikato/wekaDeeplearning4j/releases/new), add a changelog and the sha256 sums that was generated during (4) in `build/sums.sha256`

5. Upload all ZIP files in the `build` directory generated in step (4) and add the `install-cuda-libs` and `uninstall-cuda-libs` (`.sh`,`.ps1`) from the `cuda-scripts` directory as well.