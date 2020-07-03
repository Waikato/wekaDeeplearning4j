# Weka Run Test Scripts

The scripts in this folder test a range of different usage scenarios with WekaDeeplearning4j.
These can serve as *usage examples* but more importantly as *automated tests* - they
test package usage from the **command line** as opposed to the included test suite (`src/test/java/weka/*`)
which only tests the package from the **Java API**.

## Usage

Before running these, you will need to build the package (i.e., `./gradlew clean makeMain`) and install it
via the `PackageManager`.

- Move your terminal into this folder: `$ cd weka-run-test-scripts`
- Run a test script
    - An individual script i.e., `$ ./iris.sh`, `$ ./tests/Dl4jMlpClassifier_mnist_default.sh` or;
    - The entire test suite: `./run_all_tests.sh`