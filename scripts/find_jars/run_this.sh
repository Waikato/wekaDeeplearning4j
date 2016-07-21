rm jars.txt
rm jars_unique.txt
./train_iris.sh >> jars.txt
./train_mnist_minimal.sh >> jars.txt
cat jars.txt | python compute_set.py > jars_unique.txt
echo "deeplearning4j*.jar" >> jars_unique.txt
