As per Eibe's suggestion:

http://stackoverflow.com/questions/17528288/how-to-safely-remove-unnecessary-maven-dependencies-in-eclipse

DL4J is pulling in too many jar files, and we don't need all of them for this DL4J package. We can
'roughly' determine what jars we need by running the scripts in this directory.
