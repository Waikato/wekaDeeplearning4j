import sys
import os

for line in sys.stdin:
    line = line.rstrip()
    if "from file:" not in line:
        continue
    else:
        jar_file = line.split("from file:")[1]
        jar_file = jar_file[0 : len(jar_file)-1 ] # get rid of the ] character
        print os.path.basename(jar_file)
