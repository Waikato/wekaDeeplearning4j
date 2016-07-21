import sys

ss = set()
for line in sys.stdin:
    line = line.rstrip()
    ss.add(line)

for jar in ss:
    print jar
