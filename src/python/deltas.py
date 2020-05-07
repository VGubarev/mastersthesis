import sys
import datetime

f = open('data/simulator/message_received.log', 'r')
f1 = f.readlines()

# print (f1[0])
prev = datetime.datetime.strptime(f1[0],"%Y-%m-%d %H:%M:%S.%f UTC\n").timestamp()
f1 = f1[1:]
for line in f1:
    now = datetime.datetime.strptime(line,"%Y-%m-%d %H:%M:%S.%f UTC\n").timestamp()
    if (now - prev > 5):
        continue
    output = f"{now - prev:.9f}"
    print (output)
    prev = now
