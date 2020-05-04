import sys

try:
    name = sys.argv[1]
    print('Hello ' + name)
    sys.stdout.flush()
except:
    print('error')
    sys.stdout.flush()