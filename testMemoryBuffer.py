from random import choice
from subprocess import call
from sys import exit

macro = """#define TYPE %s
#define data_type_name \"%s\"
"""

signables = ('char', 'int', 'short', 'long')
unsignables = ('float', 'double', 'wchar_t')

for t in signables + unsignables:
    if t in signables:
        sign = choice(('signed', 'unsigned'))
        t = sign + ' ' + t
    print "testing %s" % t
    f = open("datatype.h", "wt")
    f.write(macro % (t, t))
    f.close()
    r = call(["make"])
    if r:
        print "make exited abnormally for %s" % t
        exit()
    r = call(["./testMemoryBuffer"])
    if r:
        print "testMemoryBuffer exited abnormally for %s" % t
        exit()
print "All tests passed without problem."




    
    
