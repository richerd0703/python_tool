import re
a = 'group10_100'
b = a.split('-')[0]
s = int(re.findall("\d+", a)[0])
if s > 9:
    print(s)
