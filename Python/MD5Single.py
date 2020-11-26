# -*- coding: utf-8 -*-
import hashlib


str = '123456'

h1 = hashlib.md5()

h1.update(str.encode("utf-8"))

print(':' +str)
print(':' +h1.hexdigest())