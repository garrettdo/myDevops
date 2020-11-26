# -*- coding: utf-8 -*-
import hashlib


def md5value(s):
    md5 = hashlib.md5()
    md5.update(s)
    return md5.hexdigest()
#对字典中的值进行加密
def mdfive():
    sign = {'phone':'18503008588','workAddress':'深圳市南山区科技中一路19号赛百诺大厦'}
    for k, v in sign.items():
        print(v, '->', md5value(v.encode()))

if __name__ == '__main__':
    mdfive()