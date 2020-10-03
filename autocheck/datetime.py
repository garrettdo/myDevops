#! /usr/bin/env python
# coding:utf-8

#导入包
import datetime
from datetime import datetime
import _datetime
import time


# 范围时间
localtime = time.localtime(time.time())
print ("本地时间为 :", localtime)
# 获取现在时间
today = _datetime.date.today()
print(today)
print(localtime.tm_mday)


# 当前时间
# 判断当前时间是否在范围时间内
if localtime.tm_mday == 12 :
 print("Dezhiwuxing..01")
else:
 print("Dezhi951101")
