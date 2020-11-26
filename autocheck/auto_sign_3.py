#! /usr/bin/env python
# coding:utf-8

#导入包
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
# import random
import time
import _datetime


#加载驱动
# driver = webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
driver = webdriver.Firefox()

#获取网址
driver.get('https://oa.dc66.net/?m=login')
#sleep
# sleeptime=random.randint(0,10)
# time.sleep(sleeptime)
# browser.close()


# # 范围时间
# localtime = time.localtime(time.time())
# print ("本地时间为 :", localtime)
# # 获取现在时间
# today = _datetime.date.today()
# print(today)
# print(localtime.tm_mday)