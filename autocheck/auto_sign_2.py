#! /usr/bin/env python
# coding:utf-8

#导入包
from selenium import webdriver
# from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import random
import time
import _datetime


#加载驱动
# gecko = os.path.normpath(os.path.join(os.path.dirname(__file__), 'geckodriver'))
# binary = FirefoxBinary(r'C:\Program Files\Firefox Developer Edition\firefox.exe')
# browser = webdriver.Firefox(firefox_binary=binary,executable_path=gecko+'.exe')
# path = 'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'
# driver = webdriver.Chrome(executable_path = path)
# driver = webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
firefox_capabilities = DesiredCapabilities.FIREFOX
firefox_capabilities['marionette'] = True
firefox_capabilities['binary'] = 'C:\Program Files\Firefox Developer Edition\firefox.exe'

driver = webdriver.Firefox(capabilities=firefox_capabilities)

#获取网址
driver.get('https://oa.dc66.net/?m=login')
#sleep
sleeptime=random.randint(0,10)
time.sleep(sleeptime)
browser.close()

#模仿用户输入关键字
driver.find_element_by_xpath('//*[@name="adminuser"]').send_keys('garrett')
# 范围时间
localtime = time.localtime(time.time())
print ("本地时间为 :", localtime)
# 获取现在时间
today = _datetime.date.today()
print(today)
print(localtime.tm_mday)

# 判断当前时间是否在范围时间内
if localtime.tm_mday == 12 :
    driver.find_element_by_xpath('//*[@placeholder="请输入密码"]').send_keys('Dezhi951101')
else:
    driver.find_element_by_xpath('//*[@placeholder="请输入密码"]').send_keys('Dezhiwuxing..01')


#sleep
sleeptime=random.randint(0,10)
time.sleep(sleeptime)
#模仿用户点击按钮登陆
driver.find_element_by_xpath('//*[@name="button"]').click()


#sleep
sleeptime=random.randint(30,60)
time.sleep(sleeptime)
#模仿用户点击打卡
driver.find_element_by_xpath('//*[@class="btn btn-success"]').click()

#sleep and cloese
sleeptime=random.randint(30,60)
time.sleep(sleeptime)
browser.close()

