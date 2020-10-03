#最新的selenium配合最新的firefox
import subprocess
from PIL import Image
from PIL import ImageOps
from selenium import webdriver
import time,os,sys
#验证码识别库
import pytesseract


def cleanImage(imagePath):
    image = Image.open(imagePath)   #打开图片
    image = image.point(lambda x: 0 if x<143 else 255)  #处理图片上的每个像素点，使图片上每个点“非黑即白”
    image = image.convert('1')
    borderImage = ImageOps.expand(image,border=1,fill='white')
    borderImage.save(imagePath)


def getAuthCode(driver):
    captcha = driver.find_element_by_id("imgRandom")
    captcha.screenshot("captcha.png")
    #driver.save_screenshot("captcha.jpeg")  # 截屏，并保存图片
    time.sleep(1)
    cleanImage("captcha.png")
    '''
    #使用ubuntu安装的tesseract识别
    p = subprocess.Popen(["tesseract", "captcha.png", "captcha"], stdout= \
        subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    f = open("captcha.txt", "r")

    # Clean any whitespace characters
    captchaResponse = f.read().replace(" ", "").replace("\n", "")

    print("Captcha solution attempt: " + captchaResponse)
    if len(captchaResponse) == 4:
        return captchaResponse
    else:
        return False
    '''
    out = Image.open("captcha.png")
    text = pytesseract.image_to_string(out)
    print("text:" + text)
    return text

bs = webdriver.Firefox()
bs.get('http://url')
time.sleep(1)
authCode = getAuthCode(bs)
failed = True
while failed:
    if authCode:
        print("while loop")
        #因name加密，只能根据转换为xml的path寻找
        username = bs.find_element_by_xpath("//div[@class='logonPanel']/div[2]/div[2]/input[1]")
        username.send_keys('username')
        pwd = bs.find_element_by_xpath("//div[@class='logonPanel']/div[2]/div[3]/input[1]")
        pwd.send_keys('password')
        yzm = bs.find_element_by_xpath("//div[@class='logonPanel']/div[2]/div[4]/input[1]")
        yzm.send_keys(authCode)
        btn_reg = bs.find_element_by_id('loginButton')
        btn_reg.click()
        try:
            time.sleep(3)
            btn_signin = bs.find_element_by_xpath("//a[@class='mr36']")
            btn_signin.click()
            failed = False
        except:
            print("authCode Error:", authCode)
            bs.refresh()
    else:
        failed = True
        bs.refresh()
    time.sleep(3)
    authCode = getAuthCode(bs)