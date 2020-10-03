@echo off

start brave.exe https://oa.dc66.net/
start brave.exe http://classmate.easydevops.net/
start brave.exe http://git.easydevops.net/
start brave.exe http://kibana.easydevops.net/
start brave.exe https://www.anquanke.com/

ping -n 30 127.0.0.1  
 
@taskkill /f /IM brave.exe
