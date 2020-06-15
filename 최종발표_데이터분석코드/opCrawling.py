from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver

# #웃음 사진 크롤링
search = input("검색어 : ")
url = f'https://www.google.com/search?q={quote_plus(search)}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj-qYP8yfvpAhUDat4KHfAyARoQ_AUoAXoECBMQAw&biw=767&bih=712'

driver = webdriver.Chrome('C:/Users/YEOREUM/Downloads/chromedriver_win32/chromedriver.exe')
driver.get(url)

for i in range(500):
    driver.execute_script("window.scrollBy(0,10000)")

html = driver.page_source
soup = BeautifulSoup(html)
img = soup.select('.rg_i.Q4LuWd')
n = 1
imgurl = []

for i in img:
    try:
        imgurl.append(i.attrs["src"])
    except:
        imgurl.append(i.attrs["data-src"])

for i in imgurl:
    urlretrieve(i,"C:/Users/YEOREUM/Desktop/smile/"+search+str(n)+".jpg")
    n += 1
    print(imgurl) 
driver.close()           


#무표정 크롤링
search = input("검색어 : ")
url = f'https://www.google.com/search?q={quote_plus(search)}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj-qYP8yfvpAhUDat4KHfAyARoQ_AUoAXoECBMQAw&biw=767&bih=712'

driver = webdriver.Chrome('C:/Users/YEOREUM/Downloads/chromedriver_win32/chromedriver.exe')
driver.get(url)

for i in range(500):
    driver.execute_script("window.scrollBy(0,10000)")

html = driver.page_source
soup = BeautifulSoup(html)
img = soup.select('.rg_i.Q4LuWd')
n = 1
imgurl = []

for i in img:
    try:
        imgurl.append(i.attrs["src"])
    except:
        imgurl.append(i.attrs["data-src"])

for i in imgurl:
    urlretrieve(i,"C:/Users/YEOREUM/Desktop/notsmile/"+search+str(n)+".jpg")
    n += 1
    print(imgurl) 
driver.close()