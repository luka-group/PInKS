import IPython
import chromedriver_binary
# import sys
# # sys.path.insert(0, '/nas/home/qasemi/miniconda3/envs/web/lib/python3.8/site-packages/chromedriver_binary')
# # sys.path.insert(0, 'miniconda3/envs/web/lib/python3.8/site-packages/chromedriver_binary')
# sys.path.insert(0, '/nas/home/qasemi/miniconda3/envs/web/lib/python3.8/site-packages/chromedriver_binary')

# print(sys.path)

from selenium import webdriver

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
wd = webdriver.Chrome(
    'chromedriver',
    chrome_options=chrome_options)

page_link = 'https://drive.google.com/u/0/uc?id=0B6N7tANPyVeBNmlSX19Ld2xDU1E&export=download'
wd.get(page_link)

IPython.embed()