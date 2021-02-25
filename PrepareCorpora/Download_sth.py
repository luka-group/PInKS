import IPython
# import chromedriver_binary
# import sys
# # sys.path.insert(0, '/nas/home/qasemi/miniconda3/envs/web/lib/python3.8/site-packages/chromedriver_binary')
# # sys.path.insert(0, 'miniconda3/envs/web/lib/python3.8/site-packages/chromedriver_binary')
# sys.path.insert(0, '/nas/home/qasemi/miniconda3/envs/web/lib/python3.8/site-packages/chromedriver_binary')

# print(sys.path)

from selenium import webdriver
# from selenium import .firefox.FirefoxProfile

_profile = webdriver.FirefoxProfile()
_profile.set_preference("browser.download.folderList", 2)
_profile.set_preference("browser.download.dir", "/nas/home/qasemi/CQplus/Outputs/Corpora/OpenWebText")
_profile.set_preference(
    "browser.helperApps.neverAsk.saveToDisk",
    "text/csv,"
    "application/java-archive, "
    "application/x-msexcel,application/excel,"
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
    "application/x-excel,"
    "application/vnd.ms-excel,"
    "image/png,"
    "image/jpeg,"
    "text/html,"
    "text/plain,"
    "application/msword,"
    "application/xml,"
    "application/vnd.microsoft.portable-executable,"
    "application/x-tar, application/x-xz, application/x-gtar,"
    "application/zip,"
    "application/gzip, application/x-gzip, application/x-gtar"
)

_options = webdriver.FirefoxOptions()
_options.profile = _profile
_options.add_argument('--headless')
_options.add_argument('--no-sandbox')
_options.add_argument('--disable-dev-shm-usage')
# wd = webdriver.Chrome(
#     'chromedriver',
#     chrome_options=chrome_options)
wd = webdriver.Firefox(options=_options)

# page_link = 'https://drive.google.com/u/0/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx&export=download'
page_link = 'https://drive.google.com/u/0/uc?id=0B6N7tANPyVeBNmlSX19Ld2xDU1E&export=download'
wd.get(page_link)
print("Start Downloading")
wd.find_element_by_xpath("//*[@id=\"uc-download-link\"]").click()

# wd.get("https://www.seleniumhq.org/download/")
# wd.find_element_by_xpath("/html/body/div[2]/div[2]/p/a[1]").click()

print(f'Done')
# IPython.embed()