# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:17:36 2020

@author: toztel17
"""


# HW2
from bs4 import BeautifulSoup
import os
import urllib 
import re
import numpy as np
import pandas as pd
import seaborn; seaborn.set() # set plot style
web_address = 'https://www.araba.com/otomobil?sayfa=2'
web_page = urllib.request.urlopen(web_address)

soup = BeautifulSoup(web_page.read())
soup.prettify()
price = []
soup.find_all('td')

km = []
i = 2

while i <= 81:
    if i == 44 or i == 66:
        i = i+1
    km_tag = soup.find_all('td')[i]
    km_tag = str(km_tag)
    km_tag = km_tag.replace('<td>\n                                    ','')
    km_tag = km_tag =km_tag.replace('\n                            </td>','')                                
    km_tag = km_tag.replace('.','')
    km.append(int(km_tag))

    i = i+7


re.sub(r'<[^>]+>', '', str(km_tag)) 
price_tag = soup.find_all('td',{'class':'price-column'})
price_tag = re.sub('<td class="price-column">', '', str(price_tag))
price_tag = re.sub('TL</td>', '', price_tag)
price_tag = price_tag.replace('[','')
price_tag = price_tag.replace(']','')
price_tag = price_tag.split(',')


brand_tag = soup.find_all('p', {'class':'row-content-title'})
brand_tag = re.sub('<p class="row-content-title">', '', str(brand_tag))
brand_tag = re.sub('\n</p>', '', brand_tag)
brand_tag = re.sub('</p>', '', brand_tag)
brand_tag = brand_tag.replace('[','')
brand_tag = brand_tag.replace(']','')
brand_tag = brand_tag.split(',')





for i in range(0,len(price_tag)): #len brand_tag idi değiştirdim
    price_tag[i] = price_tag[i].replace(' ','')
    price.append(int(price_tag[i].replace('.','')))

price = np.array(price)

brand = np.array(brand_tag)
data = {'KM':km, 'Price':price,'Brand':brand}
data_to_write = pd.DataFrame(data,columns=['Price', 'KM', 'Brand'])


data_to_write.to_csv(r'C:\Users\toztel17\Desktop\data.csv')

#%%

from lin_reg import lin_reg

brand = brand
brand_num = []
unique_brand = list(set(brand))
#recoding my control variable
if len(brand) == len(unique_brand): #tagging nominal string data with nominal numerical data
   for n in range(0,len(brand)):
     brand_num.append(n)
elif len(brand) > len(unique_brand): #this means there are duplicates
     for i in range(0,len(brand)):
         for ii in range(0,len(unique_brand)):
             if brand[i] == unique_brand[ii]:
                 brand_num.append(unique_brand.index(brand[i]))
                 
#from sklearn import preprocessing #to scale
brand_num = np.array(brand_num)
km = np.array(km).reshape(len(km),1)
price = price.reshape(len(price),1)
#brand_num_scaled = preprocessing.scale(brand_num)
#km_scaled = preprocessing.scale(km)
#price_scaled = preprocessing.scale(price)
#ind_var = np.hstack([km_scaled,np.resize(brand_num_scaled,(len(brand_num),1))])
ind_var = np.hstack([km,np.resize(brand_num,(len(brand_num),1))]) #independent variables. I wasnt sure if i should scale a nominal data (i think i shouldnt as it doesnt have a real mean) so I preffered to go with raw data

#%% safe part
import matplotlib.pyplot as plt

r = lin_reg()
r.fit(price,ind_var)

fig = plt.figure()
ax = plt.axes()
plt.title("Linear relationship between car KM and price")
plt.xlabel("KM")
plt.ylabel("Price");

plt.plot(km,km*r.B[1]+r.B[0],label='predicted values of price')
plt.scatter(km, price, marker='o',label='observed data')
plt.legend();
plt.errorbar(km, km*r.B[1]+r.B[0], xerr=0.5, yerr=r.SE[1],linestyle='-o')
plt.show()