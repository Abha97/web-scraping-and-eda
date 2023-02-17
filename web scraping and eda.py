#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv('Fraud.csv')


# In[37]:



data.head()


# In[38]:


data.info()


# In[39]:


data.describe()


# In[40]:


# Select three indices of your choice you wish to sample from the dataset
indices = [22,154,398]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
print("Chosen samples of wholesale customers dataset:")
display(samples)


# In[41]:


# look at percentile ranks
#pcts = 100. * data.rank(axis=0, pct=True).iloc[indices].round(decimals=3)
pcts = 100. * data.rank(axis=0, pct=True).iloc[indices].round(decimals=3)
# visualize percentiles with heatmap

sns.heatmap(pcts, annot=True, vmin=1, vmax=99, fmt='.1f', cmap='YlGnBu')
plt.title('Percentile ranks of\nsamples\' category spending')
plt.xticks(rotation=45, ha='center');


# Samples: - 0: This customer ranks above the 90th percentile for annual spending amounts in Fresh, Frozen, and the Delicatessen categories. These features along with above average spending for detergents_paper could lead us to believe this customer is a market. Markets generally put an emphasis on having a large variety of fresh foods available and often contain a delicatessen or deli.
# 1: On the opposite side of the spectrum, this customer ranks in the bottom 10th percentile across all product categories. It's highest ranking category is 'Fresh' which might suggest it is a small cafe or similar.
# 
# 2: Our last customer spends a lot in the Fresh and Frozen categories but moreso in the latter. I would suspect this is a wholesale retailer because of the focus on Fresh and Frozen foods.
# 
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some
# amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.

# In[34]:


# Import libraries for Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Remove column Milk
new_data = data.drop('InvoiceNo',axis=1)


# In[35]:


# Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
X_train, X_test, y_train, y_test = train_test_split(new_data, data['InvoiceNo'], test_size=0.25, random_state=1)

# Create a decision tree regressor and fit it to the training set
regressor =  DecisionTreeRegressor(random_state=1)
regressor.fit(X_train, y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print(score)


# In[42]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


df = pd.read_csv("Online Retail.csv")
df.head()


# In[1]:


plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')


# In[2]:


from urllib.request import urlopen


# In[3]:


from bs4 import BeautifulSoup as soup 


# In[5]:


httpObject = urlopen("https://www.flipkart.com/laptops/~laptops-under-rs50000/pr?sid=6bo%2Cb5g")
webdata = httpObject.read()
print(webdata)


# In[7]:


soupdata = soup(webdata)


# In[8]:


dir(soup)


# In[9]:


soupdata


# In[14]:


containers = soupdata.findAll('div' ,{'class':'_2kHMtA'})
print(type(containers),len(containers))


# In[15]:


dir(containers)


# In[16]:


containers[0]


# In[ ]:


for container in containers:
    product

