#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Create a function that decides if a list contains any odd numbers.
#return type: bool
#function name must be: contains_odd
#input parameters: input_list


# In[149]:


list = [2,4,1]

def contains_odd(input_list):
    for x in input_list:
        if (x%2==1):
            return True
    return False
    
contains_odd(list)


# In[52]:


#Create a function that accepts a list of integers, and returns a list of bool.
#The return list should be a "mask" and indicate whether the list element is odd or not.
#(return should look like this: [True,False,False,.....])
#return type: list
#function name must be: is_odd
#input parameters: input_list


# In[107]:


list = [1,3,1,3,2,1]
def is_odd(input_list):
    bool_list = []   
    for x in input_list:
        if (x%2==1):
            bool_list.append(True)
        else:
            bool_list.append(False)
    return bool_list
            
is_odd(list)


# In[69]:


#Create a function that accpects 2 lists of integers and returns their element wise sum. 
#(return should be a list)
#return type: list
#function name must be: element_wise_sum
#input parameters: input_list_1, input_list_2


# In[104]:


input_list_1 = [1,2,1,3,2,1]
input_list_2 = [1,2,1,3,2,1]
def element_wise_sum(input_list_1, input_list_2):
    wise_sum = []
    for idx, x in enumerate(input_list_1):
        wise_sum.append(x+input_list_2[idx])
    return wise_sum
            
element_wise_sum(input_list_1,input_list_2)


# In[105]:


#Create a function that accepts a dictionary and returns its items as a list of tuples
#(return should look like this: [(key,value),(key,value),....])
#return type: list
#function name must be: dict_to_list
#input parameters: input_dict


# In[153]:


thisdict = {
  "FistName": "Levente",
  "LastName": "Bihari",
  "BirthYear": 2001,
  "Sex": "Male"}

def dict_to_list(input_dict):
    tuple_list = []   
    for x, y in input_dict.items():
        tuple_list.append((x,y))
    return tuple_list
            
dict_to_list(thisdict)


# In[ ]:




