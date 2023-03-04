#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Create a function that returns with a subsest of a list.
#The subset's starting and ending indexes should be set as input parameters (the list aswell).
#return type: list
#function name must be: subset
#input parameters: input_list,start_index,end_index


# In[332]:


#list = [1,2,3,4,5,6,7,8,9]

def subset(input_list,start_index,end_index):
    subsetList = input_list[start_index:end_index]
    return subsetList
    
#subset(list,2,4)


# In[2]:


#Create a function that returns every nth element of a list.
#return type: list
#function name must be: every_nth
#input parameters: input_list,step_size


# In[39]:


#list = [1,2,3,4,5,6,7,8,9]

def every_nth(input_list,step_size):
    nthList = input_list[::step_size]
    return nthList
    
#every_nth(list,2)


# In[3]:


#Create a function that can decide whether a list contains unique values or not
#return type: bool
#function name must be: unique
#input parameters: input_list


# In[333]:


#list = [2,2,3,3,2,1]

def unique(input_list):
    for i, x in enumerate(input_list):
        existsUnique = True
        for j, y in enumerate(input_list):
            if (i != j):
                if (input_list[i] == input_list[j]):
                    existsUnique = False
        if (existsUnique == True):
            return True
    return False                
#unique(list)


# In[60]:


#Create a function that can flatten a nested list ([[..],[..],..])
#return type: list
#fucntion name must be: flatten
#input parameters: input_list


# In[334]:


#list = [[1,2,3],[4,5,6],[7,8,9]]

def flatten(input_list):
    flattenList = []
    for x in input_list:
        for y in x:
            flattenList.append(y)
    return flattenList
    
#flatten(list)


# In[5]:


#Create a function that concatenates n lists
#return type: list
#function name must be: merge_lists
#input parameters: *args


# In[335]:


#list = [1,2,3,4,5,6,7,8,9]
#list2 = [10]
#list3 = [9,8,7,6,5,4,3,2,1]

def merge_lists(*args):
    merge_List = []
    for arg in args:
        for x in arg:
            merge_List.append(x)
    return merge_List
    
#merge_lists(list, list2, list3)


# In[6]:


#Create a function that can reverse a list of tuples
#example [(1,2),...] => [(2,1),...]
#return type: list
#fucntion name must be: reverse_tuples
#input parameters: input_list


# In[336]:


#list = [(1,2),(3,4),(5,6),(7,8),(9,10)]

def reverse_tuples(input_list):
    reverseList = []
    for tuple in input_list:
        reverseList.append(tuple[::-1])
    return reverseList

#reverse_tuples(list)


# In[7]:


#Create a function that removes duplicates from a list
#return type: list
#fucntion name must be: remove_duplicates
#input parameters: input_list


# In[338]:


#list = [2,6,3,3,2,1,4,5,4,6]

def remove_duplicates(input_list):
    for i, x in enumerate(input_list):
        for j, y in enumerate(input_list):
            if (i != j):
                if (input_list[i] == input_list[j]):
                    input_list.pop(j)
    return input_list            
#remove_duplicates(list)


# In[ ]:


#Create a function that transposes a nested list (matrix)
#return type: list
#function name must be: transpose
#input parameters: input_list


# In[339]:


#list = [[1,2,3],[4,5,6],[7,8,9]]

def transpose(input_list):
    transposeList = [[0 for x in range(len(input_list))] for y in range(len(input_list[0]))] 
    for i, x in enumerate(input_list):
        for j, y in enumerate(x):
            transposeList[j][i] = y        
    return transposeList
    
#transpose(list)


# In[17]:


#Create a function that can split a nested list into chunks
#chunk size is given by parameter
#return type: list
#function name must be: split_into_chunks
#input parameters: input_list,chunk_size


# In[341]:



#list = [0,1,2,3,4,5,6,7,8,9]

def split_into_chunks(input_list,chunk_size):
    onechunkSize = int(len(input_list) / chunk_size) + (len(input_list) % chunk_size > 0)
    splittedList = [[] for i in range(chunk_size)]
    for idx, x in enumerate(input_list):
        splittedList[int(idx/onechunkSize)].append(input_list[idx])
    return splittedList

#split_into_chunks(list,2)


# In[9]:


#Create a function that can merge n dictionaries
#return type: dictionary
#function name must be: merge_dicts
#input parameters: *dict


# In[96]:

"""
namesDict = {
  "FirstName": "Levente",
  "LastName": "Bihari"
}

sexAgeDict = {
  "Sex": "Male",
  "Age": "22"
}

hairEyeDict = {
  "Hair": "Brown",
  "Eye": "Blue"
}
"""

def merge_dicts(*dict):
  merged_Dict = {}
  for dict in dict:
    for d,y in dict.items():
      merged_Dict[d] = y
    
  return merged_Dict
    
#merge_dicts(namesDict, sexAgeDict, hairEyeDict)


# In[10]:


#Create a function that receives a list of integers and sort them by parity
#and returns with a dictionary like this: {"even":[...],"odd":[...]}
#return type: dict
#function name must be: by_parity
#input parameters: input_list


# In[342]:


#list = [1,2,3,4,5,6,7,8,9]

def by_parity(input_list):
    parityDict = {
        "even": [],
        "odd": [],}
    for x in input_list:
        if (x%2==0):
            parityDict["even"].append(x)
        else:
            parityDict["odd"].append(x)
    return parityDict

#by_parity(list)


# In[84]:


#Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
#and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
#in short calculates the mean of the values key wise
#return type: dict
#function name must be: mean_key_value
#input parameters: input_dict


# In[343]:

"""
dict = {
        "some_key": [1,3,2,5],
        "another_key": [2,4,6,1],
        "another_key2": [2,7,9,1]}
"""

def mean_key_value(input_dict):
    meanDict = {}
    for item in input_dict:
        meanDict[item] = sum(input_dict[item]) / len(input_dict[item])
    return meanDict

#mean_key_value(dict)

