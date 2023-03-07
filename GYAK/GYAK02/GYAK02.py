# %%
import numpy as np

# %%
#FONTOS!!!

# CSAK OTT LEHET HASZNÁLNI FOR LOOP-OT AHOL A FELADAT KÜLÖN KÉRI!
# [1,2,3,4] --> ezek az értékek np.array-ek. Ahol listát kérek paraméterként ott külön ki fogom emelni!
# Ha végeztél a feladatokkal, akkor notebook-ot alakítsd át .py.
# A FÁJLBAN CSAK A FÜGGVÉNYEK LEGYENEK! (KOMMENTEK MARADHATNAK)

# %%
#Készíts egy függvényt ami létre hoz egy nullákkal teli numpy array-t.
#Paraméterei: mérete (tuple-ként), default mérete pedig legyen egy (2,2)
#Be: (2,2)
#Ki: [[0,0],[0,0]]
#create_array()

# %%
#sizesTuple = (3,3)

def create_array(input_tuple = (2,2)):
    return np.zeros(input_tuple)
    
#create_array(sizesTuple)

# %%
#Készíts egy függvényt ami a paraméterként kapott array-t főátlóját feltölti egyesekkel
#Be: [[1,2],[3,4]]
#Ki: [[1,2],[3,1]]
#set_one()

# %%
#baseArray = np.array([[1,2,4],[3,4,3],[3,3,3]])

def set_one(input_array):
    np.fill_diagonal(input_array,1)
    return input_array

#set_one(baseArray)

# %%
# Készíts egy függvényt ami transzponálja a paraméterül kapott mártix-ot:
# Be: [[1, 2], [3, 4]]
# Ki: [[1, 2], [3, 4]]
# do_transpose()

# %%
#matrix = np.array([[1, 2], [3, 4]])

def do_transpose(input_array):
    return input_array.transpose()

#do_transpose(matrix)

# %%
# Készíts egy olyan függvényt ami az array-ben lévő értékeket N tizenedjegyik kerekíti, ha nincs megadva ez a paraméter, akkor legyen az alapértelmezett a kettő 
# Be: [0.1223, 0.1675], 2
# Ki: [0.12, 0.17]
# round_array()

# %%
#array = np.array([0.1223, 0.1675])

def round_array(matrix , rnd = 2):
    return np.round(matrix,rnd)

#round_array(array,2)

# %%
# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 0 - False-ra, az 1 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ True False False], [ True  True  True], [False False False]]
# bool_array()

# %%
#array = np.array([[1, 0, 0], [1, 1, 1],[0, 0, 0]])

def bool_array(zeroOneArray):
    zeroOneArray = np.array(zeroOneArray,dtype='bool')
    return zeroOneArray

#bool_array(array)


# %%
# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 1 - False-ra az 0 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ True False False], [ True  True  True], [False False False]]
# invert_bool_array()

# %%
#array = np.array([[1, 0, 0], [1, 1, 1],[0, 0, 0]])

def invert_bool_array(zeroOneArray):
    return np.invert(np.array(zeroOneArray,dtype='bool'))

#invert_bool_array(array)

# %%
# Készíts egy olyan függvényt ami a paraméterként kapott array-t kilapítja
# Be: [[1,2], [3,4]]
# Ki: [1,2,3,4]
# flatten()


# %%
#array = np.array([[1, 0, 0], [1, 1, 1],[0, 0, 0]])

def flatten(array):
    return np.ndarray.flatten(array)

#flatten(array)

# %%



