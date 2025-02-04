# %%
import numpy as np

# %%
#FONTOS!!!

# CSAK OTT LEHET HASZNÁLNI FOR LOOP-OT AHOL A FELADAT KÜLÖN KÉRI!
# [1,2,3,4] --> ezek az értékek np.array-ek. Ahol listát kérek paraméterként ott külön ki fogom emelni!
# Ha végeztél a feladatokkal, akkor notebook-ot alakítsd át .py.
# A FÁJLBAN CSAK A FÜGGVÉNYEK LEGYENEK! (KOMMENTEK MARADHATNAK)

# %%
# Írj egy olyan fügvényt, ami megfordítja egy 2d array oszlopait. Bemenetként egy array-t vár.
# Be: [[1,2],[3,4]]
# Ki: [[2,1],[4,3]]
# column_swap()

# %%
#array = np.array([[1,2,3,4],[5,6,7,8]])

def column_swap(input_array): 
    return np.fliplr(input_array)

#column_swap(array)


# %%
# Készíts egy olyan függvényt ami összehasonlít két array-t és adjon vissza egy array-ben, hogy hol egyenlőek 
# Pl Be: [7,8,9], [9,8,7] 
# Ki: [1]
# compare_two_array()
# egyenlő elemszámúakra kell csak hogy működjön

# %%
#a = np.array([7,8,9])
#b = np.array([9,8,7])

def compare_two_array(aArray, bArray):
    return np.array(np.where(np.equal(aArray, bArray) != False)[0].tolist())

#compare_two_array(a,b)

# %%
# Készíts egy olyan függvényt, ami vissza adja string-ként a megadott array dimenzióit:
# Be: [[1,2,3], [4,5,6]]
# Ki: "sor: 2, oszlop: 3, melyseg: 1"
# get_array_shape()
# 3D-vel még műküdnie kell!, 

# %%
#array = np.array([[1,2,3],[1,2,3]])

def get_array_shape(input_array):
    
    shapes = np.shape(input_array)
    if len(shapes) == 1:
        return "sor: {sor}".format(sor=shapes[0])
    elif len(shapes) == 2:
        return "sor: {sor}, oszlop: {oszlop}".format(sor=shapes[0], oszlop=shapes[1])
    elif len(shapes) == 3:
        return "sor: {sor}, oszlop: {oszlop}, melyseg: {melyseg}".format(sor=shapes[0], oszlop=shapes[1], melyseg=shapes[2])
    else:
        return "error"

#get_array_shape(array)

# %%
# Készíts egy olyan függvényt, aminek segítségével elő tudod állítani egy neurális hálózat tanításához szükséges pred-et egy numpy array-ből. 
# Bementként add meg az array-t, illetve hogy mennyi class-od van. Kimenetként pedig adjon vissza egy 2d array-t, ahol a sorok az egyes elemek. Minden nullákkal teli legyen és csak ott álljon egyes, ahol a bementi tömb megjelöli. 
# Pl. ha 1 van a bemeneten és 4 classod van, akkor az adott sorban az array-ban a [1] helyen álljon egy 1-es, a többi helyen pedig 0.
# Be: [1, 2, 0, 3], 4
# Ki: [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# encode_Y()

# %%
#array = np.array([1, 2, 0, 3])

def encode_Y(input_array, number):
    #encodedArray = np.zeros((number,number))
    #encodedArray[input_array[0:number], input_array[input_array[0:number]]] = 1
    
    encodedArray = np.zeros((len(input_array), number))
    encodedArray[np.arange(len(input_array)), input_array] = 1
    return encodedArray

#encode_Y(array, 4)

# %%
# A fenti feladatnak valósítsd meg a kiértékelését. Adj meg a 2d array-t és adj vissza a decodolt változatát
# Be:  [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# Ki:  [1, 2, 0, 3]
# decode_Y()

# %%
#array = np.array([[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

def decode_Y(input_array):
    return np.where(input_array[0:4] == 1)[1]

#decode_Y(array)

# %%
# Készíts egy olyan függvényt, ami képes kiértékelni egy neurális háló eredményét! Bemenetként egy listát és egy array-t és adja vissza azt az elemet, aminek a legnagyobb a valószínüsége(értéke) a listából.
# Be: ['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6]. # Az ['alma', 'körte', 'szilva'] egy lista!
# Ki: 'szilva'
# eval_classification()

# %%
#fruits = ['alma', 'körte', 'szilva']
#probabilities = np.array([0.2, 0.4, 0.6])

def eval_classification(input_list, input_array):
    index = np.where(input_array == np.max(input_array))
    return input_list[int(index[0])]

#eval_classification(fruits, probabilities)

# %%
# Készíts egy olyan függvényt, ahol az 1D array-ben a páratlan számokat -1-re cseréli
# Be: [1,2,3,4,5,6]
# Ki: [-1,2,-1,4,-1,6]
# replace_odd_numbers()

# %%
#numbers = np.array([1,2,3,4,5,6])

def replace_odd_numbers(input_array):
    input_array[input_array%2 == 1] *= -1 
    return input_array

#replace_odd_numbers(numbers)

# %%
# Készíts egy olyan függvényt, ami egy array értékeit -1 és 1-re változtatja, attól függően, hogy az adott elem nagyobb vagy kisebb a paraméterként megadott számnál.
# Ha a szám kisebb mint a megadott érték, akkor -1, ha nagyobb vagy egyenlő, akkor pedig 1.
# Be: [1, 2, 5, 0], 2
# Ki: [-1, 1, 1, -1]
# replace_by_value()

# %%
#numbers = np.array([1, 2, 5, 0])

def replace_by_value(input_array, limit):
    #a if condition else b
    input_array[input_array < limit] = -1 
    input_array[input_array >= limit] = 1 
    return input_array

#replace_by_value(numbers, 2)

# %%
# Készíts egy olyan függvényt, ami egy array értékeit összeszorozza és az eredményt visszaadja
# Be: [1,2,3,4]
# Ki: 24
# array_multi()
# Ha több dimenziós a tömb, akkor az egész tömb elemeinek szorzatával térjen vissza

# %%
#numbers = np.array([1,2,3,4])

def array_multi(input_array):
    return np.prod(input_array)

#array_multi(numbers)

# %%
# Készíts egy olyan függvényt, ami egy 2D array értékeit összeszorozza és egy olyan array-el tér vissza, aminek az elemei a soroknak a szorzata
# Be: [[1, 2], [3, 4]]
# Ki: [2, 12]
# array_multi_2d()

# %%
#numbers = np.array([[1, 2], [3, 4]])

def array_multi_2d(input_array):
    return np.prod(input_array, axis = 1)

#array_multi_2d(numbers)

# %%
# Készíts egy olyan függvényt, amit egy meglévő numpy array-hez készít egy bordert nullásokkal. Bementként egy array-t várjon és kimenetként egy array jelenjen meg aminek van border-je
# Be: [[1,2],[3,4]]
# Ki: [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
# add_border()

# %%
#numbers = np.array([[1,2],[3,4]])

def add_border(input_array):
    borderedArray = np.pad(input_array, pad_width=1, mode='constant', constant_values=0)
    return borderedArray

#add_border(numbers)

# %%
# A KÖTVETKEZŐ FELADATOKHOZ NÉZZÉTEK MEG A NUMPY DATA TYPE-JÁT!

# %%
# Készíts egy olyan függvényt ami két dátum között felsorolja az összes napot és ezt adja vissza egy numpy array-ben. A fgv ként str vár paraméterként 'YYYY-MM' formában.
# Be: '2023-03', '2023-04'  # mind a kettő paraméter str.
# Ki: ['2023-03-01', '2023-03-02', .. , '2023-03-31',]
# list_days()

# %%
#date1, date2 = '2023-03', '2023-04'

def list_days(start_date, end_date):
    return np.arange(start_date, end_date, dtype='datetime64[D]')

#list_days(date1, date2)

# %%
# Írj egy fügvényt ami vissza adja az aktuális dátumot az alábbi formában: YYYY-MM-DD. Térjen vissza egy 'numpy.datetime64' típussal.
# Be:
# Ki: 2017-03-24
# get_act_date()

# %%
def get_act_date():
    return np.datetime64('today')

#get_act_date()

# %%
# Írj egy olyan függvényt ami visszadja, hogy mennyi másodperc telt el 1970 január 01. 00:02:00 óta. Int-el térjen vissza
# Be: 
# Ki: másodpercben az idó, int-é kasztolva
# sec_from_1970()

# %%
def sec_from_1970():
    currentDate = np.datetime64('now')
    fromDate = '1970-01-01 00:02:00'
    return (np.datetime64(currentDate) - np.datetime64(fromDate)).astype(int)

sec_from_1970()


