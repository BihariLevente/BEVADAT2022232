# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''

# %%
'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''

# %%
def csv_to_df(path):
    performance_df = pd.read_csv(path)
    return performance_df

#csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv")

# %%
'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''

# %%
def capitalize_columns(df_data):
    df_data_capitalized = df_data.copy()
    df_data_capitalized.columns = [col.upper() if 'e' not in col and 'E' not in col else col for col in df_data_capitalized.columns]
    return df_data_capitalized

#capitalize_columns(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''

# %%
def math_passed_count(df_data) -> int:
    df_copy = df_data.copy()
    count = 0
    for score in df_copy["math score"]:
        if score >= 50:
            count += 1
    return count

#math_passed_count(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''

# %%
def did_pre_course(df_data):
    df_copy = df_data.copy()
    df_did_pre_course = df_copy[df_copy['test preparation course'] == "completed"] 

    return df_did_pre_course

#did_pre_course(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''

# %%
def average_scores(df_data):
    df_copy = df_data.copy()
    df_average_scores = df_copy.groupby("parental level of education")["math score", "reading score", "writing score"].mean()

    return df_average_scores

average_scores(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''

# %%
import random
random.seed(42)

def add_age(df_data):
    df_data_with_age = df_data.copy()
    df_data_with_age["age"] = [random.randint(18,66) for _ in range(len(df_data_with_age))]

    return df_data_with_age

#add_age(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''

# %%
def female_top_score(df_data) -> tuple:
    df_copy = df_data.copy()
    df_sortByScores = df_copy.sort_values(["math score", "reading score", "writing score"], ascending = [False, False, False])
    topScoredFemale = df_sortByScores[df_sortByScores.gender == "female"].iloc[0]
    bestFameleScores = (topScoredFemale["math score"], topScoredFemale["reading score"], topScoredFemale["writing score"])
    return bestFameleScores

female_top_score(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''

# %%
def add_grade(df_data) -> tuple:
    df_data_with_grade = df_data.copy()
    df_data_with_grade["score percentage"] = (df_data_with_grade["math score"] + df_data_with_grade["reading score"] + df_data_with_grade["writing score"]) / 300
    df_data_with_grade["grade"] = df_data_with_grade["score percentage"].apply(lambda score: "A" if score >= 0.9 else "B" if score >= 0.8 else "C" if score >= 0.7 else "D" if score >= 0.6 else "F")
    del df_data_with_grade["score percentage"] 

    return df_data_with_grade

#add_grade(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''

# %%
def math_bar_plot(df_data):
    gender_scores  = df_data.copy().groupby("gender")["math score"].mean()
    fig, ax = plt.subplots()
    ax.bar(gender_scores.index, gender_scores.values)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Math Score")
    plt.title("Average Math Score by Gender")

    return fig

#math_bar_plot(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''

# %%
def writing_hist(df_data):
    new_df = df_data.copy()
    fig, ax = plt.subplots()
    ax.hist(new_df["writing score"])
    #ax.hist(new_df["writing score"], bins=50)
    ax.set_xlabel("Writing Score")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Writing Scores")

    return fig

writing_hist(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''

# %%
def ethnicity_pie_chart(test_df):
    ethnicities = test_df.groupby("race/ethnicity")["race/ethnicity"].count()
    fig, ax = plt.subplots()
    ax.pie(ethnicities.values, labels = ethnicities.index, autopct="%0.01f%%")
    plt.title("Proportion of Students by Race/Ethnicity")
    return fig

ethnicity_pie_chart(csv_to_df("C:/Users/Bihari Levente/Downloads/StudentsPerformance.csv"))

# %%



