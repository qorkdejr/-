# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:09:47 2024

@author: user
"""

    import pandas as pd
    red_df= pd.read_csv('C:/Users/kmj/MY_Python/7장_data/winequality-red.csv',sep=';',header = 0, emgine = 'python')
    white_df= pd.read_csv('C:/Users/kmj/MY_Python/7장_data/winequality-white.csv',sep=';',header = 0, emgine = 'python')
    red_df.to_csv('C:/Users/kmj/MY_Python/7장_data/winequality-red2.csv',index = False)
    white_df.to_csv('C:/Users/kmj/MY_Python/7장_data/winequality-white2.csv',index = False)
    
    red_df.head()
    red_df.insert(0, column = 'type', value = 'red')
    red_df.head()
    red_df.shape
    white_df.head()
    white_df.insert(0, column = 'type', value = 'white')
    white_df.head()
    white_df.shape
    wine = pd.concat([red_df, white_df])
    wine.shape
    wine.to_csv('C:/Users/kmj/MY_Python/7장_data/wine.csv', index = False)
    
    
    #데이터 탐색
    print(wine.info())
    wine.columns = wine.columns.str.replace(' ','_')
    wine.head()
    wine.describe()
    sorted(wine.quality..unique())
    wine.quality.value_counts()

    #데이터 모델링
    wine.groupby('type')['quality'].describe()
    wine.groupby('type')['quality'].mean()
    wine.groupby('type')['quality'].std()
    wine.groupby('type')['quality'].agg(['mean', 'std'])

    #t-검정
    from scipy import stats
    from statsmodels.formula.api import ols, glm
    red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
    white_wine_quality = white.loc[wine['type'] == 'white', 'quality']
    stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var = False)
    Rformula = 'quality ~ fixed_acidity + colatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + total_sulfur_doxide + density + pH + sulphates + alcoho'
    regression_result = ols(Rformula, data=wine).fit()
    regression_result.summary()

    #회귀 분석 샘플 예측
    sample1 = wine[wine.columns.difference(['quality','type'])]
    sample1 = sample1[0:5][:]
    sample1_predict = regression_result.predict(sample1)
    sample1_predict
    wine[0:5]['quality']
    sample2 = pd.DataFrame(data, columns=sample1.columns)
    sample2
    sample2_predict = regression_result.predict(sample2)
    sample2_predict

    #결과시각화
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('dark')
    sns.distplot(red_wine_quality, kde = True, color ="red", label = 'red wine')
    sns.distplot(white_wine_quality, kde = True, label = 'white wine')
    plt.title("Quality of Wine Type")
    plt.legend()
    plt.show()

    import statsmodels.api as sm
    others = list(set(wine.columns).difference(set(["quality", "fixed_acidity"])))
    p, resids = sm.graphics.plot_partregress("quality", "fixed_acidity",others, data = wine  , ret_coords = True)
    plt.show()
    fig= plt.figure(figsize = (8, 13))
    sm.graphics.plot_partregress_grid(regression_result, fig = fig)
    plt.show()
    
    #데이터 수집
    import seaborn as sns
    import pandas as pd
    titanic = sns.load_dataset("titanic")
    titanic.to_csv('C:/Users/kmj/MY_Python/7장_data/titanic.csv', index = False)
    
    titanic.isnull().sum()
    titanic['age'] = titanic['age'].fillna(titanic['age'].median())
    titanic['embarked'].value_counts()
    titanic['embarked']=titanic['embarked'].fillna('S')
    titanic['embark_town'].value_counts()
    titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')
    titanic['deck'].value_counts()
    titanic['deck']=titanic['deck'].fillna('C')
    titanic.isnull().sum()
    titanic.info()
    titanic.survived.value_counts()
    
    #차트를 그려 데이터 탐색
    import matplotlib.pyplot as plt
    f,ax = plt.subplots(1,2, figsize = (10,5))
    titanic['survived'][titanic['sex']=='male'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow= True)
    titanic['survived'][titanic['sex']=='female'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow= True)
    ax[0].set_title('Survived (Male)')
    ax[0].set_title('Survived (feMale)')
    plt.show()\
    sns.countplot(x= 'pclass', hue= 'survived',data = titanic)
    plt.title('Pclass vs Survived')
    plt.show()
    titanic_corr = titanic.corr(method ='pearson')
    titanic_corr
    titanic_corr.to_csv('C:/Users/kmj/MY_Python/7장_data/titanic_corr.csv',index = False)
    titanic['survived'].corr(titanic['adult_male'])
    titanic['survived'].corr(titanic['fare'])
    sns.pairplot(titanic, hue = 'survived')
    plt.show()
    
    sns.catplot(x= 'pclass', y='survived',hue= 'sex', data = titanic, kind = 'point')
    plt.show()
    def category_age(x):
        if x < 10:
            return 0
        elif x <20:
            return 1
        elif x < 30:
            return 2
        elif x < 40:
            return 3
        elif x < 50:
            return 4
        elif x < 60:
            return 5
        elif x < 70:
            return 6
        else:
            return 7
    titanic['age2'] = titanic['age'].apply(category_age)
    titanic['sex'] = titanic['sex'].map({'male':1, 'female':0})
    titanic['family'] = titanic['shbsp'] + titanic['parch'] + 1
    titanic.to_csv('C:/Users/kmj/MY_Python/7장_data/titanic3.csv', index = False)
    heatmap_data=titanic[['survived','sex', 'age2', 'family','pclass','fare']]
    colormap = plt.cm.RdBu
    sns.heatmap(heatmap_data.astype(float).corr(), linesidths = 0.1, vmax= 1.0, square = True, cmap=colormap,linecolor = 'white', annot =True,annot_kws = {"size":10})
    plt.show()