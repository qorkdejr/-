Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
np.__version__
'1.26.4'
ar1 = np.array([1,2,3,4,5])
ar1
array([1, 2, 3, 4, 5])
type(ar1)
<class 'numpy.ndarray'>
ar2=np.array([[10,20,30],[40,50,60]])
ar2
array([[10, 20, 30],
       [40, 50, 60]])
ar3=np.arragne(1,11,2)
Traceback (most recent call last):
  File "<pyshell#7>", line 1, in <module>
    ar3=np.arragne(1,11,2)
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\__init__.py", line 347, in __getattr__
    raise AttributeError("module {!r} has no attribute "
AttributeError: module 'numpy' has no attribute 'arragne'
ar3=np.arange(1,11,2)
                         
ar3
                         
array([1, 3, 5, 7, 9])
ar4=ar3=np.array(p1,2,3,4,5,6]).reshape((3,2))
SyntaxError: closing parenthesis ']' does not match opening parenthesis '('
ar4=ar3=np.array([1,2,3,4,5,6]).reshape((3,2))
ar4
array([[1, 2],
       [3, 4],
       [5, 6]])
ar5=np.zeros((2,3))
ar5
array([[0., 0., 0.],
       [0., 0., 0.]])
ar6=ar2[0:2, 0:2]
ar6
array([[10, 20],
       [40, 50]])
ar7=ar2[0,:]
ar7
array([10, 20, 30])
ar8=ar1+10
ar8
array([11, 12, 13, 14, 15])
ar1+ar8
array([12, 14, 16, 18, 20])
ar8-ar1
array([10, 10, 10, 10, 10])
ar1*2
array([ 2,  4,  6,  8, 10])
ar1/2
array([0.5, 1. , 1.5, 2. , 2.5])
ar9=np.dot(ar2,ar4)
ar9
array([[220, 280],
       [490, 640]])
import pandas as pd
p
pd.__version__
'2.2.2'
data1=[10,20,30,40,50]
data1
[10, 20, 30, 40, 50]
data2=['1반','2반','3반','4반','5반']
data2
['1반', '2반', '3반', '4반', '5반']
sr1= pd.Series(data1)
sr1
0    10
1    20
2    30
3    40
4    50
dtype: int64
sr2=pd.Series(data2)
sr2
0    1반
1    2반
2    3반
3    4반
4    5반
dtype: object
sr3=pd.Series([101,102,103,104,104])
sr3
0    101
1    102
2    103
3    104
4    104
dtype: int64
sr4=pd.Series(['월','화','수','목','금'])
sr4
0    월
1    화
2    수
3    목
4    금
dtype: object
s
sr5=pd.Series(data1,index =[1000,1001,1002,1003,1004])
sr5
1000    10
1001    20
1002    30
1003    40
1004    50
dtype: int64
sr6=pd.Series(data1,index=data2)
sr6
1반    10
2반    20
3반    30
4반    40
5반    50
dtype: int64
sr7=pd.Series(data2,index=data1)
sr7
10    1반
20    2반
30    3반
40    4반
50    5반
dtype: object
sr8=pd.Series(data2,index=sr4)
sr8
월    1반
화    2반
수    3반
목    4반
금    5반
dtype: object
sr8[2]

Warning (from warnings module):
  File "<pyshell#49>", line 1
FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
'3반'
sr8['수']
'3반'
sr8[-1]
'5반'
sr8[0:4]
월    1반
화    2반
수    3반
목    4반
dtype: object
sr8.index
Index(['월', '화', '수', '목', '금'], dtype='object')
sr8.values
array(['1반', '2반', '3반', '4반', '5반'], dtype=object)
sr1+sr3
0    111
1    122
2    133
3    144
4    154
dtype: int64
sr4+sr2
0    월1반
1    화2반
2    수3반
3    목4반
4    금5반
dtype: object
data_dic={
    'year':[2018,2019,2020],
    'sales':[350,480'1099]
             
SyntaxError: unterminated string literal (detected at line 3)
data_dic={
    'year':[2018,2019,2020],
    'sales':[350,480,1099]
    }
             
data_dic
             
{'year': [2018, 2019, 2020], 'sales': [350, 480, 1099]}
df1=pd.DataFrame(data_dic)
             
df1
             
   year  sales
0  2018    350
1  2019    480
2  2020   1099
df2=pd.DataFrame([[89.2,92.5,90.8],[92.8,89.9,95.2]],index=['중간고사','기말고사'], columns=data2[0:3])
             
df2
             
        1반    2반    3반
중간고사  89.2  92.5  90.8
기말고사  92.8  89.9  95.2
data_df+[['20201101','Hong','90','95'],['20201102','Kim','93','94'],['20201103','Lee','87','97']]
             
Traceback (most recent call last):
  File "<pyshell#67>", line 1, in <module>
    data_df+[['20201101','Hong','90','95'],['20201102','Kim','93','94'],['20201103','Lee','87','97']]
NameError: name 'data_df' is not defined. Did you mean: 'data_dic'?
data_df=[['20201101','Hong','90','95'],['20201102','Kim','93','94'],['20201103','Lee','87','97']]
             
df3=pd.DataFrame(data_df)
             
df3
             
          0     1   2   3
0  20201101  Hong  90  95
1  20201102   Kim  93  94
2  20201103   Lee  87  97
df3.columns=['학번','이름','중간고사','기말고사']
             
df3
             
         학번    이름 중간고사 기말고사
0  20201101  Hong   90   95
1  20201102   Kim   93   94
2  20201103   Lee   87   97
df3.head(2)
             
         학번    이름 중간고사 기말고사
0  20201101  Hong   90   95
1  20201102   Kim   93   94
df3.taIl(2)
             
Traceback (most recent call last):
  File "<pyshell#74>", line 1, in <module>
    df3.taIl(2)
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\generic.py", line 6299, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'taIl'. Did you mean: 'tail'?
df3.tail(2)
         학번   이름 중간고사 기말고사
1  20201102  Kim   93   94
2  20201103  Lee   87   97
df3['이름']
0    Hong
1     Kim
2     Lee
Name: 이름, dtype: object
import matplotlib
Traceback (most recent call last):
  File "<pyshell#77>", line 1, in <module>
    import matplotlib
ModuleNotFoundError: No module named 'matplotlib'
import matplotlib
Traceback (most recent call last):
  File "<pyshell#78>", line 1, in <module>
    import matplotlib
ModuleNotFoundError: No module named 'matplotlib'
matplotlib.__version__
Traceback (most recent call last):
  File "<pyshell#79>", line 1, in <module>
    matplotlib.__version__
NameError: name 'matplotlib' is not defined
import matplotlib

>>> matplotlib.__version
Traceback (most recent call last):
  File "<pyshell#81>", line 1, in <module>
    matplotlib.__version
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\matplotlib\_api\__init__.py", line 217, in __getattr__
    raise AttributeError(
AttributeError: module 'matplotlib' has no attribute '__version'. Did you mean: '_version'?
>>> matplotlib.__version__
'3.9.0'
>>> import matplotlib.pyplot as plt
>>> x=[2016,2017,2018,2019,2020]
>>> y=[350,410,520,695,543]
>>> plt.plot(x,y)
[<matplotlib.lines.Line2D object at 0x0000013DE529BDA0>]
>>> plt.title('Annual sales')
Text(0.5, 1.0, 'Annual sales')
>>> plt.xlable('years')
Traceback (most recent call last):
  File "<pyshell#88>", line 1, in <module>
    plt.xlable('years')
AttributeError: module 'matplotlib.pyplot' has no attribute 'xlable'. Did you mean: 'table'?
>>> plt.xlabel('years')
Text(0.5, 0, 'years')
>>> plt.ylabel('sales')
Text(0, 0.5, 'sales')
>>> plt.show()
>>> y1=[350,410,520,695]
>>> y2=[200,250,385,350]
>>> x=range(len(y1))
>>> plt.bar(x,y1,width=0.7,color="blue")
<BarContainer object of 4 artists>
>>> plt.bar(x,y2,width=0.7,color="red",bottom=y1)
<BarContainer object of 4 artists>
>>> plt.title('Quarterly sales')
Text(0.5, 1.0, 'Quarterly sales')
>>> plt.xlabel('Quarters')
Text(0.5, 0, 'Quarters')
>>> plt.ylabel('sales')
Text(0, 0.5, 'sales')
>>> xLabel=['first','second'm'third','fourth']
SyntaxError: invalid syntax. Perhaps you forgot a comma?
>>> xLabel=['first','second','third','fourth']
>>> plt.xticks(x,xLabel,frontsize=10)
Traceback (most recent call last):
  File "<pyshell#102>", line 1, in <module>
    plt.xticks(x,xLabel,frontsize=10)
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\matplotlib\pyplot.py", line 2157, in xticks
    labels_out = ax.set_xticklabels(labels, minor=minor, **kwargs)
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\matplotlib\axes\_base.py", line 74, in wrapper
    return get_method(self)(*args, **kwargs)
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\matplotlib\axis.py", line 2095, in set_ticklabels
    tick.label1._internal_update(kwargs)
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\matplotlib\artist.py", line 1216, in _internal_update
    return self._update_props(
  File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\matplotlib\artist.py", line 1190, in _update_props
    raise AttributeError(
AttributeError: Text.set() got an unexpected keyword argument 'frontsize'
>>> plt.legend(['chairs','desks'])
<matplotlib.legend.Legend object at 0x0000013DE5298FB0>
>>> plt.show()
