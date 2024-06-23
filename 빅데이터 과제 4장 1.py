Python 3.12.1 (tags/v3.12.1:2305ca5, Dec  7 2023, 22:03:25) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
a=123
a=12.34
a=1 + 2j
a.real
1.0
a.real
1.0
a.imag
2.0
a.conjugate()
(1-2j)
abs(a)
2.23606797749979
a=0o12
a
10
a=0x12A
a
298
b=True
b
True
a=3
b=4
a+b
7
a-b
-1
a*b
12
a
a/b
0.75
a**b
81
2
2**3
8
a%b
3
7%3
1
a
a//b
0
7//3
2
s1='Hello Python'
s1
'Hello Python'
s2="Hello Python"
s2
'Hello Python'
s3='''Hello python'''
s3
'Hello python'
s4="""Hello Python"""
s4
'Hello Python'
head = "Python"
tail= " is fun"
head + tail
'Python is fun'
head * 2
'PythonPython'
print("="*5)
=====
a="Now is better than never"
a[0]
'N'
a[4]
'i'
a
a[-1]
'r'
a[-2]
'e'
b=a[0]+a[1]+a[2]
b
'Now'
a[0:3]
'Now'
a[4:6]
'is'
a
a[19:]
'never'
a
a[:3]
'Now'
a[:]
'Now is better than never'
a[7:-11]
'better'
a="python"
a.count('p')
1
a="Python"
a.count('p')
0
a.find('y')
1
a.find('p')
-1
a.index('y')
1
a.index('p')
Traceback (most recent call last):
  File "<pyshell#59>", line 1, in <module>
    a.index('p')
ValueError: substring not found
b=","
c=b.join('Abcd')
c

c
'A,b,c,d'
a.upper()
'PYTHON'
a
a.lower()
'python'
d=" py  "
d.lstrip()
'py  '
d
d.rstrip()
' py'
d.strip()
'py'
a='Pithon'
a[1] = 'y'
Traceback (most recent call last):
  File "<pyshell#71>", line 1, in <module>
    a[1] = 'y'
TypeError: 'str' object does not support item assignment
a="Python is difficult"
a.replace("difficult","funny")
'Python is funny'
a.split()
['Python', 'is', 'difficult']
b="a,b,c,d"
b
'a,b,c,d'
b.split(',')
['a', 'b', 'c', 'd']
a=[1,2,3]
b=['Life','is','too','short']
c=[1,2,'Life','is']
d=[1,2,[3,4],['Life','is']]
d[0]
1
d[2]
[3, 4]
d
d[3]
['Life', 'is']
d[3][-1]
'is'
d[0:3]
[1, 2, [3, 4]]
a+b
[1, 2, 3, 'Life', 'is', 'too', 'short']
b
b[0] + " hi~~^^;"
'Life hi~~^^;'
a[0]+"hi~~^^;"
Traceback (most recent call last):
  File "<pyshell#90>", line 1, in <module>
    a[0]+"hi~~^^;"
TypeError: unsupported operand type(s) for +: 'int' and 'str'
a*3
[1, 2, 3, 1, 2, 3, 1, 2, 3]
a[2] = 99
a
[1, 2, 99]
a[1:2] = ['a','b','c']
a
[1, 'a', 'b', 'c', 99]
a[-1] =['d','b','c']
a
[1, 'a', 'b', 'c', ['d', 'b', 'c']]
a[-1]=['d','
       
SyntaxError: unterminated string literal (detected at line 1)
a[-1]=['d','e','f']
       
a
       
[1, 'a', 'b', 'c', ['d', 'e', 'f']]
del a[-1]
       
a
       
[1, 'a', 'b', 'c']
a.append(5)
       
a
       
[1, 'a', 'b', 'c', 5]
b.sort()
       
b
       
['Life', 'is', 'short', 'too']
a=[3,4,1,9]
       
a.reverse()
       
a
       
[9, 1, 4, 3]
a.index(9)
       
0
a
a.insert(0,99)
       
a
       
[99, 9, 1, 4, 3]
a.remove(99)
       
a
       
[9, 1, 4, 3]
b=[1,2,3]
       
b.pop()
       
3
b
       
[1, 2]
b.pop(0)
       
1
b

a=[2,1,0,2,3,2,4,2]
       
a.count(2)
       
4
t1=(1,)
       
t2=(1,2,3)
       
t3=1,2,3
       
t4=(1,2,(3,4),('Life','is'))
       
t4[0]
       
1
t4[3][-1]
       
'is'
t4[0:3]
       
(1, 2, (3, 4))
t1+t2
       
(1, 1, 2, 3)
t1+"hi~~^^;"
       
Traceback (most recent call last):
  File "<pyshell#130>", line 1, in <module>
    t1+"hi~~^^;"
TypeError: can only concatenate tuple (not "str") to tuple
t2
t2*3
       
(1, 2, 3, 1, 2, 3, 1, 2, 3)
t2[2]=99
       
Traceback (most recent call last):
  File "<pyshell#132>", line 1, in <module>
    t2[2]=99
TypeError: 'tuple' object does not support item assignment
dic={'name':'Hong','phone':'01012345678','birth':'0814'}
       
d
dic[1]='a'
       
dic
       
{'name': 'Hong', 'phone': '01012345678', 'birth': '0814', 1: 'a'}
dic['pet']='dog'
       
dic
       
{'name': 'Hong', 'phone': '01012345678', 'birth': '0814', 1: 'a', 'pet': 'dog'}
d
del dic[1]
       
dic
       
{'name': 'Hong', 'phone': '01012345678', 'birth': '0814', 'pet': 'dog'}
dic['pet']
       
'dog'
dic['name']
       
'Hong'
dic.keys()
       
dict_keys(['name', 'phone', 'birth', 'pet'])
list(dic.keys())
       
['name', 'phone', 'birth', 'pet']
dic.values()
       
dict_values(['Hong', '01012345678', '0814', 'dog'])
list(dic.values())
       
['Hong', '01012345678', '0814', 'dog']
dic.items()
       
dict_items([('name', 'Hong'), ('phone', '01012345678'), ('birth', '0814'), ('pet', 'dog')])
dic.clear()
       
dic
       
{}
s1={1,2,'a',5}
       
s2=set([1,2,3,4,5,6])
       
s2
       
{1, 2, 3, 4, 5, 6}
s3=set([4,5,6,7,8,9])
       
s3
       
{4, 5, 6, 7, 8, 9}
s2 & s3
       
{4, 5, 6}
s
s2.intersection(s3)
       
{4, 5, 6}
s2.union(s3)
       
{1, 2, 3, 4, 5, 6, 7, 8, 9}
s2 -s3
       
{1, 2, 3}
s2.difference(s3)
       
{1, 2, 3}
s3.difference(s2)
       
{8, 9, 7}
s2.add(7)
       
s2
       
{1, 2, 3, 4, 5, 6, 7}
s2.updata([6,7,8,9,10])
       
Traceback (most recent call last):
  File "<pyshell#164>", line 1, in <module>
    s2.updata([6,7,8,9,10])
AttributeError: 'set' object has no attribute 'updata'. Did you mean: 'update'?
s2.update([6,7,8,9,10])
       
s2
       
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
s2.remove(7)
       
s2
       
{1, 2, 3, 4, 5, 6, 8, 9, 10}
a='abc'
       
a
       
'abc'
a=['a','b','c']
       
a
       
['a', 'b', 'c']
a=('a','b','c')
       
a
       
('a', 'b', 'c')
a={'name':"Hong"}
       
a
       
{'name': 'Hong'}
a=set(['a','b','c'])
       
a
       
{'a', 'b', 'c'}
x=3
       
y=2
       
x==y
       
False
x!=y
       
True
x>=y
       
True
money=1300
       
if money >=1200 and money < 3500:
       print("버스를 탈 수 있습니다.")

       
버스를 탈 수 있습니다.
1 in [1,2,3]
       
True
x in [1,2,3]
       
True
x not in [1,2,3]
       
False
'a' in ('a','b','c','d')
       
True
'i' not in 'Python'
       
True
if money <10:
       pass
else:
    print("저금하자!")

    
저금하자!
test_list = ['one','two','three']
for i in test_list:
    x=i+ '!
    
SyntaxError: unterminated string literal (detected at line 2)
for i in test_list:
    x=i+'!'
    print(x)

    
one!
two!
three!
number=0
for score in [90,25,67,45,93]:
    number += 1
    if score >= 60:
        print("%d번 학생은 합격입니다." %number)
    else:
        print("%d번 학생은 불합격입니다." %number)

        
1번 학생은 합격입니다.
2번 학생은 불합격입니다.
3번 학생은 합격입니다.
4번 학생은 불합격입니다.
5번 학생은 합격입니다.
i=0
while i <5:
    i+=1
    print('*'*i)

    
*
**
***
****
*****
def sum1(a,b):
    x=a+b
    return x

def sum2(*args):
    x=0
    for i in args:
        x+= i
    return x

a=5
b
b=3
sum1(a,b)
8
sum1(3,5)
8
s
sum2(1,2,3,4,5)
15
sum2(2,3.5,10)
15.5
abs(-3.5)
3.5
all([1,2,3,4,])
True
a
all([4,-2,0.0,4])
False
any([1,2,3,4])
True
any([4,-2,0.0,4])
True
chr(97)
'a'
chr(48)
'0'
ord('a')
97
ord('0')
48
dir([1,2,3,])
['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
dir([1,2,3])
['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
dir({'1':'a'})
['__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__ior__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__or__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__ror__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
dir(1)
['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__getstate__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_count', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'is_integer', 'numerator', 'real', 'to_bytes']
d
divmod(7,3)
(2, 1)
divmod(1.3,0.2)
(6.0, 0.09999999999999998)
oct(8)
'0o10'
oct(234)
'0o352'
hex(16)
'0x10'
hex(234)
'0xea'
a=3
id(a)
140713931520504
int('3')
3
str(3)
'3'
list('Python')
['P', 'y', 't', 'h', 'o', 'n']
list((1,2,3))
[1, 2, 3]
tuple("Python")
('P', 'y', 't', 'h', 'o', 'n')
tuple([1,2,3,])
(1, 2, 3)
type("abc")
<class 'str'>
type(a)
<class 'int'>
sum=lambda a,b: a+b
sum
<function <lambda> at 0x000001AD37694A40>
sum(3,5)
8
max([1,4,2,8,6])
8
max("Python")
'y'
min([1,4,2,8,6])
1
min("Python")
'P'
pow(2,4)
16
c=input()
c
c
'c'
c=input()
21
c
'21'
>>> c=input("정수를 입력하세요:")
정수를 입력하세요:21
>>> c
'21'
>>> range(5)
range(0, 5)
>>> list(range(5))
[0, 1, 2, 3, 4]
>>> list(range(5,10))
[5, 6, 7, 8, 9]
>>> list(range(5,10,2))
[5, 7, 9]
>>> len('Python')
6
>>> sorted([3,0,2,1])
[0, 1, 2, 3]
>>> sorted('Python')
['P', 'h', 'n', 'o', 't', 'y']
>>> Request('http://www.hanb.co.kr')
Traceback (most recent call last):
  File "<pyshell#286>", line 1, in <module>
    Request('http://www.hanb.co.kr')
NameError: name 'Request' is not defined
>>> imort urllib.reqest
SyntaxError: invalid syntax
>>> import urllib.reqest
Traceback (most recent call last):
  File "<pyshell#288>", line 1, in <module>
    import urllib.reqest
ModuleNotFoundError: No module named 'urllib.reqest'
>>> urimort urllib.request
SyntaxError: invalid syntax
>>> urllib.request.Request('http://www.hanb.co.kr')
Traceback (most recent call last):
  File "<pyshell#290>", line 1, in <module>
    urllib.request.Request('http://www.hanb.co.kr')
NameError: name 'urllib' is not defined. Did you forget to import 'urllib'?
>>> import pandas
Traceback (most recent call last):
  File "<pyshell#291>", line 1, in <module>
    import pandas
ModuleNotFoundError: No module named 'pandas'
>>> pandas.DataFrame()
Traceback (most recent call last):
  File "<pyshell#292>", line 1, in <module>
    pandas.DataFrame()
NameError: name 'pandas' is not defined
>>> frome datatime import datetime
SyntaxError: invalid syntax
>>> frme datetime import datetime
SyntaxError: invalid syntax
>>> from datetime import datetime
>>> datetime.now
<built-in method now of type object at 0x00007FFA83DAAFD0>
>>> from datetime import datetime
>>> datetime.now()
datetime.datetime(2024, 6, 13, 13, 18, 59, 369778)
>>> f=open
>>> f
<built-in function open>
f=open("D:새파일.txt,'a')
for i in range(6,11):
    data= "%d번째 줄 추가입니다.\n"% i
    f.write(data)
f.close()
f=open("D:새파일.txt,'r')
line = f.readlines()
print(lines)
for line in lines:
    print(line)
f.close()
f=open("D:새파일.txt,'r')
data = f.read()
data
with open("D:새파일.txt,'w')as f:
    f.write("Now is better than never.")
data = f.read()
