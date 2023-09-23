# def karatsuba(num1, num2):
#     num1Str = str(num1)
#     num2Str = str(num2)
#     if (num1 < 10) or (num2 < 10):
#         return num1*num2

#     maxLength = max(len(num1Str), len(num2Str))
#     splitPosition = maxLength // 2
#     high1, low1= int(num1Str[:-splitPosition]), int(num1Str[-splitPosition:])
#     high2, low2= int(num2Str[:-splitPosition]), int(num2Str[-splitPosition:])
#     z0 = karatsuba(low1, low2)
#     z1 = karatsuba((low1 + high1), (low2 + high2))
#     z2 = karatsuba(high1, high2)

#     return (z2*10**(2*splitPosition)) + ((z1-z2-z0)*10**(splitPosition))+z0
# a = 3141592653589793238462643383279502884197169399375105820974944592
# b = 2718281828459045235360287471352662497757247093699959574966967627
# print(karatsuba(a,b))

# def check(a):
#     if a == int(a):
#         return int(a)
#     return a
# v = []
# for i in range(3):
#     v.append(check(float(input())))
# v.sort()
# for i in v:
#     print(i)

# def isOdd(a):
#     b = a%10
#     if b%2==0:
#         if (b/2)%2==0:
#             return 'YES'
#     return 'NO'
# a = int(input())
# if a == 0: print('NO')
# else: print(isOdd(a))

# def palindrome(a):
#     n = len(a)
#     for i in range(n):
#         if a[i]!=a[n-1-i]:
#             return 'false'
#     return 'true'
# a = input()
# print(palindrome(a))

# d,r,n = map(int,input().split())
# L= 2*r*(n-2) + 2*(2*r+d)
# print(L)
# import numpy
# print(numpy.version.version)

# a = int(input())
# for i in range(2, a, 2):
#     if (a - i) % 2 == 0:
#         print("YES")
#         break
# else:
#     print("NO")

# a = float(input())
# b = float(input())
# c = float(input())
# sort_arr = sorted([a, b, c])
# for i in range(3):
#     if int(sort_arr[i]) == sort_arr[i]:
#         sort_arr[i]= int(sort_arr[i])
# print(sort_arr[0], sort_arr[1], sort_arr[2])

# num = float(input())
# pound = 0.453592 
# inch = 2.54**2
# weight = num*pound/inch
# t, d= weight, 0
# while t>=1:
#     t=t/10
#     d+=1
# if weight<1: d=1
# print(round(weight,6-d))

# x = int(input())
# for i in range(4):
#     for j in range(6):
#         if i==0 or i==4-1:
#             print(x, end=' ')
#         elif j==0 or j==5:
#             print(x, end=' ')
#         else: print(' ',end=' ')
#         if j==5:
#             print('', end='\n')
# for i in range(5):
#     a,b=5-i,5+i
#     for j in range(12):
#         if j==a or j==b:
#             print(x,end='')
#         else:
#             print(' ',end='')
#         if j==12-1: print('',end='\n')
# for i in range(6): print(x,end=' ')

# n = int(input())
# sum = 0
# while n>=1:
#     a=n%10
#     sum+=int(a)
#     n/=10
# print(sum)

# import math
# a,b,c = map(float,input().split())
# p=(a+b+c)/2
# s = math.sqrt(p*(p-a)*(p-b)*(p-c))
# print('{:.2f}'.format(s))

# a,b = map(int,input().split())
# y=(b-2*a)/2
# x=a-y
# print(int(x),int(y))

# age = int(input())
# gender = input()
# if gender == 'M' or gender == 'm':
#     if age>=21: print(1)
#     else: print(3)
# elif gender =='F' or gender == 'f':
#     if age>=21: print(2)
#     else: print(4)
# else: print('I do not know why')
    
# n = int(input())
# sum=1
# for i in range(n,0,-2):
#     sum*=i
# print(sum)


# a,b = map(int,input().split())
# max = ((a+b)+abs(a-b))/2
# min = ((a+b)-abs(a-b))/2
# print('max =',int(max))
# print('min =',int(min))

# import math
# a,b,c = map(float,input().split())
# x = a**5 - 2*(math.sqrt(abs(b))) + a*b*c
# print('%.2f' % x)

# n = int(input())
# sum = 1 
# for i in range(2,int(n**0.5)+1):
#     if n % i == 0:
#         sum+=i
#         a = n/i
#         if i != a:
#             sum+=a
# print(int(sum))

# def Fibonanci(n):
#     if n==1 or n==2: 
#         return 1
#     return Fibonanci(n-1) + Fibonanci(n-2)
# n = int(input())
# if n<1 or n>30:
#     print('So',n,'khong nam trong khoang [1,30].')
# else:
#     print(Fibonanci(n))

# l = input()
# rs = ''
# dic = ['A', 'O', 'Y', 'E', 'U', 'I']
# for i in l:
#     d=0
#     for j in dic:
#         if i.upper()==j: 
#             d=1
#     if d==0: rs+='.'+i.lower()
# print(rs)

# k,t = map(int,input().split())
# l= t//k
# if l%2!=0:
#     print(k-t%k)
# else:
#     print(t%k)

# n = int(input())
# s = str(n)
# d, four, seven = 0, 0, 0
# for i in s:
#     if i=='4' or i=='7' : 
#         d+=1
# if d==4 or d==7:
#     print('YES')
# else: print('NO')
# import math
# n, m, h, w = map(int,input().split())
# d, d1, h1, w1 = 0, 0, w, h
# while True:
#     if n<=h and m<=w: break
#     if n>h: h*=2
#     else: w*=2
#     d+=1
# while True:
#     if n<=h1 and m<=w1: break
#     if n>h1: h1*=2
#     else: w1*=2
#     d1+=1
# print(min(d,d1))

# s = input()
# d = 0 
# for i in range(len(s)):
#     if s[i]!=s[i-1]:
#         d+=1
# if d%2==0: print('Lose')
# else: print('Win')

# n = int(input())
# if n%2==0:
#     print(0)
# else: print(1)

# def check(a):
#     if int(a)==a:
#         return int(a)
#     return a
# a = float(input())
# b = float(input())
# c = float(input())
# delta = b**2-4*a*c
# if delta>0:
#     x1 = (-b + delta**0.5)/(2*a)
#     x2 = (-b - delta**0.5)/(2*a)
#     print('PT co hai nghiem phan biet:')              
#     print('x1 =', check(x1))
#     print('x2 =', check(x2))
# elif delta==0:
#     print('PT co nghiem kep: x1 = x2 =', check(-b/(2*a))
# else:
#     print('PTVN')

# n = int(input())
# s = ''
# for i in range(n+1):
#     s+= str(i)
# print(s[n])]

# n, m = map(int,input().split())
# print( m//(10**len(str(n))) + (m%(10**len(str(n)))>=n))



# n = int(input())
# c = list(map(int,input().split()))   
# c.sort()
# if c == [0]*n or max(c)<n:
#     print(0)
# else:
#     for i,x in enumerate(c):
#         if x>= n-i:
#             print(n-i)
#             break
    
# n = int(input())
# l = list(map(int,input().split()))  
# l.sort()
# print(min(l[-1]-l[1],l[-2]-l[0]))

# arr = map(int,l)
# l = []
# posX, posY = 0, 0
# for i in range(5):
#     l.append(input().split())
# for i in range(5):
#     for j in range(5):
#         if l[i][j]=='1':
#             posX=i
#             posY=j
#             break
# print(abs(2-posX)+abs(2-posY))

# n = int(input())
# l = []
# money = 0
# for i in range(n):
#     l.append(input().split())
# for i in range(n):
#     if l[i][0]=='W':
#         money-= int(l[i][1])
#     else: money+= int(l[i][1])
# print(money)
    
# def gt(n):
#     if n==0:
#         return 1
#     return gt(n-1)*n
# n = int(input())
# print(gt(n))

# n, k = map(int,input().split())
# s = input()
# char = ''
# for i in range(k): char+=chr(65+i)
# c = [s.count(i) for i in char]
# c.sort()
# print(c[len(char)-k]*k)
# print(char,s.count(char[0]),c)
# def delete_space(s):
#     a = ''
#     s.strip()
#     while "  " in s:
#         s = s.replace("  "," ")
#     for i in range(len(s)):
#         if i==0 or s[i-1]== " ":
#             a = s[
#         else: 
#             a = s[i].lower()
#     return s
# a = input()
# s = delete_space(a)
# print(s)

# s = input()
# n, w = 0, 0
# for i in s:
#     if i.isalpha():
#         w+=1
#     if i.isdigit():
#         n+=1
# print(w)
# print(n)

# def lt(a,b):
#     d = 10**9+7
#     if b == 0: return 1
#     if b == 1: return a%d
#     t = lt(a,b//2)
#     if b%2==0: return t*t%d
#     return t*t%d*a%d

# m, n = map(int,input().split())
# print(lt(m,n))

# a = int(input())
# b = int(input())
# l = ''
# for i in range(a,b+1):
#     if i%2==0:
#         l=l+str(i)+','
# print(l[:-1]) 

# s1 = input()
# s2 = input()
# s = s2[::-1]
# if s==s1:
#     print('YES')
# else: print('NO')

# def count_palindrome(s):
#     n = len(s)
#     count = 0
#     for i in range(n):
#         l, r = i, i
#         while l >= 0 and r < n and s[l] == s[r]:
#             count += 1
#             l -= 1
#             r += 1
#         l, r = i, i + 1
#         while l >= 0 and r < n and s[l] == s[r]:
#             count += 1
#             l -= 1
#             r += 1
#     return count
# s = input()
# print(count_palindrome(s))

# n = int(input())
# s = input()
# d = s.count('8')
# if d==0:
#     print(0)
# else:
#     if n//11 < d:
#         print(int(n/11))
#     else: print(d)

# def count_x(n, x):
#     count = 0
#     for i in range(1, n+1):
#         if x % i == 0 and x // i <= n:
#             count += 1
#     return count
# n,x = map(int,input().split())
# print(count_x(n,x))

# n = int(input())
# x,y = map(int,input().split())
# dw, db = 0, 0
# t = min(x,y)-1
# d = min(x,y)
# if x == d: dw= t + y-d
# else: dw = t + x-d
# t = n - max(x,y)
# d = max(x,y)
# if x == d: db= t + d - y
# else: db = t + d - x
# if dw<=db: print('White')
# else: print('Black')

# n = int(input())
# l = list(map(int,input().split()))
# a ,b = min(l), max(l)
# print((b-a+1)-len(l))

# def cal(x):
#     count= 0
#     while True:
#         if x%2==0 and x>2:
#             return count
#         x+=1
#         count+=1
# n = int(input())
# l = []
# for i in range(n):
#     l.append(int(input()))
# for i in range(n):
#     print(cal(l[i]))

# m,n = map(int,input().split())
# print((m-1)*(n-1))

# def point(x,y):
#     if x>y:
#         return 3
#     elif x==y: return 1
#     return 0
# l = []
# d1,h1,s1,c1,d2,h2,s2,c2 = 0,0,0,0,0,0,0,0
# for i in range(6):
#     l.append(list(map(int,input().split(" "))))
# for i in range(3):
#     d1 += point(l[i][0],l[i][1])
#     d2 += point(l[i+3][0],l[i+3][1])
#     h1 +=l[i][0]-l[i][1]
#     h2 +=l[i+3][0]-l[i+3][1]
#     s1 +=l[i][0]
#     s2 +=l[i+3][0]
#     c1 +=l[i][2]
#     c2 +=l[i+3][2]
# rs = [[d1,h1,s1,c1],[d2,h2,s2,c2]]
# for i in range(3):
#     if rs[0][i]>rs[1][i]:
#         print(d1,h1,s1,c1)
#         break
#     elif rs[0][i]<rs[1][i]:
#         print(d2,h2,s2,c2)
#         break
#     else: continue

# import pandas as pd
# list_link = ['1','2','3','4','5']
# list_authors = [['lvk','ntl','ptb','lntd'],['lvk','ntl','ptb','lntd'],['lvk','ntl','ptb','lntd'],['lvk','ntl','ptb','lntd'],['lvk','ntl','ptb','lntd']]
# list_comments = [['a','b','c','d','e'],['a','b','c','d','e'],['a','b','c','d','e'],['a','b','c','d','e'],['a','b','c','d','e']]
# path = 'C:/Users/VQ\Desktop/CS232-TTDPT/Lab 1/Crawl Post and Comment from Online Newspaper/data.csv'
# df = pd.DataFrame()
# for i in range(5):
#         new_row = {'Post':list_link[i],'Name of author':list_authors[i][0],'Comment':list_comments[i][0] }
#         df = df.append(new_row, ignore_index=True)
#         for j in range(1,5):
#             print(j)
#             new_row = {'Name of author':list_authors[i][j],'Comment':list_comments[i][j] }
#             df = df.append(new_row, ignore_index=True)
# df.to_csv(path,index = False, encoding = 'utf-8-sig')


# import numpy as np
# from numpy.linalg import matrix_rank
# n = int(input())
# arr = np.array([input().strip().split() for _ in range(n)],int)
# if np.linalg.matrix_rank(arr)==n: print('YES')
# else: print('NO')

# import numpy as np
# from numpy.linalg import matrix_rank
# n = int(input())
# arr = np.array([input().strip().split() for _ in range(n)],int)
# t = np.eye(n)
# if np.array_equal(arr,t) or np.array_equal(arr,np.rot90(t,k=1)):
#     print('YES')
# else: print('NO')

# import numpy as np
# from numpy.linalg import matrix_rank
# n = int(input())
# a = np.array([input().strip().split() for _ in range(n)],int)
# b = np.array([input().strip().split() for _ in range(n)],int)
# d = 0
# for i in range(4):
#     b = np.rot90(b)
#     if np.array_equal(a,b):
#         d = 1
#         break
# if d == 0: 
#     print('NO')
# else: print('YES')


# 4
# def find_len(a,c):
#     l = []
#     for i in range(c):
#         m = np.max(a[:,i])
#         n = np.min(a[:,i])
#         t = len(str(m))
#         if t < len(str(n)): t = len(str(n))
#         l.append(t)
#     return l
# def space(d):
#     for i in range(d):
#         print(' ',end='')
# import numpy as np
# r,c = map(int,input().split())
# a = np.array([input().strip().split() for _ in range(r)],int)
# l = find_len(a,c)
# b = np.array(a,str)
# for i in range(r):
#     for j in range(c):
#         space(l[j]-len(b[i][j]))
#         print(b[i][j],end=' ')
#     print('')

# 5 
# import numpy as np
# from numpy.linalg import matrix_rank
# h,w = map(int,input().split())
# a = np.array([input().strip().split() for _ in range(h)],int)
# t,l,b,r = map(int,input().split())
# for i in range(h):
#     for j in range(w):
#         if i < l or i > r or j < t or j > b:
#             if a[i,j] == 1:
#                 a[i,j] = 0
# for row in a:
#     print(*row)



# 6   
# def printMT(matrix):
#     for i in matrix:
#         for j in i:
#             print(j, end = " ")
#         print()
# n,m = map(int,input().split())
# r,c = map(int,input().split())
# matr = []
# for i in range(n):
#     matr.append(input().split())
# if n*m != r*c:
#     printMT(matr)
# else:
#     re = [[0 for i in range(c)] for i in range(r)]
#     k = 0
#     for i in range(len(matr)):
#         for j in range(len(matr[0])):
#             re[k//c][k%c] = matr[i][j]
#             k+=1
#     printMT(re)

# 7

# from sys import stdin,stdout
# M = 10**9 + 7
# dp = []
# for i in range(2*10**5):
#     if i<10: dp.append(i)
#     else: dp.append((dp[i-9] + dp[i-10]) %M)
# for _ in range(int(stdin.readline())):
#     n,m = stdin.readline().split()
#     m = int(m)
#     ans = 0
#     for i in n:
#         ans = (ans+dp[m+int(i)]) % M
#     stdout.write(str(ans)+'\n')
# 8
# import numpy as np
# n = int(input())
# a = np.array(input().split(),int)
# max_prof, max_prof_s, max_prof_e = -1e9, 0, 0
# cur_prof = 0
# cur_s = 0
# for i in range(n):
#     cur_prof += a[i]
#     if cur_prof > max_prof:
#         max_prof = cur_prof
#         max_prof_s = cur_s
#         max_prof_e = i
#     if cur_prof < 0:
#         cur_prof = 0
#         cur_s = i + 1
# print(max_prof_s+1, max_prof_e+1, max_prof)

        
# import statistics
# a = [-463,	-438	,-430,	-478,	-477]
# print(a.mean())

import requests
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
url =  'https://openaccess.thecvf.com/WACV2023'
page = requests.urlopen(url)
soup = BeautifulSoup(page.content, 'html.parser')

titles = []
papers = soup.find_all('dt', class_='ptitle')
titles = [paper.find('a').text for paper in papers]

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_word = stopwords.words('english')

words = word_tokenize(titles)
words = words.lower()
words = words.replace(':','')

