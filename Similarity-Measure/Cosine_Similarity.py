import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances #사이킷 런 모듈에 내장되어 있는 코사인 유사도 함수

#numpy 모듈에 탑재되어 있는 dot 함수를 활용한 목록간의 유사도 측정.

from numpy import dot
from numpy.linalg import norm

list_1=[4,2,1,5,2]
list_2=[4,2,1,3,3]

result_dot=dot(list_1,list_2)/(norm(list_1)*norm(list_2))
print(result_dot)


# 사이킷 런 모듈을 활용한 A,B 두 좌표간의 코사인 유사도 값.
A=np.array([10,3])
B=np.array([8,7])

result=cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
# print(result)
