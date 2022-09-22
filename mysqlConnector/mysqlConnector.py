import pymysql
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



con = pymysql.connect(host='XXXXXX',
                      user='XXXX',
                      password='XXX',
                      db='XXXXX',
                      charset='utf8'
                      )

# connection으로 부터 Cursor 설정.
# db에서 추출한 값을 pandas DataFrame화 시켜야 하기 때문에, 딕셔너리 형태로 리턴해주는 cursor을 사용한다.
cursor = con.cursor(pymysql.cursors.DictCursor)

sql="SELECT * FROM wte_food_data"
cursor.execute(sql)

# 데이터 가져오기
rows=cursor.fetchall()
a = pd.DataFrame(rows)
print(a)


tfidf=TfidfVectorizer().fit(a) # tfidf 알고리즘을 동작시키기 위한 vectorizer 객체.
tfidf_matrix = tfidf.fit_transform(a.feature) # 비교해야되는 대상이 각 음식이 가지고 있는 feature 이기 때문에 데이터에서 feature 칼럼만 추출.
# print("tfidf_matrix")
# print(tfidf_matrix.shape) #feature에 대해서 tf-idf를 진행한 결과값 => (80,92) 80->feature 갯수 ,92-> 서로 다른 유니크한 속성값의 갯수

# tf-idf 로직을 수행한 데이터셋에 대해서 cosine 유사도를 구한 결과값을 리턴.
res_cosine=linear_kernel(tfidf_matrix,tfidf_matrix)
# print(res_cosine)

# pandas 시리즈 배열을 통해서, 인덱스가 음식명이고, 인덱스번호가 값인 판다스 시리즈 배열 선언.
idx = pd.Series(a.index, index=a.name).drop_duplicates()
print(idx)

# 유저가 입력한 인덱스 번호(음식명)을 통해서 코사인 유사도가 가장 유사한 값을 추출하는 모델.

def food_chk(name, res_cosine=res_cosine):
    # 유저가 선택한 음식 카테고리를 기반으로 인덱스 번호를 추출한다. -> flask 에서 get 요청으로 받은 값을 기반으로, 진행한다.
    getId = idx[name]

    # 모든 음식 데이터에 대해서 유사도를 구한다.
    sims_cosine= list(enumerate(res_cosine[getId]))

    # 코싸인 유사도에 따라 음식데이터를 정렬한다.
    sims_cosine = sorted(sims_cosine, key=lambda x:x[1], reverse=True)

    # 가장 유사한 데이터 5개를 받아온다.
    sims_cosine = sims_cosine[1:6]

    # 가장 유사한 데이터 5개의 인덱스를 받아온다.
    food_idx = [i[0] for i in sims_cosine]

    result_df = a.iloc[food_idx].copy()
    result_df['score'] = [i[1] for i in sims_cosine]

    # 결과값에서 특징 속성 제거.
    del result_df['feature']
    return result_df

print(food_chk("계란김밥"))
print(food_chk("계란김밥").name)
print(food_chk("계란김밥").imgUrl)
print(food_chk("계란김밥").score)


