import json
import pymysql
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel #코사인 유사도

disabled_1 = '김치'
disabled_2 = '소고기'
disabled_3 = '크림'
disabled_4 = '치즈'
disabled_5 = '조개'
disabled_6 = ''


con = pymysql.connect(host='XXXX',
                      user='XXXX',
                      password='XXXX',
                      db='XXXX',
                      charset='XXXX'
                      )

# connection으로 부터 Cursor 설정.
# db에서 추출한 값을 pandas DataFrame화 시켜야 하기 때문에, 딕셔너리 형태로 리턴해주는 cursor을 사용한다.
cursor = con.cursor(pymysql.cursors.DictCursor)

# python 에서 REGEXP 쿼리를 사용할 때, 다음과 같이 분류한다.
# sql="SELECT * FROM ETY_DB.wte_food_data WHERE feature NOT REGEXP ('"+disabled_5+"|"+disabled_1+"|"+disabled_2+"')"
sql="SELECT * FROM ETY_DB.wte_food_data"


cursor.execute(sql)

# 데이터 가져오기
rows=cursor.fetchall()
a = pd.DataFrame(rows)
# print(a)


tfidf=TfidfVectorizer().fit(a) # tfidf 알고리즘을 동작시키기 위한 vectorizer 객체.
tfidf_matrix = tfidf.fit_transform(a.feature) # 비교해야되는 대상이 각 음식이 가지고 있는 feature 이기 때문에 데이터에서 feature 칼럼만 추출.
# print("tfidf_matrix")
# print(tfidf_matrix.shape) #feature에 대해서 tf-idf를 진행한 결과값 => (80,92) 80->feature 갯수 ,92-> 서로 다른 유니크한 속성값의 갯수

# tf-idf 로직을 수행한 데이터셋에 대해서 cosine 유사도를 구한 결과값을 리턴.
res_cosine=linear_kernel(tfidf_matrix,tfidf_matrix)
# print(res_cosine)

# pandas 시리즈 배열을 통해서, 인덱스가 음식명이고, 인덱스번호가 값인 판다스 시리즈 배열 선언.
idx = pd.Series(a.index, index=a.name).drop_duplicates()
# print(idx)

# 유저가 입력한 인덱스 번호(음식명)을 통해서 코사인 유사도가 가장 유사한 값을 추출하는 모델.

def food_chk(name, res_cosine=res_cosine):
    # 유저가 선택한 음식 카테고리를 기반으로 인덱스 번호를 추출한다. -> flask 에서 get 요청으로 받은 값을 기반으로, 진행한다.
    getId = idx[name]

    # 모든 음식 데이터에 대해서 유사도를 구한다.
    sims_cosine= list(enumerate(res_cosine[getId]))

    # 코싸인 유사도에 따라 음식데이터를 정렬한다.
    sims_cosine = sorted(sims_cosine, key=lambda x:x[1], reverse=True)

    # 가장 유사한 데이터 5개를 받아온다.
    sims_cosine = sims_cosine[1:3]

    # 가장 유사한 데이터 5개의 인덱스를 받아온다.
    food_idx = [i[0] for i in sims_cosine]

    result_df = a.iloc[food_idx].copy()
    result_df['score'] = np.round([i[1] for i in sims_cosine], 2) * 100
    result_df['score'] = result_df['score'].astype(str)+'%'
    # 결과값에서 특징 속성 제거.
    del result_df['feature']
    return result_df

# print(type(food_chk("계란김밥")))

testcase = pd.concat([food_chk("계란김밥")], ignore_index=True)
# print(testcase)

res = pd.concat([food_chk("계란김밥"),food_chk("김치찌개"),food_chk("족발")], ignore_index=True)

# res5 = pd.concat([food_chk("")],ignore_index=True)
# print(res5)

res2 = pd.concat([res, food_chk("보쌈")], ignore_index=True)
print(type(res2))


# print(json.dumps(res2.to_dict('records'), ensure_ascii=False,indent=4))


# 서버에서 받아온 유저가 선호하는 음식 항목 컴마로 각 아이템이 구분지어있기 때문에, 별도로 데이터를 배열에 뽑아서 넣어준다.
keys = "족발,보쌈,라볶이,떡볶이"

case =[]
print(type(keys.split(","))) # 리스트로 저장됨.

for i in keys.split(","):
    case.append(i)

print(case)


# 이렇게 분기로 나누어 뽑는 방법은 그렇게 좋은 방식은 아닌데, 다른 방법이 생각 안남.
# ver 1.0 에서는 분기로 나누어 사용하고, ver 1.1 에서는 클라이언트에서 받아온 데이터의 수에 맞게 처리하도록 수정.
if len(case) == 1:
    print("len -> 1")
    print("value -> "+case[0])
    pdSample = pd.concat([food_chk(case[0])], ignore_index=True)
elif len(case) == 2:
    print("len -> 2")
    print("value -> "+case[0]+case[1])
    pdSample = pd.concat([food_chk(case[0]),food_chk(case[1])], ignore_index=True)
elif len(case) == 3:
    print("len ->3")
    print("value -> "+case[0]+case[1]+case[2])
    pdSample = pd.concat([food_chk(case[0]),food_chk(case[1]),food_chk(case[2])], ignore_index=True)
elif len(case) == 4:
    print("len ->4")
    print("value -> "+case[0]+case[1]+case[2]+case[3])
    pdSample = pd.concat([food_chk(case[0]),food_chk(case[1]),food_chk(case[2]),food_chk(case[3])], ignore_index=True)
else :
    print("len -> 5 이상.")
    print("value -> "+case[0]+case[1]+case[2]+case[4]+case[5])
    pdSample = pd.concat([food_chk(case[0]),food_chk(case[1]),food_chk(case[2]),food_chk(case[3]),food_chk(case[4])], ignore_index=True)

print("--------------")
print(pdSample)
print("--------------")
print("--------------")
print("정렬")
print(pdSample.sort_values('score', ascending=False))
print("--------------")
print("--------------")
print("중복제거")
print(pdSample.drop_duplicates(['name']))
print("--------------")
print("--------------")
print(json.dumps(pdSample.sort_values('score', ascending=False).to_dict('records'), ensure_ascii=False, indent=4))
print("--------------")




#
# print(res.to_json(force_ascii=False, orient = 'records', indent=4))
# # 최종적으로 정상적으로 추출한 값.
# # to_json 으로 url 추출시 \/ => 형식으로 데이터가 뽑히는 경우가 있는데, 내부적으로 JSON blob으로 인코딩해서, 슬래시를 이스케이프한단다. 그래서 json.dump로 수정.
# print(json.dumps(res.to_dict('records'),ensure_ascii=False,indent=4))

# print(food_chk("떡갈비"))

# json 형태로 뽑은 결과값.
# print(food_chk("계란김밥").to_json(force_ascii=False, orient = 'records', indent=4))

# print(food_chk("계란김밥"))
# print(food_chk("계란김밥").name)
# print(food_chk("계란김밥").imgUrl)
# print(food_chk("계란김밥").score)


