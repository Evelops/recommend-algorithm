import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 사용하고자 하는 csv 데이터 셋을 pndas의 read_csv 메서드를 통해서 불러온다.
wte_food_data=pd.read_csv('/Users/eomseung-yeol/PycharmProjects/recommend-algorithm/csv/wte_food_data.csv')
# print(wte_food_data)

df = pd.DataFrame(wte_food_data)
# print(df)
# print(df.query('feature == "김치"'))
# csv 파일을 pandas의 DataFrame으로 변경하고 데이터 프레임에서 특정 속성의 키워드가 포함되어 있는 항목을 추출해서 뽑을 수 있다. query 속성에 false를 주면 포함하는 애들을 제외하고 출력한다.
# print(df.query('feature.str.contains("마늘|양파|된장|치즈")==False'))

testWd = "마늘,양파,된장,치즈"
kwds = [] # 파이썬의 리스트.

tesky = df.query('feature.str.contains("")')

# print(tesky)

for i in testWd.split(","):
    tesky=tesky.query('feature.str.contains(@i)==False')
    # print(df.query('feature.str.contains(@kwds)==False')) # str.contains에 할당한 변수 값을 넣어주어서 추출할때는 다음과 같이 @를 변수명 앞에 붙인다.

print(tesky) # 최종적으로 필터링 된 값.

# print(wte_food_data.shape)
# print(wte_food_data.feature)

tfidf=TfidfVectorizer().fit(wte_food_data) # tfidf 알고리즘을 동작시키기 위한 vectorizer 객체.
tfidf_matrix = tfidf.fit_transform(wte_food_data.feature) # 비교해야되는 대상이 각 음식이 가지고 있는 feature 이기 때문에 데이터에서 feature 칼럼만 추출.
# print("tfidf_matrix")
# print(tfidf_matrix.shape) #feature에 대해서 tf-idf를 진행한 결과값 => (80,92) 80->feature 갯수 ,92-> 서로 다른 유니크한 속성값의 갯수

# tf-idf 로직을 수행한 데이터셋에 대해서 cosine 유사도를 구한 결과값을 리턴.
res_cosine=linear_kernel(tfidf_matrix, tfidf_matrix)
# print(res_cosine)

# pandas 시리즈 배열을 통해서, 인덱스가 음식명이고, 인덱스번호가 값인 판다스 시리즈 배열 선언.
idx = pd.Series(wte_food_data.index, index=wte_food_data.name).drop_duplicates()
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
    sims_cosine = sims_cosine[1:6]

    # 가장 유사한 데이터 5개의 인덱스를 받아온다.
    food_idx = [i[0] for i in sims_cosine]

    result_df = wte_food_data.iloc[food_idx].copy()
    result_df['score'] = [i[1] for i in sims_cosine]

    # 결과값에서 특징 속성 제거.
    del result_df['feature']
    return result_df

# print(food_chk("계란김밥"))
# print(food_chk("계란김밥").name)
# print(food_chk("계란김밥").imgUrl)
# print(food_chk("계란김밥").score)


# 개발 프로세스 -> 서버에서 DB를 조회해서 가져옴. 이때, 불필요한 재료가 있는 경우, 쿼리로 비선호 재료가 들어있는 row를 제거하고 가져옴.
