import mlxtend # apriori,fp-growth algorithm 을 활용하기 위한 모듈 호출.
import numpy as np
import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# sample data with apriori
# apriori algorithm => 가장 빈번하게 일어나는 패턴대로 추천해주는 알고리즘.
#ex_ 왓차에서 user1 이 주로 보는 영화는 액션, 판타지, 스릴러임 user1은 멜로 장르를 거의 안봄.
# -> apriori algorithm에 근거하면 위의 user1은 액션,판타지, 스릴러를 위주로 영화를 보기 때문에 그와 연관된 영화 목록을 추천해주고 나머지 장르는 추천 항목에서 제외.


# 카테고리 데이터를 행렬 형식으로 수정해준다.
# 밑의 2차원 배열에서 각 배열의 인자값이 갖는 배열의 항목들의 값에 가중치를 준다.
# 배열의 첫번째 인자값은 휴지, 물티슈, 샴푸를 갖는다. 그럼 배열의 첫번째 요소는 휴지=1, 비누=0, 샴퓨=1, 수세미=0, 휴지=1 와 같은 데이터 셋을 갖는다.
# apriori 알고리즘에서 지지도 값을 고려해야함. 여기서 지지도란, 2차원 매트릭스에서 각각의 데이터가 들어있는 빈도수를 수치화해서 나타낸 값을 지지도라고한다.
# 최소지지도 값을 임의로 정하고, 임의로 정한 최소지지도 값보다 작은 값들은 탈락 시키고, 최소지지도를 만족하는 값들만 따로 뽑아서 다시 알고리즘을 돌리는 형식으로 진행된다.

start=time.time()
data= np.array([
                 ['수박', '포도'],
                 ['사과', '딸기'],
                 ['사과', '딸기', '수박'],
                 ['딸기', '복숭아', '수박'],
                 ['사과', '딸기', '복숭아']
])

df_data=pd.DataFrame(data)
# print(df_data)

te=TransactionEncoder()
te_array=te.fit(data).transform(data)
df=pd.DataFrame(te_array, columns=te.columns_)

# print(df)

# min_support => apriori 알고리즘에서 사용되는 최소 지지도 설정.
# 최소지지도는 알고리즘 설계자가 잉의로 지정한다.
print(apriori(df,min_support=0.5,use_colnames=True))
print('측정시간:',time.time()-start)

