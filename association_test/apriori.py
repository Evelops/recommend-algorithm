import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import time
# 실제로 사용할 메타 데이터를 기반으로 apriori 알고리즘을 먼저 작성해보자.
# 데이터를 matrix화 했을 때,생성되는 카테고리 -> 한식, 중식, 양식, 일식, 치킨, 스시, 피자, 디저트, 고기, 피자, 샐러드, 분식, 돈까스
start = time.time()
data = np.array([
    ['피자', '치킨'],
    ['한식', '치킨', '피자', '디저트'],
    ['족발', '치킨', '피자'],
    ['샐러드', '스시'],
    ['분식', '치킨'],
    ['샐러드'],
    ['양식', '치킨', '피자'],
    ['한식', '족발', '치킨'],
    ['한식', '스시'],
    ['스시', '피자', '샐러드', '스테이크'],
    ['스테이크', '샐러드', '치킨'],
    ['스시', '고기', '돈까스'],
    ['분식', '돈까스'],
    ['족발', '샐러드', '분식', '양식'],
    ['돈까스', '스시'],
    ['양식', '스시', '중식', '일식', '돈까스', '피자'],
    ['일식', '스시', '분식'],
    ['스테이크', '스시', '치킨', '스시', '피자', '샐러드', '돈까스', '분식', '중식', '한식'],
    ['스시', '피자', '샐러드', '스테이크'],
    ['스테이크', '스시', '치킨'],
    ['스테이크', '스시','한식'],
    ['샐러드', '디저트'],
    [ '치킨', '피자'],
    ['샐러드', '피자', '일식'],
    ['분식'],
    ['샐러드', '중식', '일식'],
    ['양식', '피자'],
    ['돈까스', '족발', '샐러드'],
    ['한식', '분식'],
    ['스시', '중식', '샐러드', '스테이크'],
    ['스테이크', '샐러드', '치킨'],
    ['중식', '고기', '돈까스'],
    ['분식', '한식'],
    ['족발', '피자'],
    ['돈까스'],
    ['양식', '스시', '중식', '일식', '돈까스', '피자'],
    ['양식', '분식'],
    ['스시', '치킨', '스시', '피자', '샐러드', '돈까스', '분식', '중식', '한식'],
    ['스시', '피자', '샐러드', '스테이크'],
    ['스테이크', '스시', '치킨']
])

# print(data)
df_data=pd.DataFrame(data)
# print(df_data)

te=TransactionEncoder()
te_array=te.fit(data).transform(data)
# print(te_array)
df=pd.DataFrame(te_array, columns=te.columns_)

test=apriori(df, min_support=0.4, use_colnames=True)
print(test)
print('측정시간:',time.time()-start)

# print(df)

# min_support => apriori 알고리즘에서 사용되는 최소 지지도 설정.
# 최소지지도는 알고리즘 설계자가 잉의로 지정한다.
# print(apriori(df,min_support=0.5,use_colnames=True))


