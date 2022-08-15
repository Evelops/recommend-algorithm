import mlxtend
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

data= np.array([
                 ['휴지','물티슈','샴푸'],
                 ['수세미','물티슈','비누'],
                 ['휴지','수세미','물티슈','비누'],
                 ['수세미','비누']
])

df_data=pd.DataFrame(data)
# print(df_data)
te=TransactionEncoder()
te_array=te.fit(data).transform(data)
df=pd.DataFrame(te_array, columns=te.columns_)

# print(df)

# min_support => FP_Growth 알고리즘에서 사용되는 최소 지지도 설정.
print(fpgrowth(df,min_support=0.5,use_colnames=True))
