import pymysql
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori



con = pymysql.connect(host='XXX.XXX.XXX',
                      user='XXX',
                      password='XXXXXX',
                      db='XXXX',
                      charset='utf8'
                      )

# connection으로 부터 Cursor 설정.
cur=con.cursor()

sql="SELECT * FROM user_recommned_score"
cur.execute(sql)

# 데이터 가져오기
rows=cur.fetchall()
# con.close()
print(rows)

sql2="SELECT GROUP_CONCAT(fav_food SEPARATOR ',') as fav_food FROM user_recommned_score GROUP BY user_id"
cur.execute(sql2)
rows2=cur.fetchall()
print(rows2)





# numpy 라이브러리를 사용한 이중 리스트로 데이터를 뽑는다.
data=np.array(rows2)
print(data)
#
df_data=pd.DataFrame(data)
# print(df_data)
#
te=TransactionEncoder()
te_array=te.fit(data).transform(data)
df=pd.DataFrame(te_array, columns=te.columns_)
#
# print(df)

# min_support => apriori 알고리즘에서 사용되는 최소 지지도 설정.
# 최소지지도는 알고리즘 설계자가 잉의로 지정한다.
print(apriori(df,min_support=0.5,use_colnames=True))






#password='_password_',  db='access_db', charset='utf8'
