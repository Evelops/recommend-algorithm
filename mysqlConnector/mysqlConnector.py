import pymysql
con = pymysql.connect(host='wte-rds-server.c16ihe3nmzly.ap-northeast-2.rds.amazonaws.com',
                      user='wte_rds',
                      password='djaEhd0426',
                      db='ETY_DB',
                      charset='utf8'
                      )

# connection으로 부터 Cursor 설정.
cur=con.cursor()

sql="SELECT * FROM user_recommned_score"
cur.execute(sql)

# 데이터 가져오기
rows=cur.fetchall()
print(rows)

con.close()


#password='_password_',  db='access_db', charset='utf8'
