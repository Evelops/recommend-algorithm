import mlxtend

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = ['돼지고기 두부 김치 고춧가루 양파 소금', '소고기 두부 양파 소금 대파',
             '대구 무 대파 고춧가루 미나리', '오징어 고추장 대파 ']

# tf-idf 벡터라이저 객체를 선언한다. 파이썬은 따로 객체 + 객체명이 아니라 객체명에 바로 객체를 할당 할 수 있음.
tfidf_vector = TfidfVectorizer();
# tf-idf 벡터라이저 객체와 위에서 임의로 선정한 data를 raw 값을 생성한다-> 랜덤으로 생성됨.
tfidf_vector.fit(text_data)

print(tfidf_vector.vocabulary_)
sentence = [text_data[3]]
print(tfidf_vector.transform(text_data).toarray())

# tf-idf 알고리즘이 어떤 형태로 이루어 져있는지를 확인하기 위한 shape -> 결과값 : 4, 12 라고 나오는데, 문서수:4, 추출한 텍스트 수 : 12
print(tfidf_vector.transform(text_data).shape)

