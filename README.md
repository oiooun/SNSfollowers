### SNS followers Regression

인스타그램 데이터를 확보하여 각 변수들을 노출도에 따른 변수, 반응에 대한 변수, 게시물에 대한 변수로 구분한 뒤 데이터 전처리를 진행하였다. 

 Y : 팔로우를 한 사람의 수(Follows)
 X : 팔로우에 영향을 주는 요소들
 F : 회귀모델링



## 주요 분석 과정
1. 전처리
- Hashtags_count: Hashtags에서 해시태그 개수 파생
- Caption_length: Caption의 길이 계산
- 수치형 데이터 표준화 (StandardScaler 사용)

2. 탐색적 데이터 분석 (EDA)
- 변수 간 상관관계 분석 및 히트맵 시각화
- 각 변수에 대한 기본 통계 확인
- Scatter plot, Box plot 등을 통한 관계 파악

3. 다중공선성 확인
- VIF(Variance Inflation Factor)를 통해 높은 상관관계 제거
- Impressions, Saves 변수 제거

4. 회귀 분석
- 선형 회귀 모델 학습 (train/test 분할)
- 회귀 계수 해석 및 중요 변수 도출
- statsmodels.OLS를 이용한 유의성 검정
- 예측값과 실제값 비교 시각화



## 모델 성능 평가
** 설명력 (R²) **
Training set: 약 0.92
Test set: 약 0.75

** RMSE (Root Mean Squared Error) **
Training set: 약 11.89
Test set: 약 14.18



[머신러닝기반데이터분석.pdf](https://github.com/user-attachments/files/21108848/default.pdf)
