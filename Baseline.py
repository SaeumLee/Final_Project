# 필요한 라이브러리 설치
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# 데이터 경로 설정
path = './data'  # 실제 데이터 경로에 맞게 수정

# 데이터 로드
train = pd.read_csv(os.path.join(path, 'train.csv'), index_col='id')
test = pd.read_csv(os.path.join(path, 'test.csv'), index_col='id')
submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'), index_col='id')

# 데이터 확인
print(train.shape, test.shape, submission.shape)
print(train.head(3))

# 데이터 분포 시각화
sns.displot(train['yield'])
plt.title("Yield Distribution")
plt.show()

# 피처 & 타겟 분리
X = train.drop(['yield'], axis=1)
y = train['yield']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.9)

# 모델 학습
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mae_score = mean_absolute_error(y_test, y_pred)
print(f'MAE Score: {mae_score:.4f}')
