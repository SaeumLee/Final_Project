## 이상값 _low

# 데이터 준비
path = ('/content/drive/MyDrive/9.파이널프로젝트/회귀 데이터/')
train = pd.read_csv( os.path.join(path,'train.csv'), index_col='id' )
test  = pd.read_csv( os.path.join(path,'test.csv'), index_col='id' )
submission = pd.read_csv( os.path.join(path,'sample_submission.csv'), index_col='id' )
original =  pd.read_csv( os.path.join(path,'WildBlueberryPollinationSimulationData.csv'), index_col=[0])
original = original.reset_index()
original['id'] = original['Row#']
original = original.drop(columns = ['Row#']).set_index('id')
train = pd.concat([train, original])

# prompt: train에서 'yield'값이 3000이하 8000이상인 값들 추출
train_low= train[(train['yield'] < 2000)]
X_test = test[(test['fruitset'] <= 0.233554492)]
# train_high = train[(train['yield'] > 7000)]
# train_mid = train[(train['yield'] >= 3000) & (train['yield'] <= 8000)]
# print(train_low.shape, train_high.shape, train_mid.shape )


# train_lowhigh = pd.concat([train_low,train_high])
# train_lowhigh = train_lowhigh.reset_index(drop=True)
# train_lowhigh.index.name = 'id'
# X_train  = train_lowhigh.drop(['yield'],axis=1)
# y_train = train_lowhigh['yield']
# X_train  = train_low.drop(['yield'],axis=1)
# y_train = train_low['yield']

drop_cols = [
    'MinOfUpperTRange','AverageOfUpperTRange', 'AverageOfLowerTRange',
    'MaxOfUpperTRange', 'MaxOfLowerTRange',

]
train_low = train_low.drop(drop_cols, axis=1)
X_test = X_test.drop(drop_cols, axis=1)

# Add features
train_low["fruit_seed"] = train_low["fruitset"] * train_low["seeds"]
X_test["fruit_seed"] = X_test["fruitset"] * X_test["seeds"]
# train.shape, y_train.shape

numeric_columns = [_ for _ in train_low.columns if 'yield' not in _]
sc = StandardScaler() # MinMaxScaler or StandardScaler
train_low[numeric_columns] = sc.fit_transform(train_low[numeric_columns])
X_test[numeric_columns] = sc.transform(X_test[numeric_columns])
print(train_low.shape, test.shape)

time_limit = 60
predictor = TabularPredictor(label='yield', eval_metric='mean_absolute_error').fit(train_low)

pred = predictor.leaderboard(train_low, silent=True)
pred

feature_importance = predictor.feature_importance(train_low)
print(feature_importance)

# 검증해볼 데이터셋 컬럼에서 분류
train = pd.read_csv( os.path.join(path,'train.csv'), index_col='id' )
X_train = train[(train['fruitset'] <= 0.233554492)]



X_train_fruit  = X_train.drop(['yield'],axis = 1)
y_train_fruit = X_train['yield']

drop_cols = [
    'MinOfUpperTRange','AverageOfUpperTRange', 'AverageOfLowerTRange',
    'MaxOfUpperTRange', 'MaxOfLowerTRange',

]

# # Add features
X_train_fruit.drop(drop_cols, axis = 1, inplace = True)
X_train_fruit["fruit_seed"] = X_train_fruit["fruitset"] *X_train_fruit["seeds"]


numeric_columns = [_ for _ in X_train_fruit.columns]
sc = StandardScaler() # MinMaxScaler or StandardScaler
X_train_fruit[numeric_columns] = sc.fit_transform(X_train_fruit[numeric_columns])
# best_param = {'subsample_for_bin': 260000,
#               'subsample': 0.6515151515151515,
#               'reg_lambda': 0.061224489795918366,
#               'reg_alpha': 0.2040816326530612,
#               'num_leaves': 121,
#               'min_child_samples': 6,
#               'learning_rate': 0.3457879414369262,
#               'is_unbalance': False,
#               'colsample_bytree': 0.6,
#               'boosting_type': 'dart'}
print(X_train_fruit.shape)

pred_y = predictor.predict(X_train_fruit)
mae = mean_absolute_error(y_train_fruit, pred_y)
print("Blended Model MAE:", mae)

pred_t = predictor.predict(X_test)
pred_t

# prompt: submission 과 pred_t 의 동일한 인덱스에 pred_t값을 넣기
submission = pd.read_csv( '/content/submission_stack.csv', index_col='id' )
common_index = submission.index.intersection(pred_t.index)
submission.loc[common_index, 'yield'] = pred_t.loc[common_index].values
#submission.loc[16532]
submission.to_csv('submission_15.csv')



## 이상값 _high

# 데이터 준비
path = ('/content/drive/MyDrive/9.파이널프로젝트/회귀 데이터/')
train = pd.read_csv( os.path.join(path,'train.csv'), index_col='id' )
test  = pd.read_csv( os.path.join(path,'test.csv'), index_col='id' )
submission = pd.read_csv( os.path.join(path,'sample_submission.csv'), index_col='id' )
original =  pd.read_csv( os.path.join(path,'WildBlueberryPollinationSimulationData.csv'), index_col=[0])
original = original.reset_index()
original['id'] = original['Row#']
original = original.drop(columns = ['Row#']).set_index('id')
train = pd.concat([train, original])

# 원본데이터를 보고 이상치구간 중 잘 학습할만한 구간 선정
# 구간선정의 기준 = > 지정한 'yield'값 내에서 특정 수치가 반복되는 컬럼있는지 확인해서 추출
# -> 주로 fruitset~seed에 분포
train_high= train[(train['yield'] >= 8743.52098)]
X_test = test[(test['fruitset'] == 0.641617746) | (test['fruitset'] == 0.645475445) ]


# train_high = train[(train['yield'] > 7000)]
# train_mid = train[(train['yield'] >= 3000) & (train['yield'] <= 8000)]
# print(train_low.shape, train_high.shape, train_mid.shape )


# train_lowhigh = pd.concat([train_low,train_high])
# train_lowhigh = train_lowhigh.reset_index(drop=True)
# train_lowhigh.index.name = 'id'
# X_train  = train_lowhigh.drop(['yield'],axis=1)
# y_train = train_lowhigh['yield']
# X_train  = train_low.drop(['yield'],axis=1)
# y_train = train_low['yield']

drop_cols = [
    'MinOfUpperTRange','AverageOfUpperTRange', 'AverageOfLowerTRange',
    'MaxOfUpperTRange', 'MaxOfLowerTRange',

]
train_high = train_high.drop(drop_cols, axis=1)
X_test = X_test.drop(drop_cols, axis=1)

# Add features
train_high["fruit_seed"] = train_high["fruitset"] * train_high["seeds"]
X_test["fruit_seed"] = X_test["fruitset"] * X_test["seeds"]
# train.shape, y_train.shape

numeric_columns = [_ for _ in train_high.columns if 'yield' not in _]
sc = StandardScaler() # MinMaxScaler or StandardScaler
train_high[numeric_columns] = sc.fit_transform(train_high[numeric_columns])
X_test[numeric_columns] = sc.transform(X_test[numeric_columns])
print(train_high.shape, X_test.shape)

predictor = TabularPredictor(label='yield', eval_metric='mean_absolute_error').fit(train_high)

pred = predictor.leaderboard(train_high, silent=True)
pred
# feature_importance = predictor.feature_importance(train_high)
# print(feature_importance)

# 검증해볼 데이터셋 컬럼에서 분류
train = pd.read_csv( os.path.join(path,'train.csv'), index_col='id' )
X_train = train[(train['fruitset'] == 0.641617746) | (train['fruitset'] == 0.645475445)]



X_train_mass  = X_train.drop(['yield'],axis = 1)
y_train_mass = X_train['yield']

drop_cols = [
    'MinOfUpperTRange','AverageOfUpperTRange', 'AverageOfLowerTRange',
    'MaxOfUpperTRange', 'MaxOfLowerTRange',

]

# # Add features
X_train_mass.drop(drop_cols, axis = 1, inplace = True)
X_train_mass["fruit_seed"] = X_train_mass["fruitset"] *X_train_mass["seeds"]


numeric_columns = [_ for _ in X_train_mass.columns]
sc = StandardScaler() # MinMaxScaler or StandardScaler
X_train_mass[numeric_columns] = sc.fit_transform(X_train_mass[numeric_columns])
# best_param = {'subsample_for_bin': 260000,
#               'subsample': 0.6515151515151515,
#               'reg_lambda': 0.061224489795918366,
#               'reg_alpha': 0.2040816326530612,
#               'num_leaves': 121,
#               'min_child_samples': 6,
#               'learning_rate': 0.3457879414369262,
#               'is_unbalance': False,
#               'colsample_bytree': 0.6,
#               'boosting_type': 'dart'}
print(X_train_mass.shape)

pred_y = predictor.predict(X_train_mass)
mae = mean_absolute_error(y_train_mass, pred_y)
print("Blended Model MAE:", mae)

pred_t = predictor.predict(X_test)
pred_t

# prompt: submission 과 pred_t 의 동일한 인덱스에 pred_t값을 넣기
submission = pd.read_csv( 'submission_15.csv', index_col='id' )
common_index = submission.index.intersection(pred_t.index)
submission.loc[common_index, 'yield'] = pred_t.loc[common_index].values
#submission.loc[16532]
submission.to_csv('submission_16.csv')
