from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_squared_error

# 데이터세트 로드
X, y = load_data()

# 훈련데이터와 테스트데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# 파이프라인을 통한 전처리
pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('attrubs_adder', CombinedAttributesAdder()), ('std_scaler', StandardScaler()), ])
X_train = pipeline.fit_transform(X_train)

# 모델 선언 및 학습
model = Model()
model.fit(X_train, y_train)

# 모델 테스트세트 예측
X_test = pipeline.transform(X_test)
y_pred = model.predict(X_test)

# 모델 평가 : RMSE

rmse = np.sqrt(mse)

X_test = full_pipeline.transform(X_test)
Y_pred = lin_reg.predict(X_test)

