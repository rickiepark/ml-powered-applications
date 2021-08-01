from sklearn.ensemble import RandomForestClassifier


def get_filtering_model(classifier, features, labels):
    """
    이진 분류 데이터셋에 대한 예측 에러
    :param classifier: 훈련된 분류기
    :param features: 입력 특성
    :param labels: 진짜 레이블
    """
    predictions = classifier.predict(features)
    # 에러는 1, 올바르면 0인 레이블을 만듭니다.
    is_error = [pred != truth for pred, truth in zip(predictions, labels)]

    filtering_model = RandomForestClassifier()
    filtering_model.fit(features, is_error)
    return filtering_model
