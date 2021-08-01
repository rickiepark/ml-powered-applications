import os
from pathlib import Path

import pandas as pd
import joblib
from scipy.sparse import vstack, hstack

from ml_editor.data_processing import add_v1_features

FEATURE_ARR = [
    "action_verb_full",
    "question_mark_full",
    "text_len",
    "language_question",
]

curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_1.pkl")
vectorizer_path = Path("../models/vectorizer_1.pkl")
VECTORIZER = joblib.load(curr_path / vectorizer_path)
MODEL = joblib.load(curr_path / model_path)


def get_model_probabilities_for_input_texts(text_array):
    """
    질문이 높은 점수를 받을 가능성을 나타내는 확률 점수의 배열을 반환합니다.
    포맷: [ [prob_low_score1, prob_high_score_1], ... ]
    :param text_array: 점수를 매길 질문의 배열
    :return: 예측 확률 배열
    """
    global FEATURE_ARR, VECTORIZER, MODEL
    vectors = VECTORIZER.transform(text_array)
    text_ser = pd.DataFrame(text_array, columns=["full_text"])
    text_ser = add_v1_features(text_ser)
    vec_features = vstack(vectors)
    num_features = text_ser[FEATURE_ARR].astype(float)
    features = hstack([vec_features, num_features])
    return MODEL.predict_proba(features)


def get_model_predictions_for_input_texts(text_array):
    """
    질문 배열에 대한 레이블 배열을 반환합니다.
    True는 높은 점수, False는 낮은 점수입니다.
    포맷: [ False, True, ...]
    :param text_array:  분류할 질문의 배열
    :return: 클래스 배열
    """
    probs = get_model_probabilities_for_input_texts(text_array)
    predicted_classes = probs[:, 0] < probs[:, 1]
    return predicted_classes
