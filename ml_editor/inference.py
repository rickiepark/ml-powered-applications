"""inference.py: 이 모듈을 책의 예제를 서빙하기 위한 스텁(stub) 함수를 담고 있습니다.
이 함수는 ml_editor나 노트북에서 사용되지 않습니다.
"""

from functools import lru_cache

REQUIRED_FEATURES = [
    "is_question",
    "action_verb_full",
    "language_question",
    "question_mark_full",
    "norm_text_len",
]


def find_absent_features(data):
    missing = []
    for feat in REQUIRED_FEATURES:
        if feat not in data.keys():
            missing.append(feat)
    return missing


def check_feature_types(data):
    types = {
        "is_question": bool,
        "action_verb_full": bool,
        "language_question": bool,
        "question_mark_full": bool,
        "norm_text_len": float,
    }
    mistypes = []
    for field, data_type in types:
        if not isinstance(data[field], data_type):
            mistypes.append((data[field], data_type))
    return mistypes


def run_heuristic(question_len):
    pass


@lru_cache(maxsize=128)
def run_model(question_data):
    """
    스텁 함수입니다. 실제로 app.py에서 lru_cache를 사용합니다.
    :param question_data:
    """
    # 아래 느린 모델 추론을 추가하세요.
    pass


def validate_and_handle_request(question_data):
    missing = find_absent_features(question_data)
    if len(missing) > 0:
        raise ValueError("누락된 특성: %s" % missing)

    wrong_types = check_feature_types(question_data)
    if len(wrong_types) > 0:
        # 데이터가 잘못되었지만 질문의 길이가 있다면 경험 규칙을 실행합니다.
        if "text_len" in question_data.keys():
            if isinstance(question_data["text_len"], float):
                return run_heuristic(question_data["text_len"])
        raise ValueError("잘못된 타입: %s" % wrong_types)

    return run_model(question_data)


def verify_output_type_and_range(output):
    if not isinstance(output, float):
        raise ValueError("잘못된 출력 타입: %s, %s" % (output, type(output)))
    if not 0 < output < 1:
        raise ValueError("범위 밖의 출력: %s" % output)


def validate_and_correct_output(question_data, model_output):
    # 타입과 범위를 검사해 적절히 에러를 발생시킵니다.
    try:
        # 모델 출력이 잘못되면 에러를 발생시킵니다.
        verify_output_type_and_range(model_output)
    except ValueError:
        # 경험 규칙을 실행하지만 다른 모델을 실행할 수도 있습니다.
        run_heuristic(question_data["text_len"])

    # 에러가 발생되지 않으면 모델 결과를 반환합니다.
    return model_output
