import os
from pathlib import Path

import spacy
import joblib
from tqdm import tqdm
import pandas as pd
import nltk
from scipy.sparse import vstack, hstack

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer


POS_NAMES = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

FEATURE_ARR = [
    "num_questions",
    "num_periods",
    "num_commas",
    "num_exclam",
    "num_quotes",
    "num_colon",
    "num_stops",
    "num_semicolon",
    "num_words",
    "num_chars",
    "num_diff_words",
    "avg_word_len",
    "polarity",
]
FEATURE_ARR.extend(POS_NAMES.keys())

SPACY_MODEL = spacy.load("en_core_web_sm")
tqdm.pandas()

curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_2.pkl")
vectorizer_path = Path("../models/vectorizer_2.pkl")
VECTORIZER = None
MODEL = None


def count_each_pos(df):
    """
    품사의 등장 횟수를 세어 입력 DataFrame에 추가합니다.
    :param df: SPACY_MODEL로 전달된 텍스트를 담고 있는 입력 DataFrame
    :return: 등장 회수가 포함된 DataFrame
    """
    global POS_NAMES
    pos_list = df["spacy_text"].apply(lambda doc: [token.pos_ for token in doc])
    for pos_name in POS_NAMES.keys():
        df[pos_name] = (
            pos_list.apply(
                lambda x: len([match for match in x if match == pos_name])
            )
            / df["num_chars"]
        )
    return df


def get_word_stats(df):
    """
    단어 카운트 같은 통계적 특성을 DataFrame에 추가합니다.
    :param df: 훈련 세트의 질문을 full_text 열에 담고 있는 DataFrame
    :return: 새로운 열이 추가된 DataFrame
    """
    global SPACY_MODEL
    df["spacy_text"] = df["full_text"].progress_apply(lambda x: SPACY_MODEL(x))

    df["num_words"] = (
        df["spacy_text"].apply(lambda x: 100 * len(x)) / df["num_chars"]
    )
    df["num_diff_words"] = df["spacy_text"].apply(lambda x: len(set(x)))
    df["avg_word_len"] = df["spacy_text"].apply(lambda x: get_avg_wd_len(x))
    df["num_stops"] = (
        df["spacy_text"].apply(
            lambda x: 100 * len([stop for stop in x if stop.is_stop])
        )
        / df["num_chars"]
    )

    df = count_each_pos(df.copy())
    return df


def get_avg_wd_len(tokens):
    """
    단어 리스트가 주어지면 단어의 평균 길이를 반환합니다.
    :param tokens: 단어 배열
    :return: 단어 당 평균 문자 개수
    """
    if len(tokens) < 1:
        return 0
    lens = [len(x) for x in tokens]
    return float(sum(lens) / len(lens))


def add_char_count_features(df):
    """
    구둣점 문자 개수를 DataFrame에 추가합니다.
    :param df: 훈련 세트의 질문을 full_text 열에 담고 있는 DataFrame
    :return: 카운트가 추가된 DataFrame
    """
    df["num_chars"] = df["full_text"].str.len()

    df["num_questions"] = 100 * df["full_text"].str.count("\?") / df["num_chars"]
    df["num_periods"] = 100 * df["full_text"].str.count("\.") / df["num_chars"]
    df["num_commas"] = 100 * df["full_text"].str.count(",") / df["num_chars"]
    df["num_exclam"] = 100 * df["full_text"].str.count("!") / df["num_chars"]
    df["num_quotes"] = 100 * df["full_text"].str.count('"') / df["num_chars"]
    df["num_colon"] = 100 * df["full_text"].str.count(":") / df["num_chars"]
    df["num_semicolon"] = 100 * df["full_text"].str.count(";") / df["num_chars"]
    return df


def get_sentiment_score(df):
    """
    nltk를 사용해 입력 질문의 극성 점수(polarity score)를 반환합니다.
    :param df: 훈련 세트의 질문을 full_text 열에 담고 있는 DataFrame
    :return: 극성 점수가 추가된 DataFrame
    """
    sid = SentimentIntensityAnalyzer()
    df["polarity"] = df["full_text"].progress_apply(
        lambda x: sid.polarity_scores(x)["pos"]
    )
    return df


def add_v2_text_features(df):
    """
    모델 v2에 사용할 여러 특성을 DataFrame에 추가합니다.
    :param df: 훈련 세트의 질문을 full_text 열에 담고 있는 DataFrame
    :return: 특성 열이 추가된 DataFrame
    """
    df = add_char_count_features(df.copy())
    df = get_word_stats(df.copy())
    df = get_sentiment_score(df.copy())
    return df


def get_model_probabilities_for_input_texts(text_array):
    """
    질문이 높은 점수를 받을 가능성을 나타내는 확률 점수의 배열을 반환합니다.
    포맷: [ [prob_low_score1, prob_high_score_1], ... ]
    :param text_array: 점수를 매길 질문의 배열
    :return: 예측 확률 배열
    """
    global FEATURE_ARR, VECTORIZER, MODEL, curr_path, vectorizer_path, model_path
    if VECTORIZER == None:
        VECTORIZER = joblib.load(curr_path / vectorizer_path)
    if MODEL == None:
        MODEL = joblib.load(curr_path / model_path)
    vectors = VECTORIZER.transform(text_array)
    text_ser = pd.DataFrame(text_array, columns=["full_text"])
    text_ser = add_v2_text_features(text_ser.copy())
    vec_features = vstack(vectors)
    num_features = text_ser[FEATURE_ARR].astype(float)
    features = hstack([vec_features, num_features])
    return MODEL.predict_proba(features)


def get_question_score_from_input(text):
    """
    하나의 샘플 질문에 대한 양성 클래스의 확률을 얻기 위한 헬퍼 함수
    :param text: 입력 문자열
    :return: 높은 점수를 받는 질문의 예측 확률
    """
    preds = get_model_probabilities_for_input_texts([text])
    positive_proba = preds[0][1]
    return positive_proba


def get_pos_score_from_text(input_text):
    """
    플래스크 앱에 출력할 점수를 구합니다.
    :param input_text: 입력 문자열
    :return: 높은 점수를 받는 질문의 예측 확률
    """
    positive_proba = get_question_score_from_input(input_text)
    output_str = (
        """
        질문 점수 (0는 최악, 1은 최상):
        <br/>
        %s
    """
        % positive_proba
    )

    return output_str
