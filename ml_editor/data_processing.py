import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.sparse import vstack, hstack


def format_raw_df(df):
    """
    데이터를 정제하고 질문과 대답을 합칩니다.
    :param df: 원본 DataFrame
    :return: 처리된 DataFrame
    """
    # 타입을 고치고 인덱스를 설정합니다.
    df["PostTypeId"] = df["PostTypeId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["AnswerCount"] = df["AnswerCount"].fillna(-1)
    df["AnswerCount"] = df["AnswerCount"].astype(int)
    df["OwnerUserId"].fillna(-1, inplace=True)
    df["OwnerUserId"] = df["OwnerUserId"].astype(int)
    df.set_index("Id", inplace=True, drop=False)

    df["is_question"] = df["PostTypeId"] == 1

    # 문서화된 것 이외의 PostTypeId를 필터링합니다.
    df = df[df["PostTypeId"].isin([1, 2])]

    # 질문과 대답을 연결합니다.
    df = df.join(
        df[["Id", "Title", "body_text", "Score", "AcceptedAnswerId"]],
        on="ParentId",
        how="left",
        rsuffix="_question",
    )
    return df


def train_vectorizer(df):
    """
    벡터화 객체를 훈련합니다.
    훈련 데이터와 그 외 데이터를 변환하는데 사용할 벡터화 객체를 반환합니다.
    :param df: 벡터화 객체를 훈련하는데 사용할 데이터
    :return: 훈련된 벡터화 객체
    """
    vectorizer = TfidfVectorizer(
        strip_accents="ascii", min_df=5, max_df=0.5, max_features=10000
    )

    vectorizer.fit(df["full_text"].copy())
    return vectorizer


def get_vectorized_series(text_series, vectorizer):
    """
    사전 훈련된 벡터화 객체를 사용해 입력 시리즈를 벡터화합니다.
    :param text_series: 텍스트의 판다스 시리즈
    :param vectorizer: 사전 훈련된 sklearn의 벡터화 객체
    :return: 벡터화된 특성 배열
    """
    vectors = vectorizer.transform(text_series)
    vectorized_series = [vectors[i] for i in range(vectors.shape[0])]
    return vectorized_series


def add_text_features_to_df(df):
    """
    DataFrame에 특성을 추가합니다.
    :param df: DataFrame
    :return: 특성이 추가된 DataFrame
    """
    df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
    df = add_v1_features(df.copy())

    return df


def add_v1_features(df):
    """
    입력 DataFrame에 첫 번째 특성을 추가합니다.
    :param df: 질문 DataFrame
    :return: 특성이 추가된 DataFrame
    """
    df["action_verb_full"] = (
        df["full_text"].str.contains("can", regex=False)
        | df["full_text"].str.contains("What", regex=False)
        | df["full_text"].str.contains("should", regex=False)
    )
    df["language_question"] = (
        df["full_text"].str.contains("punctuate", regex=False)
        | df["full_text"].str.contains("capitalize", regex=False)
        | df["full_text"].str.contains("abbreviate", regex=False)
    )
    df["question_mark_full"] = df["full_text"].str.contains("?", regex=False)
    df["text_len"] = df["full_text"].str.len()
    return df


def get_vectorized_inputs_and_label(df):
    """
    DataFrame 특성과 텍스트 벡터를 연결합니다.
    :param df: 계산된 특성의 DataFrame
    :return: 특성과 텍스트로 구성된 벡터
    """
    vectorized_features = np.append(
        np.vstack(df["vectors"]),
        df[
            [
                "action_verb_full",
                "question_mark_full",
                "norm_text_len",
                "language_question",
            ]
        ],
        1,
    )
    label = df["Score"] > df["Score"].median()

    return vectorized_features, label


def get_feature_vector_and_label(df, feature_names):
    """
    벡터 특성과 다른 특성을 사용해 입력과 출력 벡터를 만듭니다.
    :param df: 입력 데이터프레임
    :param feature_names: (‘vectors’ 열을 제외한) 특성 열 이름
    :return: 특성 배열과 레이블 배열
    """
    vec_features = vstack(df["vectors"])
    num_features = df[feature_names].astype(float)
    features = hstack([vec_features, num_features])
    labels = df["Score"] > df["Score"].median()
    return features, labels


def get_normalized_series(df, col):
    """
    DataFrame 열을 정규화합니다.
    :param df: DataFrame
    :param col: 열 이름
    :return: Z-점수를 사용해 정규화된 시리즈 객체
    """
    return (df[col] - df[col].mean()) / df[col].std()


def get_random_train_test_split(posts, test_size=0.3, random_state=40):
    """
    DataFrame을 훈련/테스트 세트로 나눕니다.
    DataFrame이 질문마다 하나의 행을 가진다고 가정합니다.
    :param posts: 모든 포스트와 레이블
    :param test_size: 테스트 세트로 할당할 비율
    :param random_state: 랜덤 시드
    """
    return train_test_split(
        posts, test_size=test_size, random_state=random_state
    )


def get_split_by_author(
    posts, author_id_column="OwnerUserId", test_size=0.3, random_state=40
):
    """
    훈련 세트와 테스트 세트로 나눕니다.
    작성자가 두 세트 중에 하나에만 등장하는 것을 보장합니다.
    :param posts: 모든 포스트와 레이블
    :param author_id_column: author_id가 들어 있는 열 이름
    :param test_size: 테스트 세트로 할당할 비율
    :param random_state: 랜덤 시드
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    splits = splitter.split(posts, groups=posts[author_id_column])
    train_idx, test_idx = next(splits)
    return posts.iloc[train_idx, :], posts.iloc[test_idx, :]
