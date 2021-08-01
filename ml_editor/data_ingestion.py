import os
from pathlib import Path

from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import pandas as pd

from ml_editor.data_processing import format_raw_df, add_v1_features
from ml_editor.model_v2 import add_v2_text_features


def generate_model_text_features(raw_df_path, save_path=None):
    """
    모델 2를 위한 특성을 생성하고 디스크에 저장하는 함수
    이 특성을 계산하는데 몇 분 정도 걸립니다.
    :param raw_df_path: (parse_xml_to_csv에서 생성한) 원본 DataFrame 경로
    :param save_path: 처리된 DataFrame을 저장할 경로
    :return: 처리된 DataFrame
    """
    df = pd.read_csv(raw_df_path)
    df = format_raw_df(df.copy())
    df = df.loc[df["is_question"]].copy()
    df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")

    df = add_v1_features(df.copy())
    df = add_v2_text_features(df.copy())

    if save_path:
        df.to_csv(save_path)
    return df


def parse_xml_to_csv(path, save_path=None):
    """
    .xml 포스트 덤프를 열어 텍스트에서 csv로 변환합니다.
    :param path: 포스트가 담긴 xml 문서의 경로
    :return: 처리된 텍스트의 데이터프레임
    """

    # 파이썬의 표준 라이브러리로 XML 파일을 파싱합니다.
    doc = ElT.parse(path)
    root = doc.getroot()

    # 각 행은 하나의 질문입니다.
    all_rows = [row.attrib for row in root.findall("row")]

    # tdqm을 사용해 전처리 과정을 표시합니다.
    for item in tqdm(all_rows):
        # HTML에서 텍스트를 추출합니다.
        soup = BeautifulSoup(item["Body"], features="html.parser")
        item["body_text"] = soup.get_text()

    # 딕셔너리의 리스트에서 데이터프레임을 만듭니다.
    df = pd.DataFrame.from_dict(all_rows)
    if save_path:
        df.to_csv(save_path)
    return df


def get_data_from_dump(site_name, load_existing=True):
    """
    .xml 덤프를 로드하고, 파싱하여 csv로 만들고, 직렬화한 다음 반환합니다.
    :param load_existing: 기존에 추출한 csv를 로드할지 새로 생성할지 결정합니다.
    :param site_name: 스택익스체인지 웹사이트 이름
    :return: 파싱된 xml의 판다스 DataFrame
    """
    data_path = Path("data")
    dump_name = "%s.stackexchange.com/Posts.xml" % site_name
    extracted_name = "%s.csv" % site_name
    dump_path = data_path / dump_name
    extracted_path = data_path / extracted_name

    if not (load_existing and os.path.isfile(extracted_path)):
        all_data = parse_xml_to_csv(dump_path)
        all_data.to_csv(extracted_path)
    else:
        all_data = pd.DataFrame.from_csv(extracted_path)

    return all_data
