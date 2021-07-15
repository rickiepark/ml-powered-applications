# 머신러닝 파워드 애플리케이션

![Book cover](/images/ML_Powered_cover.jpg)

이 저장소는 [머신러닝 파워드 애플리케이션](https://tensorflow.blog/mlpa/)(한빛미디어, 2021)의 코드 저장소입니다.

이 저장소는 세 부분으로 구성됩니다:
- `notebook` 폴더에는 책에서 소개된 개념을 설명하기 위한 주피터 노트북이 담겨 있습니다.

- `ml_editor` 폴더에는 책의 예제인 머신러닝 보조 글쓰기 애플리케이션을 위한 핵심 함수가 들어 있습니다.

- 플래스크 앱은 사용자에게 결과를 제공하는 간단한 예시입니다.

- `images/bmlpa_figures` 폴더에는 첫 번째 버전에서 알아 보기 힘든 몇 개의 그림이 포함되어 있습니다.

이 저장소의 코드를 리뷰해 준 [Bruno Guisard](https://www.linkedin.com/in/bruno-guisard/)에게 감사합니다.

## 설정 안내

### 파이썬 환경

이 저장소의 코드는 파이썬 3.6과 3.7에서 테스트되었습니다. 다른 파이썬 3 버전에서도 작동할 것입니다.

먼저 이 저장소를 클론합니다:

`git clone https://github.com/rickiepark/ml-powered-applications.git`

그다음 저장소 폴더에서 [virtualenv](https://pypi.org/project/virtualenv/)를 사용해 파이썬 가상환경을 만듭니다:

`cd ml-powered-applications`

`virtualenv ml_editor`

다음 명령으로 이 환경을 활성화합니다:

`source ml_editor/bin/activate`

그다음 다음 명령을 사용해 필요한 패키지를 설치합니다:

`pip install -r requirements.txt`

예제 프로젝트는 spacy에 있는 영어 모델을 사용합니다. 영어 모델을 다운로드하려면 virtualenv가 활성화된 터미널에서 다음 명령을 실행하니다:

`python -m spacy download en_core_web_sm`

`python -m spacy download en_core_web_lg`

마지막으로 노트북과 라이브러리 코드는 `nltk` 패키지를 사용합니다. 이 패키지에는 개별적으로 다운로드할 수 있는 여러 자료가 있습니다. 다운로드하려면 활성화된 가상 환경에서 파이썬 세션을 오픈한 후 `import nltk`하고 필요한 자료를 다운로드합니다.

다음은 `nltk`가 설치된 가상 환경에서 `punkt` 패키지를 다운로드하는 예입니다:

`python`

`import nltk`

`nltk.download('punkt')`

## 주피터 노트북

노트북 폴더에는 책에서 다루는 개념을 위한 예제 코드를 담고 있습니다. 대부분의 예제는 아카이브(writers.stackexchange.com 데이터)에 있는 서브폴더 중 하나만 사용합니다.

번거로움을 줄이기 위해 전처리된 데이터를 `.csv` 파일로 포함시켰습니다.

직접 이 데이터를 생성하고 싶거나 다른 폴더에 데이터를 생성하고 싶다면 다음을 참고하세요:

- 스택익스체인지 [아카이브][archives]에서 한 서브폴더를 다운로드합니다.

- `parse_xml_to_csv`을 실행해 데이터프레임으로 변환합니다.

- `generate_model_text_features`을 실행해 미리 계산된 특성을 포함한 데이터프레임을 생성합니다.

[archives]: https://archive.org/details/stackexchange

이 노트북들은 다음과 같은 몇 개의 카테고리로 나눌 수 있습니다.

### 데이터 탐색과 변환

- [데이터셋 탐색][DatasetExploration]
- [데이터 분할][SplittingData]
- [텍스트 벡터화][VectorizingText]
- [데이터 군집][ClusteringData]
- [표 데이터 벡터화][TabularDataVectorization]
- [특성 생성을 위한 데이터 탐색][ExploringDataToGenerateFeatures]

### 초기 모델 훈련과 성능 분석

- [간단한 모델 훈련][TrainSimpleModel]
- [데이터와 예측 비교Comparing Data To Predictions][ComparingDataToPredictions]
- [탑 K][TopK]
- [특성 중요도][FeatureImportance]
- [블랙 박스 설명 도구][BlackBoxExplainer]

### 모델 향상

- [두 번째 모델][SecondModel]
- [세 번째 모델][ThirdModel]

### 모델 비교

- [모델 비교][ComparingModels]

### 모델을 사용한 추천 생성

- [추천 생성][GeneratingRecommendations]

[BlackBoxExplainer]: ./notebooks/black_box_explainer.ipynb
[ClusteringData]: ./notebooks/clustering_data.ipynb
[ComparingDataToPredictions]: ./notebooks/comparing_data_to_predictions.ipynb
[ComparingModels]: ./notebooks/comparing_models.ipynb
[DatasetExploration]: ./notebooks/dataset_exploration.ipynb
[ExploringDataToGenerateFeatures]: ./notebooks/exploring_data_to_generate_features.ipynb
[FeatureImportance]: ./notebooks/feature_importance.ipynb
[GeneratingRecommendations]: ./notebooks/generating_recommendations.ipynb
[SecondModel]: ./notebooks/second_model.ipynb
[SplittingData]: ./notebooks/splitting_data.ipynb
[TabularDataVectorization]: ./notebooks/tabular_data_vectorization.ipynb
[ThirdModel]: ./notebooks/third_model.ipynb
[TopK]: ./notebooks/top_k.ipynb
[TrainSimpleModel]: ./notebooks/train_simple_model.ipynb
[VectorizingText]: ./notebooks/vectorizing_text.ipynb

## 사전 훈련된 모델

`notebook` 폴더에 있는 노트북을 사용해 모델을 훈련하고 저장할 수 있습니다.
`models` 폴더에는 세 개의 훈련된 모델과 두 개의 벡터화 객체가 저장되어 있습니다.
모델의 결과를 비교하고 플라스크 앱에서 사용하기 위해 이 모델들을 사용합니다.

## 플라스크 앱 실행하기

앱을 실행하려면 저장소 루트 폴더로 이동하여 다음 명령을 실행하세요:

`FLASK_APP=app.py flask run`

위 명령은 로컬 웹 앱을 구동하며 `http://127.0.0.1:5000/`로 접속할 수 있습니다.