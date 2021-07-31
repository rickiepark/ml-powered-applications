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