import sys
import os

# 현재 파일 기준으로 src 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from data_pipeline.data_preprocessor import DataPreprocessor

data_pipeline = DataPreprocessor()
print(data_pipeline.preprocess())