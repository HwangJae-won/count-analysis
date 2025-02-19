import os, re
from logger import logger


def load_data_from_file(file_path):
    """
    텍스트 파일 읽는 함수 
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def create_output_path(base_dir, filename):
    """
    결과 저장할 폴더 생성 
    """
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


def load_stopwords_and_mapping(stopwords_path, word_mapping_path):
    stopwords = set(load_data_from_file(stopwords_path))
    word_mapping = dict([line.split(",") for line in load_data_from_file(word_mapping_path)])
    return stopwords, word_mapping

def get_special_tokens(text, additional_tokens=None):
    """
    특수문자 포함 단어와 사용자 지정 단어를 결합하여 반환
    - text: 분석할 텍스트
    - additional_tokens: 사용자가 추가로 지정하고 싶은 단어 리스트
    """
    
    # 특수문자를 포함한 단어 추출
    special_tokens = re.findall(r"\b\w+[&\-]\w+\b", text)  # 예: R&D, K-콘텐츠
    
    # 사용자 지정 단어 추가: 확인했을 때 반영되지 않은 텍스트 
    if additional_tokens:
        special_tokens.extend(additional_tokens)
    
    return list(set(special_tokens))