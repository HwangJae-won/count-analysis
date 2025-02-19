import os

def load_words_from_file(file_path):
    """
    텍스트 파일에서 단어 목록 로드
    - file_path: 파일 경로
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.split()[0].strip() for line in f if line.strip()}  # 첫 번째 단어만 추출 (count_freq의 경우 단어와 빈도 구분)

def compare_custom_with_freq(custom_dic_path, count_freq_path):
    """
    custom_dic.txt와 count_freq.txt를 비교하여 custom_dic에만 있는 단어 출력
    - custom_dic_path: custom_dic.txt 파일 경로
    - count_freq_path: count_freq.txt 파일 경로
    """
    # 단어 목록 로드
    custom_words = load_words_from_file(custom_dic_path)
    freq_words = load_words_from_file(count_freq_path)

    # custom_dic.txt에만 있는 단어 계산
    missing_words = custom_words - freq_words

    return missing_words

# 문항별 반복 처리
base_dir = "/Users/hwangjaewon/Downloads/count_analylsis/"
result_base_path = os.path.join(base_dir, "result")
for i in range(1, 9):
    # 파일 경로 설정
    custom_dic_path = os.path.join(base_dir, f"keyword/keyword_{i}.txt")
    count_freq_path = os.path.join(result_base_path, f"{i}번/count_freq.txt")

    # 비교 및 결과 출력
    missing_words = compare_custom_with_freq(custom_dic_path, count_freq_path)
    print(f"문항 {i}에서 keyword_{i}.txt에만 있는 단어:")
    print(missing_words)