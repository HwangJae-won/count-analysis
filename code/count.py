import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from logger import logger
from adjustText import adjust_text

import matplotlib.pyplot as plt
from matplotlib import rc


##기본 세팅
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 빈도 분석 함수
def analyze_frequency(tokens):
    """
    tokens: 단어 리스트
    한 글자 단어는 제외하고 빈도 분석 수행
    """
    try:
        filtered_tokens = [token for token in tokens if len(token) > 1]  # 한 글자 단어 제거
        counter = Counter(filtered_tokens)
    except Exception as e:
        logger.error(f"[Count][analyze_frequency]: {e}")
    
    return counter.most_common()

# 시각화 함수
def visualize_frequencies(frequencies, q_num, top_n=30, output_file=None):
    try:
        # 데이터프레임 생성 (상위 30개 단어만)
        df = pd.DataFrame(frequencies[:top_n], columns=["단어", "빈도"])

        # 그래프 크기 조정
        plt.figure(figsize=(8, 6))

        # 막대 그래프 생성
        ax = df.plot.bar(x="단어", y="빈도", legend=False, figsize=(8, 6))

        # x축 레이블 회전
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # 그래프 제목 및 축 라벨 추가
        plt.title(f"{q_num}번 문항 단어 빈도 분석")
        plt.xlabel("단어")
        plt.ylabel("빈도")

        # 그래프 저장
        if output_file:
            plt.savefig(output_file, format='png', dpi=150, bbox_inches='tight')
            logger.info(f"count_plot 저장 완료")

        plt.close()

    except Exception as e:
        logger.error(f"[Count][visualize_frequencies]: {e}")


    
# 빈도 분석 결과를 텍스트로 저장하는 함수
def save_frequencies_to_text(frequencies, output_file):
    """
    빈도 분석 결과를 텍스트 파일로 저장
    frequencies: 빈도 데이터 (리스트 of 튜플)
    output_file: 저장할 파일 경로
    """
    try:
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("단어\t빈도\n")  # 헤더 추가
            for word, freq in frequencies[:100]: #상위 100개만 저장하도록 변경
                f.write(f"{word}\t{freq}\n")
        logger.info(f"count_freq 저장 완료")
        
    except Exception as e:
        logger.error(f"[Count][save_frequencies_to_text]: {e}")
    
    
# 워드 클라우드 생성 함수
def generate_wordcloud(frequencies, font_path=None, output_file=None):
    """
    frequencies: 단어 빈도 데이터 (리스트 of 튜플)
    font_path: 한글 폰트 경로 (한글 텍스트일 경우 필요)
    output_file: 저장할 파일 경로 (optional)
    """
    try:
        # 워드 클라우드 생성
        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color="white",
            font_path=font_path  # 한글 폰트를 지정해야 깨짐 방지
        ).generate_from_frequencies(dict(frequencies))
        
        # 워드 클라우드 시각화
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # 축 숨기기
        plt.title("Word Cloud", fontsize=20)
        
        if output_file:
            plt.savefig(output_file)
        logger.info("wordcloud 저장 완료")
        plt.close() 
            
    except Exception as e:
        logger.error(f"[Count][generate_wordcloud]: {e}")