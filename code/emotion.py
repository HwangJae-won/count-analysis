from transformers import pipeline
import pandas as pd
from logger import logger
from collections import defaultdict
import os

def extract_context_sentences(text, keywords, window_size=5):
    """
    텍스트 데이터에서 키워드별 문맥(문장)을 추출
    - text: 전체 텍스트 (str)
    - keywords: 문맥을 추출할 키워드 리스트
    - window_size: 키워드 주변 단어를 포함한 문맥 크기
    - return: {단어: [문장들]} 형태의 딕셔너리
    """
    context_sentences = defaultdict(list)

    # 텍스트를 문장 단위로 분리
    sentences = text.split(".")

    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence:
                # 키워드가 포함된 문장 추출
                context_sentences[keyword].append(sentence.strip())

    return context_sentences

sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment_with_context(frequencies, context_sentences, output_path):
    """
    문맥을 기반으로 빈도 상위 100개 단어의 감정 분석 수행
    - 단어가 포함된 문맥에서 감정 분석을 수행하여 긍/부정 비율을 기반으로 최종 감정 결정
    """
    try:
        sentiment_data = []

        for word, freq in frequencies[:100]:  # 상위 100개 단어만 분석
            if word not in context_sentences or not context_sentences[word]:
                continue  # 문맥이 없는 단어는 스킵
            
            positive_count = 0
            negative_count = 0
            total_count = 0

            for sentence in context_sentences[word]:  # 해당 단어가 포함된 문장들
                try:
                    # 문맥 기반 감정 분석 수행
                    result = sentiment_analyzer(sentence)
                    sentiment = result[0]['label'].lower()  # 예: "POSITIVE", "NEGATIVE", "NEUTRAL"

                    if "positive" in sentiment:
                        positive_count += 1
                    elif "negative" in sentiment:
                        negative_count += 1

                    total_count += 1

                except Exception as e:
                    logger.warning(f"문장 감정 분석 중 오류 발생: {e}")

            # 단어별 최종 감정 결정 (중립 제외)
            if total_count == 0:
                continue  # 문맥 분석이 안 된 단어는 제외
            
            if positive_count > negative_count:
                final_sentiment = "긍정"
            elif negative_count > positive_count:
                final_sentiment = "부정"
            else:
                final_sentiment = "중립" # 감정 분류가 애매한 경우 제외

            sentiment_data.append({"단어": word, "빈도": freq, "감정": final_sentiment})

        # 📌 감정 분석 결과가 없는 경우 로그 출력 후 종료
        if not sentiment_data:
            logger.error("⚠ 감정 분석 결과가 없습니다! 빈 데이터셋을 확인하세요.")
            return


        # 결과를 데이터프레임으로 저장
        df = pd.DataFrame(sentiment_data)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 감정 분석 결과 저장 완료: {output_path}")

    except Exception as e:
        logger.error(f"[Sentiment][analyze_sentiment_with_context]: {e}")