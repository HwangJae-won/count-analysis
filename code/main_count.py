import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from logger import logger
from konlpy.tag import Komoran
import pandas as pd
from process import get_hwp_text, preprocess_text
from count import analyze_frequency, visualize_frequencies, save_frequencies_to_text, generate_wordcloud
from network import build_cooccurrence_network, visualize_network, compute_centrality_measures
from utils import load_data_from_file, create_output_path, get_special_tokens, load_stopwords_and_mapping, save_frequencies_to_text

def update_custom_dic(custom_dic_path, new_words):
    """✅ 빈도수가 0인 키워드를 custom_dic.txt 형식(NNP)으로 추가"""
    existing_words = set()
    if os.path.exists(custom_dic_path):
        with open(custom_dic_path, "r", encoding="utf-8") as f:
            existing_words = set(line.strip() for line in f.readlines())

    updated_words = existing_words.union({f"{word}\tNNP" for word in new_words})
    
    with open(custom_dic_path, "w", encoding="utf-8") as f:
        for word in sorted(updated_words):
            f.write(f"{word}\n")
    
    logger.info(f"✅ 사용자 사전 업데이트 완료: {custom_dic_path}")

def main():
    base_dir = "/Users/hwangjaewon/Downloads/count_analylsis/"
    recent_data_path = os.path.join(base_dir, "final_data")
    result_base_path = os.path.join(base_dir, "result/result_final")
    stopwords_path = os.path.join(base_dir, "dictionary/stopwords.txt")
    word_mapping_path = os.path.join(base_dir, "dictionary/word_mapping.txt")
    custom_dic_path = os.path.join(base_dir, "dictionary/custom_dic.txt")
    additional_tokens = ['신기술', "AI", "없습니다",'디지털콘텐츠과', '태재대학교','자율규제위원회','테스트베드', '로블록스', '비대면',
                        '혁신융합대학사업', '정부주도', '게이미피케이션', 'XR실증센터', '지적재산권', '정보통신기획평가원', 
                        '해외기술연수', 'IP', '디지털콘텐츠육성사업', '연구과제', '중재안', '가상융합사업',
                        '게임산업진흥법', '컴투버스', 'IoT','NFT', '패스트트랙',
                        '이프랜드', '벤처캐피탈','VC','한국지능정보사회진흥원','셧다운제', 'P2E','IP밸류체인', '문화산업공정유통법',  
                        '중소기업벤처부','서울경제진흥원', '수익모델', '코리아메타버스페스티벌', '공간컴퓨팅','모르겠습니다',
                        '웹소설', '카테고리', '진흥원', '디지털교과서', '밸류체인', '안전성', '라이프스타일', '산업적', '공론화', '고령화', 
                        '사업화', '산업통산자원부', '간접자본', '바우처', '가상환경', '수용성', '대학생', 
                        '사회적', '신뢰성', '인재양성', '대학원생', '선순환', '진흥법', 'SNS', '금전적',  '시범사업',
                        '부처간', '산업적', '고령화', '심리적', '국가인권위원회', '스마트시티', '예외규정', '탄소중립', '사회적', 
                        '디지털전환','사회적문제', '진흥법', '메타버스캠퍼스', '기후변화', '윤리적', '인공지능', '실감형','신기술',
                        '사행성', '민관협력', '유기적', '예외조항', '단계별', '지역특구', '벤처캐피털', '일상', '제도화', '바우처', 
                        '정보통신전략위원회', '교육계', '오픈이노베이션', '연계성', '디지털전환', '의견수렴', '금전적', '인공지능', 
                        '통합적', '실효성', '필요성', '자율성', '캐즘', '경직성',
                        '현실세계', '실증특례', '카테고리', '파괴적혁신', '대학원', '스마트팩토리', '종합적', '재구조화', 
                        '유기적', '예외조항', '대학교', '공론화', '법제화', '사례발굴', '담론의장', '안정적', 
                        '사업화', '칼리버스', '평가위원회', '바우처', '국방부', '오픈이노베이션', '기술적', '공공기관', 
                        '사행성규제', '진흥법', '구체화', '인공지능', '자율성','비활성화', '진흥법', '공론화', '기술개발', 
                        '콘텐츠진흥법', '가상융합정책', '가이드', '인공지능', '사업화', '기획재정부', '세부적']

    stopwords, word_mapping = load_stopwords_and_mapping(stopwords_path, word_mapping_path)
    komoran = Komoran(userdic=custom_dic_path)

    logger.info("Starting Analysis!!")
    
    def load_keywords(file_path):
        if not os.path.exists(file_path):
            logger.warning(f"⚠ 키워드 파일이 존재하지 않습니다: {file_path}")
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    
    for i in range(1, 9):
        logger.info(f"Processing file: {i}번 문항")
        file_name = f"{i}번 문항.hwp"
        file_path = os.path.join(recent_data_path, file_name)
        result_path = os.path.join(result_base_path, f"{i}번")
        os.makedirs(result_path, exist_ok=True)
        
        keyword_path = os.path.join(base_dir, f"keyword/keyword_{i}.txt")
        keywords = load_keywords(keyword_path)
        
        if not keywords:
            logger.warning(f"⚠ {i}번 문항의 키워드 파일이 비어 있습니다. count_freq.txt를 생성하지 않습니다.")
            continue

        text = get_hwp_text(file_path)
        tokens = preprocess_text(text, komoran, stopwords, additional_tokens)
        mapped_tokens = [word_mapping.get(word, word) for word in tokens]

        keyword_frequencies = {kw: mapped_tokens.count(kw) for kw in keywords}
        save_frequencies_to_text(
            dict(sorted(keyword_frequencies.items(), key=lambda item: item[1], reverse=True)),
            create_output_path(result_path, "count_freq.txt")
            )
        
        zero_freq_keywords = {kw for kw, freq in keyword_frequencies.items() if freq == 0}
        update_custom_dic(custom_dic_path, zero_freq_keywords)
        komoran = Komoran(userdic=custom_dic_path)

        tokens = preprocess_text(text, komoran, stopwords, additional_tokens)
        mapped_tokens = [word_mapping.get(word, word) for word in tokens]
        keyword_frequencies = {kw: mapped_tokens.count(kw) for kw in keywords}
        save_frequencies_to_text(
            dict(sorted(keyword_frequencies.items(), key=lambda item: item[1], reverse=True)),
            create_output_path(result_path, "count_freq.txt")
            )
        
        visualize_frequencies(sorted(keyword_frequencies.items(), key=lambda x: x[1], reverse=True), i, output_file=create_output_path(result_path, "count_plot.png"))
        
        words_for_network = list(keyword_frequencies.keys())
        edges = build_cooccurrence_network(mapped_tokens, stopwords, words_for_network, window_size=2)
        
        if not edges:
            logger.warning(f"⚠ {i}번 문항에서 네트워크 엣지가 존재하지 않습니다. 중심성 분석을 건너뜁니다.")
            continue
        
        centrality_output_path = create_output_path(result_path, "centrality_measures.csv")
        compute_centrality_measures(edges, words_for_network, output_path=centrality_output_path)
        
        try:
            df_centrality = pd.read_csv(centrality_output_path).sort_values(by=['고유벡터 중심성'], ascending=False)
            df_centrality.to_csv(centrality_output_path, index=False, encoding='utf-8-sig', float_format="%.3f")
        except pd.errors.EmptyDataError:
            logger.warning(f"⚠ 중심성 분석 파일이 비어 있습니다: {centrality_output_path}")
            df_centrality = pd.DataFrame(columns=["단어", "연결 중심성", "고유벡터 중심성"])
        
        visualize_network(edges, threshold=2, output_file=create_output_path(result_path, "full_network_plot.png"), uniform_color="lightblue", uniform_size=150)
        
        df_frequency = pd.DataFrame(list(keyword_frequencies.items()), columns=["단어", "빈도수"])
        df_combined = df_frequency.merge(df_centrality, on="단어", how="left").fillna(0).sort_values(by='빈도수', ascending=False)
        df_combined.to_csv(create_output_path(result_path, "word_analysis_table.csv"), index=False, encoding="utf-8-sig", float_format="%.3f")
        
        logger.info(f"✅ {i}번 문항 분석 완료")
    
    logger.info("🎯 모든 문항 분석 완료!")

if __name__ == "__main__":
    main()
