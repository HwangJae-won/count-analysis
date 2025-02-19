import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from logger import logger
from konlpy.tag import Komoran
from process import get_hwp_text, preprocess_text
from count import analyze_frequency, visualize_frequencies, save_frequencies_to_text, generate_wordcloud
from network import build_cooccurrence_network, visualize_network,compute_centrality_measures
from utils import load_data_from_file, create_output_path,get_special_tokens, load_stopwords_and_mapping
from emotion import analyze_sentiment_with_context, extract_context_sentences


def main():
    # 기본 경로 설정
    base_dir = "/Users/hwangjaewon/Downloads/count_analylsis/"
    recent_data_path = os.path.join(base_dir, "recent_data")
    result_base_path = os.path.join(base_dir, "result_v3")
    stopwords_path = os.path.join(base_dir, "stopwords.txt")
    word_mapping_path = os.path.join(base_dir, "word_mapping.txt")

    # 불용어 및 단어 매핑 로드
    stopwords, word_mapping=  load_stopwords_and_mapping(stopwords_path, word_mapping_path)
    # Komoran 설정
    komoran = Komoran(userdic=os.path.join(base_dir, "custom_dic.txt"))
    additional_tokens = ["AI", "없습니다",'디지털콘텐츠과', '태재대학교','자율규제위원회','테스트베드', '로블록스', '비대면',
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
                         '콘텐츠진흥법', '가상융합정책', '가이드', '인공지능', '사업화', '기획재정부', '세부적', '신기술']

    # 1번부터 6번 문항 반복 처리
    logger.info("Starting Analysis!!")
    
    for i in range(1, 9):
        logger.info(f"Processing file: {i}번 문항")
        file_name = f"{i}번 문항.hwp"
        file_path = os.path.join(recent_data_path, file_name)
        result_path = os.path.join(result_base_path, f"{i}번")


        # 텍스트 추출
        text = get_hwp_text(file_path)

        # 텍스트 전처리 및 토큰화
        tokens = preprocess_text(text, komoran, stopwords, additional_tokens)

        # 단어 매핑 적용
        mapped_tokens = [word_mapping.get(word, word) for word in tokens]

        # 빈도 분석
        frequencies = analyze_frequency(mapped_tokens)

        # # 시각화 및 결과 저장
        # visualize_frequencies(frequencies, i, top_n=30, output_file=create_output_path(result_path, "count_plot.png"))
        # save_frequencies_to_text(frequencies, create_output_path(result_path, "count_freq.txt"))
        # # generate_wordcloud(frequencies, font_path="/Library/Fonts/AppleGothic.ttf", output_file=create_output_path(result_path, "wordcloud.png"))

        
        # # 빈도 상위 100개 단어를 기반으로 네트워크 생성
        # edges = build_cooccurrence_network(tokens, stopwords, frequencies, window_size=3)

        # compute_centrality_measures(edges, frequencies, 
        #                             output_path=create_output_path(result_path, "centrality_measures.csv"))
    
        # # # 최빈 단어 3개 가져오기
        # # top_keywords = [word for word, freq in frequencies[:3]]
        # # logger.info(f"최빈 단어 3개: {top_keywords}")

        # # 전체 네트워크 그래프 시각화
        # visualize_network(
        #                 edges, 
        #                 threshold=3, 
        #                 output_file= create_output_path(result_path, "full_network_plot.png"), 
        #                 uniform_color="lightblue", 
        #                 uniform_size=150  # 노드 크기 고정
        #             )
        # logger.info(f'{i}번 full 네트워크 시각화 완료')
        # 감정 분석 수행
        # 3. 문맥 자동 추출
        keywords = [word for word, _ in frequencies[:100]]
        context_sentences = extract_context_sentences(text, keywords)

        # 📌 문맥 확인 (디버깅)
        logger.info(f"🔍 추출된 문맥 예시: {list(context_sentences.items())[:5]}")

        # 4. 감정 분석 수행
        output_path = create_output_path(result_path,"sentiment_analysis.csv")

        analyze_sentiment_with_context(
            frequencies, 
            context_sentences, 
            output_path=output_path
        )

        logger.info("✅ 프로세스 완료: 감정 분석 결과가 저장되었습니다.")

        logger.info(f'{i}번 감정분석 완료')

        # # 최빈 단어별 필터링 및 그래프 생성
        # for keyword in top_keywords:
        #     # 키워드가 포함된 엣지만 필터링
        #     filtered_edges = [
        #         edge for edge in edges if keyword in edge
        #     ]
            
        #     # 그래프 저장
        #     output_file = create_output_path(result_path, f"{keyword}_network.png")
        #     visualize_network(
        #                 edges, 
        #                 threshold=1, 
        #                 output_file=output_file, 
        #                 keyword=keyword, 
        #                 size_scale=500  # 중심성에 따른 크기 스케일 조정
        #             )
        #     logger.info(f'{i}번 {keyword} 네트워크 시각화 완료')

        logger.info(f"Completed processing for {i}번 문항")
        
        
    logger.info("Processing complete for all files.")

if __name__ == "__main__":
    main()