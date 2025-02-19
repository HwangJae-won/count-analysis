import re, olefile, zlib, struct
import pandas as pd
from konlpy.tag import Komoran
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import Counter
from wordcloud import WordCloud
from itertools import combinations
import networkx as nx


##기본 세팅
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 한글 파일 읽기
basic_path = "/Users/hwangjaewon/Downloads/count_analylsis/"
file_path = basic_path+"recent_data/1번 문항.hwp"
custom_dic_path = basic_path+"custom_dic.txt"
#--------
result_path = basic_path+ "result/1번/"
count_plot_path = result_path+"count_plot.png"
freq_path = result_path+"count_freq.txt"
wordcloud_path =result_path+"wordcloud.png" 
network_path = result_path+"network.png"

#불용어 설정
stopwords = {"하다", "있다", "되다", "매우", "대하", "위하","관련", "들", "이제", "라고", "통하", "만들", "많", "이러", "계속", "좀",
             "그리고", "이나", "특히", "새롭", "생각",  "가지", "가지",  "보이", "지금", "형태", "가장", "주요"}
# 매핑 단어
word_mapping = {
    "R&D사업": "R&D", "게임": "게임인재원", "인공지능":"AI", 
    "기본":"기본계획","경제":"경제적"
}

#인식안돼서 수기로 추가할 단어
additional_tokens = ["AI", "실감미디어", "신사업선도전략"]  # 사용자가 추가하고 싶은 단어


def get_hwp_text(filename):
    f = olefile.OleFileIO(filename)
    dirs = f.listdir()

    # HWP 파일 검증
    if ["FileHeader"] not in dirs or \
            ["\x05HwpSummaryInformation"] not in dirs:
        raise Exception("Not Valid HWP.")

    # 문서 포맷 압축 여부 확인
    header = f.openstream("FileHeader")
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    # Body Sections 불러오기
    nums = []
    for d in dirs:
        if d[0] == "BodyText":
            nums.append(int(d[1][len("Section"):]))
    sections = ["BodyText/Section" + str(x) for x in sorted(nums)]

    # 전체 text 추출
    text = ""
    for section in sections:
        bodytext = f.openstream(section)
        data = bodytext.read()
        if is_compressed:
            unpacked_data = zlib.decompress(data, -15)
        else:
            unpacked_data = data

        # 각 Section 내 text 추출
        section_text = ""
        i = 0
        size = len(unpacked_data)
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff

            if rec_type in [67]:
                rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                section_text += rec_data.decode('utf-16')
                section_text += "\n"

            i += 4 + rec_len

        text += section_text
        text += "\n"

    return text

# 특수문자 포함 단어 추출 + 사용자 지정 단어 추가
def get_special_tokens(text, additional_tokens=None):
    """
    특수문자 포함 단어와 사용자 지정 단어를 결합하여 반환
    - text: 분석할 텍스트
    - additional_tokens: 사용자가 추가로 지정하고 싶은 단어 리스트
    """
    # 특수문자를 포함한 단어 추출
    special_tokens = re.findall(r"\b\w+[&\-]\w+\b", text)  # 예: R&D, K-콘텐츠
    
    # 사용자 지정 단어 추가
    if additional_tokens:
        special_tokens.extend(additional_tokens)
    
    # 중복 제거 후 반환
    return list(special_tokens)

# 빈도 분석 함수
def analyze_frequency(tokens):
    """
    tokens: 단어 리스트
    한 글자 단어는 제외하고 빈도 분석 수행
    """
    filtered_tokens = [token for token in tokens if len(token) > 1]  # 한 글자 단어 제거
    counter = Counter(filtered_tokens)
    return counter.most_common()

# 시각화 함수
def visualize_frequencies(frequencies,q_num, top_n=10, output_file=None):
    df = pd.DataFrame(frequencies[:top_n], columns=["단어", "빈도"])
    # 가로 크기 확대
    plt.figure(figsize=(15, 6))  # 가로 15, 세로 6
    ax = df.plot.bar(x="단어", y="빈도", legend=False, figsize=(15, 6))
    
    # x축 레이블 회전 및 간격 조정
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # 45도 회전 및 오른쪽 정렬
    
    
    plt.title(f"{q_num}단어 빈도 분석")
    plt.xlabel("단어")
    plt.ylabel("빈도")
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')  # 고해상도로 저장
        print(f"그래프가 '{output_file}'에 저장되었습니다.")

    # plt.show()
    
# 빈도 분석 결과를 텍스트로 저장하는 함수
def save_frequencies_to_text(frequencies, output_file):
    """
    빈도 분석 결과를 텍스트 파일로 저장
    frequencies: 빈도 데이터 (리스트 of 튜플)
    output_file: 저장할 파일 경로
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("단어\t빈도\n")  # 헤더 추가
        for word, freq in frequencies:
            f.write(f"{word}\t{freq}\n")
    print(f"빈도 분석 결과가 '{output_file}'에 저장되었습니다.")
# 워드 클라우드 생성 함수
def generate_wordcloud(frequencies, font_path=None, output_file=None):
    """
    frequencies: 단어 빈도 데이터 (리스트 of 튜플)
    font_path: 한글 폰트 경로 (한글 텍스트일 경우 필요)
    output_file: 저장할 파일 경로 (optional)
    """
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
    # plt.show()


# 네트워크 분석 데이터 준비
def build_cooccurrence_network(tokens, window_size=2):
    edges = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        edges.extend(combinations(window, 2))
    return edges


# 동시 출현 네트워크 생성 및 시각화
def visualize_network(edges, threshold=1, output_file = None):
    # 네트워크 생성
    G = nx.Graph()
    edge_weights = Counter(edges)
    filtered_edges = [edge for edge, weight in edge_weights.items() if weight > threshold]
    G.add_edges_from(filtered_edges)
    
    # 중심성 계산
    centrality = nx.degree_centrality(G)
    node_sizes = [v * 1000 for v in centrality.values()]  # 중심성에 따라 노드 크기 조정
    
    # 레이아웃 설정
    pos = nx.spring_layout(G, seed=42, k=0.2) 
    
    # 그래프 시각화
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="AppleGothic")
    plt.title("개선된 단어 동시 출현 네트워크")
    if output_file:
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')  # 고해상도로 저장
        print(f"그래프가 '{output_file}'에 저장되었습니다.")


    

text = get_hwp_text(file_path)
komoran = Komoran(userdic=custom_dic_path)

special_tokens = get_special_tokens(text, additional_tokens=additional_tokens)

# Komoran 분석용 텍스트 전처리
cleaned_text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]", "", text)  # 특수문자 제거
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()  # 공백 정리

# Komoran 품사 태깅
pos_tags = komoran.pos(cleaned_text)


# 원하는 품사 필터링
desired_pos = {"NNG", "NNP", "VV", "VA", "MAG", "MAJ", "XR"}  # 명사, 동사, 형용사, 부사 등
filtered_tokens = [word for word, tag in pos_tags if tag in desired_pos and word.lower() not in stopwords]


# Komoran 결과와 특수문자 포함 단어 병합
all_tokens = filtered_tokens + special_tokens

# 단어 매핑 적용
mapped_tokens = [word_mapping.get(word, word) for word in all_tokens]
    
# 빈도 분석
frequencies = analyze_frequency(mapped_tokens)

# 시각화
visualize_frequencies(frequencies,"1번 문항 ",top_n=30, output_file=count_plot_path)

# 빈도 분석 결과를 텍스트로 저장
save_frequencies_to_text(frequencies, output_file=freq_path)



# 워드 클라우드 생성
generate_wordcloud(
    frequencies,
    font_path="/Library/Fonts/AppleGothic.ttf", 
    output_file=wordcloud_path  # 파일로 저장하고 싶으면 경로 지정 (예: "wordcloud.png")
)


# 동시 출현 네트워크 생성
edges = build_cooccurrence_network(mapped_tokens, window_size=3)
# print("생성된 엣지:", edges)


visualize_network(edges, threshold=3,output_file =network_path)
