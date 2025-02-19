import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
from matplotlib import rc
import networkx as nx
from logger import logger
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from adjustText import adjust_text
import pandas as pd
from itertools import combinations
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities

##기본 세팅
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def filter_tokens_for_network(tokens, stopwords):
    """
    네트워크 분석용 토큰 필터링
    - 한 글자 단어와 불용어를 제외한 토큰 리스트 반환
    """
    try:
        filter_token = [token for token in tokens if len(token) > 1 and token not in stopwords]
    except Exception as e:
        logger.error(f"[Network][filter_tokens_for_network]: {e}")
        
    return filter_token


def build_cooccurrence_network(tokens, stopwords, frequencies, window_size=3):
    """동시 출현 네트워크 생성"""
    edges = []  # ✅ 기본값으로 빈 리스트 설정

    try:
        # 불용어 필터링
        filtered_tokens = [word for word in tokens if word not in stopwords]

        # 빈도 상위 100개 단어 리스트 가져오기
        if isinstance(frequencies, dict):
            top_100_words = set(list(frequencies.keys())[:30])  # ✅ 딕셔너리 처리
        elif isinstance(frequencies, list) and all(isinstance(f, tuple) and len(f) == 2 for f in frequencies):
            top_100_words = set([word for word, _ in frequencies[:30]])  # ✅ (단어, 빈도수) 튜플 처리
        else:
            top_100_words = set(frequencies)  # ✅ 단순 단어 리스트 처리

        # 필터링된 단어 중 빈도 상위 100개만 남기기
        filtered_tokens = [word for word in filtered_tokens if word in top_100_words]

        # 단어 개수가 2개 미만이면 네트워크 생성 불가
        if len(filtered_tokens) < 2:
            logger.warning(f"⚠ 동시 출현 네트워크를 만들 단어가 부족합니다.")
            return edges  # 빈 리스트 반환

        # 동시 출현 네트워크 구축
        edges = [
            combination
            for i in range(len(filtered_tokens) - window_size + 1)
            for combination in combinations(filtered_tokens[i:i + window_size], 2)
        ]
    
    except Exception as e:
        logger.error(f"[Network][build_cooccurrence_network]: {e}")

    return edges  # ✅ 이제 항상 값이 존재!

def compute_centrality_measures(edges, frequencies, output_path):
    """빈도 상위 100개 단어에 대해 연결 중심성과 고유벡터 중심성 계산 후 CSV 저장"""
    try:
        if not edges:
            logger.warning("⚠ 중심성 분석을 수행할 네트워크 엣지가 없습니다.")
            return  # 네트워크가 없으면 분석 중단
        
        G = nx.Graph()
        G.add_edges_from(edges)

        # 빈도 상위 100개 단어 가져오기
        if isinstance(frequencies, dict):
            top_100_words = set(list(frequencies.keys())[:100])
        elif isinstance(frequencies, list) and all(isinstance(f, tuple) and len(f) == 2 for f in frequencies):
            top_100_words = set([word for word, _ in frequencies[:100]])
        else:
            top_100_words = set(frequencies)  

        # 연결 중심성과 고유벡터 중심성 계산
        degree_centrality = nx.degree_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

        # 중심성 분석 결과 저장
        df = pd.DataFrame({
            "단어": list(top_100_words),
            "연결 중심성": [degree_centrality.get(word, 0) for word in top_100_words],
            "고유벡터 중심성": [eigenvector_centrality.get(word, 0) for word in top_100_words]
        })

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 중심성 분석 결과 저장 완료: {output_path}")

    except Exception as e:
        logger.error(f"[Network][compute_centrality_measures]: {e}")

# def compute_centrality_measures(edges, frequencies, output_path):
#     """빈도 상위 100개 단어에 대해 연결 중심성과 고유벡터 중심성 계산 후 CSV 저장"""
#     try:
#         G = nx.Graph()
#         G.add_edges_from(edges)

#         # 빈도 상위 100개 단어 가져오기
#         top_100_words = set([word for word, _ in frequencies[:100]])

#         # 연결 중심성과 고유벡터 중심성 계산
#         degree_centrality = nx.degree_centrality(G)
#         eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

#         # 빈도 상위 100개 단어만 필터링하여 저장
#         df = pd.DataFrame({
#             "단어": list(top_100_words),
#             "연결 중심성": [degree_centrality.get(word, 0) for word in top_100_words],
#             "고유벡터 중심성": [eigenvector_centrality.get(word, 0) for word in top_100_words]
#         })

#         # 결과를 CSV로 저장
#         df.to_csv(output_path, index=False, encoding="utf-8-sig")
#         logger.info(f"중심성 분석 결과 저장 완료: {output_path}")

#     except Exception as e:
#         logger.error(f"[Network][compute_centrality_measures]: {e}")
        
# def visualize_network(
#     edges, 
#     threshold=2, 
#     output_file=None, 
#     layout='force', 
#     uniform_color="lightblue", 
#     uniform_size=100, 
#     size_scale=500
# ):
#     """
#     필터링된 네트워크를 시각화
#     - edges: 네트워크 엣지 리스트
#     - threshold: 동시 출현 임계값
#     - uniform_color: 전체 네트워크에 동일 색상 지정 시 색상 코드
#     - uniform_size: 전체 네트워크에 동일 크기 지정 시 크기
#     - size_scale: 중심성 기반 크기 조정 스케일
#     """
#     try:
#         # 그래프 생성 및 엣지 필터링
#         G = nx.Graph()
#         edge_weights = Counter(edges)
#         filtered_edges = [edge for edge, weight in edge_weights.items() if weight >= threshold]
#         G.add_edges_from(filtered_edges)

#         # 노드 중심성 계산 (크기 조정용)
#         centrality = nx.degree_centrality(G)
        
#         # **📌 네트워크 레이아웃 설정 (노드 간 간격 증가)**
#         if layout == 'circular':
#             pos = nx.circular_layout(G)
#         else:
#             pos = nx.spring_layout(G, seed=42, k=0.5)  # 🔹 k 값 증가로 노드 간 거리 확보

#         # **📌 노드 크기 조정 (중심성이 높을수록 크기 증가)**
#         node_sizes = [centrality.get(node, 0) * size_scale + uniform_size for node in G.nodes()]

#         # **📌 네트워크 시각화**
#         plt.figure(figsize=(12, 12))
#         nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6)
#         nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=uniform_color, alpha=0.8)

#         # **📌 adjustText 적용하여 레이블 겹침 방지**
#         texts = []
#         for node, (x, y) in pos.items():
#             texts.append(plt.text(
#                 x, y, node, fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7)
#             ))

#         adjust_text(texts, only_move={'points': 'y', 'text': 'y'}, arrowprops=dict(arrowstyle="-", color='black'))

#         # 그래프 제목 및 저장
#         plt.title("네트워크 그래프", fontsize=15)
#         if output_file:
#             plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
#         plt.close()

#     except Exception as e:
#         logger.error(f"[Network][visualize_network]: {e}")      
# def visualize_network(
#     edges, 
#     threshold=2, 
#     output_file=None, 
#     layout='force', 
#     keyword=None, 
#     uniform_color=None, 
#     uniform_size=100, 
#     size_scale=500
# ):
#     """
#     필터링된 네트워크를 시각화
#     - edges: 네트워크 엣지 리스트
#     - threshold: 동시 출현 임계값
#     - uniform_color: 전체 네트워크에 동일 색상 지정 시 색상 코드
#     - uniform_size: 전체 네트워크에 동일 크기 지정 시 크기
#     - size_scale: 키워드 기반 그래프의 노드 크기 조정 스케일
#     """
#     try:
#         # 그래프 생성 및 엣지 필터링
#         G = nx.Graph()
#         edge_weights = Counter(edges)
#         filtered_edges = [edge for edge, weight in edge_weights.items() if weight >= threshold]
#         G.add_edges_from(filtered_edges)

#         # 중심성 계산
#         centrality = nx.degree_centrality(G)

#         # 전체 네트워크 설정 (일정한 크기/색상)
#         if keyword is None:
#             node_sizes = [uniform_size for _ in G.nodes()]
#             node_colors = uniform_color if uniform_color else "lightblue"

#         # 키워드 기반 네트워크 설정 (중심성 기반 크기/색상)
#         else:
#             neighbors = list(G.neighbors(keyword))
#             G = G.subgraph(neighbors + [keyword])  # 하위 그래프 생성
#             pos = nx.spring_layout(G, seed=42, k=0.3)
#             centrality = nx.degree_centrality(G)
#             node_sizes = [centrality.get(node, 0) * size_scale for node in G.nodes()]
#             node_colors = [centrality.get(node, 0) for node in G.nodes()]

#         # 레이아웃 설정
#         if layout == 'circular':
#             pos = nx.circular_layout(G)
#         else:
#             pos = nx.spring_layout(G, seed=42, k=0.3)

#         # 네트워크 시각화
#         plt.figure(figsize=(12, 12))
#         nodes = nx.draw_networkx_nodes(
#             G, 
#             pos, 
#             node_size=node_sizes, 
#             node_color=node_colors, 
#             cmap=plt.cm.Blues if keyword else None, 
#             alpha=0.8
#         )
#         nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6)
#         nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", font_family="AppleGothic",verticalalignment='bottom')

#         # 컬러바 추가 (키워드 기반 네트워크만)
#         if keyword:
#             sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
#             sm.set_array([])
#             cbar = plt.colorbar(sm, ax=plt.gca(), label="Centrality")
#             cbar.ax.tick_params(labelsize=8)

#         # 제목 및 저장
#         title = f"{keyword} 네트워크 그래프" if keyword else "전체 네트워크 그래프"
#         plt.title(title, fontsize=15)
#         if output_file:
#             plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
#         plt.close()

#     except Exception as e:
#         logger.error(f"[Network][visualize_network]: {e}")
def visualize_network(
    edges, 
    threshold=1, 
    output_file=None, 
    layout='force', 
    uniform_color="lightblue", 
    uniform_size=3000, 
    edge_scale=5
):
    """
    네트워크 그래프 시각화: 엣지 굵기를 연결 강도에 따라 설정, 텍스트 겹침 방지 및 굵게 표시
    """
    try:
        # 그래프 생성 및 엣지 필터링
        G = nx.Graph()
        edge_weights = Counter(edges)
        filtered_edges = [edge for edge, weight in edge_weights.items() if weight >= threshold]
        G.add_edges_from(filtered_edges)

        # 엣지 굵기 설정
        max_weight = max(edge_weights.values()) if edge_weights else 1
        edge_widths = [edge_weights[edge] / max_weight * edge_scale for edge in G.edges()]

        # 레이아웃 설정
        if layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=1.5)  # 노드 간 간격 넓힘

        # 그래프 시각화
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color="gray", width=edge_widths)
        nx.draw_networkx_nodes(G, pos, node_size=uniform_size, node_color=uniform_color, alpha=0.9)

        # 텍스트 위치 및 스타일 설정
        texts = []
        for node, (x, y) in pos.items():
            texts.append(plt.text(
                x, y, node, fontsize=15, ha='center', va='center', fontweight='bold'  # 글자를 굵게(fontweight='bold')
            ))

        # 텍스트 겹침 방지
        adjust_text(texts, only_move={'points': 'y', 'text': 'y'})  # 화살표 제거

        # 그래프 저장
        plt.title("네트워크 그래프", fontsize=15)
        if output_file:
            plt.savefig(output_file, format='png', dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"[Network][visualize_network]: {e}")