import sys, os, olefile, re,  zlib, struct
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from logger import logger
from utils import get_special_tokens

def get_hwp_text(filename):
    """
    한글 파일 전체 텍스트 읽기 
    - filename: 한글 파일 경로 
    """
    try:
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
        logger.info("한글 텍스트 추출 완료")
        
    except Exception as e:
        logger.error(f"[Process][get_hwp_text]: {e}")

    return text



def preprocess_text(text, komoran, stopwords, additional_tokens=None):
    try:
        cleaned_text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]", "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        pos_tags = komoran.pos(cleaned_text)
        desired_pos = {"NNG", "NNP", "VV", "VA", "MAG", "MAJ", "XR"}
        filtered_tokens = [word for word, tag in pos_tags if tag in desired_pos and word not in stopwords]
        special_tokens = get_special_tokens(text, additional_tokens)
        logger.info("텍스트 전처리 완료")
        
    except Exception as e:
        logger.error(f"[Process][preprocess_text]: {e}")
    return filtered_tokens + special_tokens