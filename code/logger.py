import logging
from logging.handlers import RotatingFileHandler

# 로거 초기화
LOG_FILE = '250126-test.log'
LOG_LEVEL = logging.INFO

def setup_logger():
    """
    공통 로거 설정 함수
    - 하나의 로거를 모든 모듈에서 공유
    """
    logger = logging.getLogger("app_logger")  # 공통 로거 이름 설정
    logger.setLevel(LOG_LEVEL)

    # 파일 핸들러
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)  # 1MB로 제한
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    # 중복 핸들러 방지
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger

# 공통 로거 생성
logger = setup_logger()