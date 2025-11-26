# Python 3.9 Slim 버전 사용 (가볍고 안정적)
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 1. 필수 시스템 패키지 설치 (GDAL, SpatialIndex 등)
# 이것들이 없으면 pip install geopandas/osmnx에서 에러가 납니다.
RUN apt-get update && apt-get install -y \
    g++ \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 환경 변수 설정 (GDAL 관련)
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 3. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 및 데이터 파일 복사
# (주의: 엑셀파일과 json 모델 파일이 같은 폴더에 있어야 합니다)
COPY . .

# 5. 실행 명령 (Gunicorn + Uvicorn)
# -w 4: 워커 프로세스 4개 (CPU 코어 수 * 2 + 1 권장)
# -k uvicorn.workers.UvicornWorker: Uvicorn 워커 사용
# --timeout 120: 그래프 생성 시간이 길 수 있으므로 타임아웃 넉넉히 설정
CMD ["gunicorn", "main:app", "--workers", "3", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]