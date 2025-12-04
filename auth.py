# auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import os
import requests

# 환경 변수 설정 (.env에 추가 필요)
COGNITO_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")

# Cognito 공개키(JWKS) URL
JWKS_URL = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user_sub(token: str = Depends(oauth2_scheme)):
    try:
        # 1. JWKS 키 가져오기 (실무에선 캐싱 권장)
        jwks = requests.get(JWKS_URL).json()

        # 2. 토큰 헤더 디코딩
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}

        # 3. 매칭되는 키 찾기
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }

        if not rsa_key:
            raise HTTPException(status_code=401, detail="Invalid token header")

        # 4. 토큰 검증 및 디코딩
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=COGNITO_CLIENT_ID,
            issuer=f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
        )

        # 5. sub 반환
        return payload.get("sub")

    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))