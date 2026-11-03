# -*- coding: utf-8 -*-
"""
퀀트랩_BIS_4.py
=====================================================================
[논문] Ahmed & Aldasoro (2026), "Stablecoins and safe asset prices",
       BIS Working Paper No 1270 (2026년 6월 개정판)
[단계] 복제 4단계: 3개월 T-bill 금리 반응 (논문 그림 7) 재현
       국소 투영(local projection, 식 (4)) - 통제만(왼쪽 패널) / 통제 + GIV 도구변수(오른쪽 패널)

[출처 표기 규칙]  (퀀트랩_BIS_1~3-1 과 같음)
  # 출처: 논문 ○○절 / 각주 ○ / 그림 ○ 주석 / 부록 ○   -> 논문 근거
  # 출처: 사용자 지시 / 사용자 확정 (날짜)               -> 연구자가 정한 조건
  # 출처: 퀀트랩_BIS_○.py                              -> 앞 단계에서 확정한 사항
  # [논문에 명시 없음]                                 -> 논문에 방법이 없어 정해야 했던 부분 (이유·영향 함께)
  # [대체 자료]                                        -> 논문 자료(유료 등)를 구할 수 없어 바꾼 자료

[식 (4)] (출처: 논문 3.3절)
  y3M_{t+h} − y3M_{t−1} = α_h + φ_h [y3M_{t−1} − y3M_{t−5}] + β_h Flow(5d)_t
                          + Σ_k γ_h^k [y^k_{t+h} − y^k_{t−1}]   (k = FFR, 6M, 1Y)
                          + θ1 Q_t + θ2 1[Q_t > 0] + θ3 MPS_t + Λ' X_t + e_{t+h}
  IV: Flow(5d)_t 를 GIV(5일 합)로 도구 (출처: 3.3절 마지막 문단, 그림 7 주석)

[확정된 설계] (출처: 사용자 확정 10/10)
  - 대체할 수 있는 변수는 대체, 구할 수 없는 변수는 뺌
      대체: 통화정책 충격(3개월 OIS, LSEG) -> SF 연준 USMPD
      뺌  : MMF 유입(iMoneyNet), CESI(Bloomberg), 국채시장 여건지수 MCI(BIS 내부 지수)
  - Newey-West 시차: 시차 1 과 겹침 시차를 둘 다 보고, 판정은 겹침 시차로
  - ETF 유입: (B) 주 분석 = ETF 빼고 2026-03-31 까지 / (A) 비교 = ETF 넣고 2025-12-31 까지
  - 재현 판정 규칙 3개 ([셀 8]) - 결과 보기 전에 확정
  - 사전 추세 점검 추가 ([셀 9]) - 기록용 (판정에는 안 씀)

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  스테이블코인_합산시총_일별.csv      (퀀트랩_BIS_1 [셀 2], 종료일 2026-03-31)
  GIV_선택조합_평일.csv              (퀀트랩_BIS_2 [셀 6])
  교란변수_원자료_일별.csv            (퀀트랩_BIS_3 [셀 5]: VIX, S&P 500, 달러 NEER, Bitcoin, Ether)
  대체교란변수_원자료_일별.csv        (퀀트랩_BIS_3-1 [셀 2]: 금·브렌트 선물)
  ETF유입_일별.csv                   (퀀트랩_BIS_3-1 [셀 5])
  통화정책충격_USMPD.xlsx            (SF 연준 사이트에서 직접 받아 저장 - [셀 4] 설명 참고)
[인터넷에서 받는 것] FRED (금리·ON-RRP·TGA), 미 재무부 Fiscal Data (3개월 T-bill 입찰)
[출력]
  데이터/4단계_FRED_원자료_일별.csv, 4단계_TB입찰_원자료.csv, 4단계_회귀자료_평일.csv
  데이터/4단계_그림7_결과.csv, 4단계_판정.csv, 4단계_사전추세.csv
  그림/그림7_재현_3개월금리.png, 그림7_재현_ETF포함비교.png

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install requests pandas numpy statsmodels matplotlib openpyxl
=====================================================================
"""

#%% [셀 0] 패키지 + 설정 =================================================
import os
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt

# FRED API 키 (출처: 사용자 지시 - FRED API 사용)
#   키는 코드에 적지 않고 프로젝트 폴더의 FRED_키.txt 에서 읽음 (퀀트랩_BIS_1.py 와 같음)
_키파일 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "FRED_키.txt"   # GitHub에 올리지 않는 파일
FRED_API_KEY = os.environ.get("FRED_API_KEY") or (
    _키파일.read_text(encoding="utf-8").strip() if _키파일.exists() else "여기에_FRED_키_붙여넣기")
FRED주소 = "https://api.stlouisfed.org/fred/series/observations"
FRED정보주소 = "https://api.stlouisfed.org/fred/series"
FD주소 = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"

# 폴더 (출처: 사용자 지시 - 바탕화면/스테이블코인_퀀트랩, 한국어 파일명)
결과폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩"
데이터폴더 = 결과폴더 / "데이터"
그림폴더 = 결과폴더 / "그림"
그림폴더.mkdir(parents=True, exist_ok=True)
CSV인코딩 = "utf-8-sig"

# 표본 기간 (출처: 논문 표 1 주석 "Sample: January 4, 2021 to March 31, 2026")
표본시작 = pd.Timestamp("2021-01-04")
표본끝 = pd.Timestamp("2026-03-31")
# [논문에 명시 없음] 예측 기간 끝(t+h)이 표본끝을 넘는 관측치는 쓰지 않음
#   이유: 논문 자료가 2026-03-31 에서 끝나므로 같은 조건. 그래서 금리도 표본끝까지만 받음

# 평일 달력 (출처: 퀀트랩_BIS_2.py [셀 0] - 표 1·표 5 의 T = 1,367 로 역산, 각주 13)
평일 = pd.bdate_range(표본시작, 표본끝)
받기시작 = pd.Timestamp("2020-12-01")            # 5일 변화용 (출처: 퀀트랩_BIS_3.py [셀 5]와 같음)
평일_넓게 = pd.bdate_range(받기시작, 표본끝)

# 예측 기간 h = 0 ~ 40 거래일 (출처: 그림 7 가로축 "Horizon (days)" 0 ~ 40)
예측기간 = list(range(0, 41))

# 효과 크기: $3.5B 유입 (출처: 3.4절 "a $3.5B aggregate stablecoin inflow (roughly two standard
#   deviations in our sample)", 그림 7 제목) -> 결과는 bp (금리 % × 100)
충격 = 3.5

# Newey-West 시차 (출처: 사용자 확정 10/10 - 둘 다 보고, 판정은 겹침 시차)
#   (1) 시차 1: 논문 표 2 주석 "Newey-West standard errors with one lag" (1단계 회귀의 표기)
#   (2) 겹침 시차 = h + 4
#       이유: t 관측치는 Flow(5d)_t (t−4 ~ t 의 5일 변화)와 y_{t+h} − y_{t−1} (t ~ t+h 의 h+1 일 변화)
#             를 함께 써서 t−4 ~ t+h 의 h+5 일 정보를 담음 -> 이웃 관측치가 h+4 일까지 정보를 공유
#       출처: Hansen & Hodrick (1980) - 겹치는 관측치의 오차는 겹치는 길이만큼 상관
#   [논문에 명시 없음] 논문은 국소 투영의 NW 시차를 밝히지 않음
def 겹침시차(h):
    return h + 4

# 유의수준 (출처: 그림 7 주석 "68% and 95% Newey-West confidence bands" -> 판정은 95% = 5%)
유의수준 = 0.05
Z68, Z95 = 0.994458, 1.959964       # 표준정규 68%, 95% 양측 구간의 z (그림 7 주석의 두 구간)

# 논문 결과 (비교용) (출처: 3.4절)
#   통제만: "less than 1 bp within 10 days, with a trough of about 1.5 bps at 20 days"
#   IV    : "contemporaneous effect of 0.71 basis points, an effect of about 4 bps within 10 days,
#            deepening to a trough of about 5 bps at 13 days, ... around 3 to 4 bps over horizons of 10 to 25 days"
논문IV = {0: -0.71, 10: -4.0, 13: -5.0}
논문통제 = {20: -1.5}

# 표 5 (T, 평균, 표준편차, 최소, 최대) - 자료 판별용 (출처: 부록 C.2 표 5)
표5 = {
    "3개월": (1310, 3.37, 2.10, 0.01, 5.63), "1개월": (1310, 3.30, 2.14, 0.00, 6.02),
    "6개월": (1310, 3.40, 2.01, 0.02, 5.61), "1년": (1310, 3.33, 1.87, 0.04, 5.49),
    "2년": (1310, 3.22, 1.64, 0.09, 5.19), "10년": (1310, 3.41, 1.14, 0.93, 4.98),
    "FFR": (1314, 3.24, 2.07, 0.05, 5.33),
    "TGA": (1365, 652, 294, 44.8, 1669), "ONRRP": (1308, 945, 844, 0.00, 2554),
    "입찰": (1367, 14.4, 29.4, 0.00, 96.1),
    "MPS": (1367, 0.002, 0.40, -11.2, 4.00),
    "Flow5": (1362, 0.88, 1.74, -4.18, 11.5), "ETF5": (1362, 0.58, 1.59, -4.21, 9.41),
}

if "붙여넣기" in FRED_API_KEY:
    raise ValueError("[셀 0]: 프로젝트 폴더에 FRED_키.txt 를 만들고 FRED API 키를 한 줄로 넣으세요 (또는 환경변수 FRED_API_KEY).")
print("데이터 폴더:", 데이터폴더)
print(f"평일 수: {len(평일)} (표 5: 1,367), 예측 기간 h = 0 ~ {예측기간[-1]}")


#%% [셀 1] 판별 함수 + 기존 자료 읽기 ======================================
def 요약(s):
    """표본(평일 2021-01-04 ~ 2026-03-31) 안의 T, 평균, 표준편차, 최소, 최대"""
    s = s.reindex(평일)
    return [int(s.notna().sum()), s.mean(), s.std(), s.min(), s.max()]


def 고르기(후보, 표5키, 이름):
    """
    여러 후보 시계열 중 표 5와 가장 가까운 것을 고름
      1순위: T 차이(휴장일 처리 방식이 같은지), 2순위: 평균·표준편차·최소·최대 차이의 합
      출처: 퀀트랩_BIS_3.py [셀 6]·3-1 [셀 3] 과 같은 "표 5로 판별" 방식 (사용자 확정 10/10)
    반환: 고른 후보 이름
    """
    목표 = 표5[표5키]
    행 = []
    for k, s in 후보.items():
        v = 요약(s)
        행.append([k, *v, abs(v[0] - 목표[0]), sum(abs(a - b) for a, b in zip(v[1:], 목표[1:]))])
    표 = pd.DataFrame(행, columns=["후보", "T", "평균", "표준편차", "최소", "최대", "T차이", "값차이합"])
    표 = 표.sort_values(["T차이", "값차이합"]).reset_index(drop=True)
    print(f"\n[{이름}] 표 5: T={목표[0]}, 평균 {목표[1]}, 표준편차 {목표[2]}, 최소 {목표[3]}, 최대 {목표[4]}")
    print(표.round(3).to_string(index=False))
    print(f"  -> 선택: {표.loc[0, '후보']}")
    return 표.loc[0, "후보"]


def 읽기(파일, 날짜열="date"):
    return pd.read_csv(데이터폴더 / 파일, parse_dates=[날짜열], index_col=날짜열, encoding=CSV인코딩)


# 5일 유입 (출처: 퀀트랩_BIS_2.py [셀 1] - 식 (1) Flow(5d)_t = MC_t − MC_{t−5}, 평일 기준, $B)
합산 = 읽기("스테이블코인_합산시총_일별.csv")
if 합산.index.max() < 표본끝:
    raise ValueError("합산시총 CSV 가 표본끝 전에 끝납니다. 퀀트랩_BIS_1 을 종료일 2026-03-31 로 실행하세요.")
MC = 합산["Total"].reindex(평일_넓게) / 1e9
Flow5 = (MC - MC.shift(5)).rename("Flow5")

# GIV 5일 합 (출처: 퀀트랩_BIS_2.py [셀 6] - 저자 답신(2026-10-12)으로 확정한 설정,
#   그림 7 주석 "instruments 5-day stablecoin flows ... with the granular IV")
GIV5 = 읽기("GIV_선택조합_평일.csv")["GIV_5일합"].reindex(평일_넓게).rename("GIV5")

# 교란변수 (출처: 퀀트랩_BIS_3.py [셀 5], 3-1 [셀 2])
교란 = 읽기("교란변수_원자료_일별.csv")
대체 = 읽기("대체교란변수_원자료_일별.csv")

# 휴장일: 앞값 채움 (출처: 퀀트랩_BIS_3.py [셀 6] - 표 5 에서 VIX, S&P 500, 금, 브렌트, NEER 의 T = 1,367)
def 앞값채움(s):
    return s.reindex(s.index.union(평일_넓게)).sort_index().ffill().reindex(평일_넓게)

수준 = pd.DataFrame({
    "VIX": 앞값채움(교란["VIX"]),
    "SP500": 앞값채움(교란["S&P 500"]),
    "NEER": 앞값채움(교란["달러 NEER"]),
    # 금·브렌트: 선물 (출처: 퀀트랩_BIS_3-1 [셀 3] - 브렌트 선물이 표 5와 다섯 값 모두 일치, 금은 평균만 0.01 차이)
    "금": 앞값채움(대체["금"]),
    "브렌트": 앞값채움(대체["브렌트"]),
    # BTC·ETH: 주말에도 거래되어 평일 값이 모두 있음 (출처: 2.1절 Yahoo Finance, 퀀트랩_BIS_3.py [셀 5])
    "BTC": 교란["Bitcoin"].reindex(평일_넓게),
    "ETH": 교란["Ether"].reindex(평일_넓게),
})

# ETF 유입 (출처: 퀀트랩_BIS_3-1 [셀 5] - 7개(운용사+CRSP), 앞값 채움이 표 1과 가장 가까움)
ETF경로 = 데이터폴더 / "ETF유입_일별.csv"
if ETF경로.exists():
    ETF일별 = 읽기("ETF유입_일별.csv")["7개 (운용사+CRSP)_앞값 채움"].reindex(평일_넓게)
    # 5일 ETF 유입 = 5 거래일 유입의 합 (출처: 표 1·표 5 "5-day ETF flow", 퀀트랩_BIS_3-1 [셀 5])
    ETF5 = ETF일별.rolling(5).sum().rename("ETF5")
    ETF끝 = ETF일별.dropna().index.max()
    print(f"ETF 유입: {ETF끝.date()} 까지")
else:
    ETF5, ETF끝 = None, None
    print("!! ETF유입_일별.csv 가 없습니다 -> (A) ETF 포함 비교는 건너뜀 (사용 설명 참고)")

print(f"5일 유입: {요약(Flow5)}  (표 5: {표5['Flow5']})")
print(f"  우리 자료의 2 표준편차 = {2 * Flow5.reindex(평일).std():.2f} $B  (효과 크기는 논문과 같은 $3.5B 로 계산)")
print(f"GIV 5일 합: T={요약(GIV5)[0]}  (표 5: 1,367)")


#%% [셀 2] FRED: 금리, FFR, ON-RRP, TGA 받기 + 표 5로 판별 ==================
# 출처: 2.1절 "We collect daily data on US market interest rates from FRED"
#       "we collect the Treasury General Account (TGA) balance and the Overnight Reverse
#        Repurchase Facility (ON-RRP) balances from FRED"
# [논문에 명시 없음] FRED 시리즈 코드 -> 후보를 받아 표 5로 판별
#   3개월: DGS3MO(만기 수익률) / DTB3(유통시장 할인율)  (출처: 퀀트랩_BIS_1.py [셀 0] 주석, 통합 보고서 "남은 결정")
#   FFR  : EFFR(뉴욕 연준 실효금리, 영업일만) / DFF(같은 금리, 주말·휴일 포함)
#   TGA  : WTREGEN(수요일 잔액) / WDTGAL(주 평균) - 둘 다 주간 -> 앞값 채움
#          (표 5 TGA 의 T = 1,365 가 평일 전체에 가까워 주간 값을 채운 것으로 보임)
#   ON-RRP: RRPONTSYD (일별)
FRED_받을것 = {"DGS1MO": "1개월", "DGS3MO": "3개월", "DTB3": "3개월", "DGS6MO": "6개월",
             "DGS1": "1년", "DGS2": "2년", "DGS10": "10년",
             "EFFR": "FFR", "DFF": "FFR", "RRPONTSYD": "ONRRP", "WTREGEN": "TGA", "WDTGAL": "TGA"}


def FRED받기(코드):
    """관측값 + 단위. FRED 는 빈 값을 "." 으로 줌 -> 빈 값 (FRED 자료 형식)"""
    자료 = requests.get(FRED주소, timeout=120, params={
        "series_id": 코드, "api_key": FRED_API_KEY, "file_type": "json",
        "observation_start": 받기시작.strftime("%Y-%m-%d"),
        "observation_end": 표본끝.strftime("%Y-%m-%d")}).json()
    정보 = requests.get(FRED정보주소, timeout=120, params={
        "series_id": 코드, "api_key": FRED_API_KEY, "file_type": "json"}).json()
    단위 = 정보["seriess"][0]["units"]
    s = pd.Series({pd.to_datetime(o["date"]): (np.nan if o["value"] == "." else float(o["value"]))
                   for o in 자료["observations"]}, name=코드)
    return s, 단위


FRED원 = {}
for 코드 in FRED_받을것:
    s, 단위 = FRED받기(코드)
    # 잔액 단위를 $B 로 (출처: 표 5 "Treasury General Account ($B)", "ON-RRP take-up ($B)")
    if 코드 in ("RRPONTSYD", "WTREGEN", "WDTGAL"):
        if "Millions" in 단위:
            s = s / 1000
        elif "Billions" not in 단위:
            raise ValueError(f"{코드} 단위 확인 필요: {단위}")
    FRED원[코드] = s
    print(f"  {코드}: {s.notna().sum():,}개, 단위 {단위}")
FRED원 = pd.DataFrame(FRED원)
FRED원.index.name = "date"
FRED원.to_csv(데이터폴더 / "4단계_FRED_원자료_일별.csv", encoding=CSV인코딩)
print("저장: 4단계_FRED_원자료_일별.csv")

# 평일 달력에 맞추기
#   금리·FFR·ON-RRP: 채우지 않음 (출처: 표 5 - 금리 T = 1,310, FFR 1,314, ON-RRP 1,308 < 1,367
#                    -> 논문은 이 변수들의 휴장일을 비워 둠)
#   TGA: 주간 -> 앞값 채움 (위 설명)
def 평일로(코드):
    return 앞값채움(FRED원[코드]) if 코드 in ("WTREGEN", "WDTGAL") else FRED원[코드].reindex(평일_넓게)


선택 = {}
for 표5키 in ["3개월", "FFR", "TGA"]:
    후보 = {코드: 평일로(코드) for 코드, 키 in FRED_받을것.items() if 키 == 표5키}
    선택[표5키] = 고르기(후보, 표5키, 표5키)
for 표5키 in ["1개월", "6개월", "1년", "2년", "10년", "ONRRP"]:
    코드 = [c for c, k in FRED_받을것.items() if k == 표5키][0]
    선택[표5키] = 고르기({코드: 평일로(코드)}, 표5키, 표5키)       # 후보 1개 - 확인만

금리 = pd.DataFrame({키: 평일로(코드) for 키, 코드 in 선택.items()})


#%% [셀 3] 미 재무부 Fiscal Data: 3개월 T-bill 입찰 받기 + 표 5로 판별 ==========
# 출처: 2.1절 "T-bill supply is measured using US Treasury auction issuance from the US Department
#       of the Treasury" / 3.3절 "we control for both the quantity of issuance, Q_t, and issuance
#       days with an indicator variable" / 표 5 "3-month T-bill auction ($B)" T = 1,367
# [대체 자료 아님] 논문과 같은 출처 (재무부). 단, 데이터셋 이름은 논문에 없음
#   -> Fiscal Data "Treasury Securities Auctions Data" (auctions_query)
# [논문에 명시 없음] 금액 항목(제시 금액 / 낙찰 금액)과 날짜(입찰일 / 발행일) -> 후보를 모두 만들어 표 5로 판별
# 3개월물 = 13주 만기 Bill (출처: 재무부 표기 "13-Week")
# 열 이름은 받아 보고 정함 (자료 안내서와 실제 열 이름이 다른 경우가 있었음 - 10/9 확인: total_accepted_amt 없음)
행들, 쪽 = [], 1
while True:
    r = requests.get(FD주소, timeout=120, params={
        "filter": f"security_type:eq:Bill,security_term:eq:13-Week,record_date:gte:{받기시작.date()}",
        "page[size]": 1000, "page[number]": 쪽}).json()
    if "data" not in r:
        raise ValueError(f"Fiscal Data 응답 확인 필요: {str(r)[:300]}")
    행들 += r["data"]
    if 쪽 >= r["meta"]["total-pages"]:
        break
    쪽 += 1
입찰 = pd.DataFrame(행들)
print(f"받은 행: {len(입찰)}, 열: {list(입찰.columns)}")
입찰.to_csv(데이터폴더 / "4단계_TB입찰_원자료.csv", index=False, encoding=CSV인코딩)

# 현금관리용 Bill(CMB) 제외 (정기 13주 Bill 만) - 열이 있을 때만
#   [논문에 명시 없음] 논문은 "3-month T-bill auction" 이라고만 씀. CMB 는 만기가 불규칙한 임시 발행이라 뺌
if "cash_management_bill_cmb" in 입찰.columns:
    print(f"  CMB 제외: {(입찰['cash_management_bill_cmb'] == 'Yes').sum()}건")
    입찰 = 입찰[입찰["cash_management_bill_cmb"] != "Yes"]

금액후보 = [c for c in ["offering_amt", "total_accepted", "total_accepted_amt"] if c in 입찰.columns]
날짜후보 = [c for c in ["auction_date", "issue_date"] if c in 입찰.columns] or \
    [c for c in ["record_date"] if c in 입찰.columns]      # 안내서: record_date = 입찰일
if not 금액후보 or not 날짜후보:
    raise ValueError(f"금액·날짜 열을 찾지 못했습니다. 위 열 목록을 알려주세요: {list(입찰.columns)}")

입찰후보 = {}
for 금액 in 금액후보:
    for 날짜 in 날짜후보:
        x = pd.DataFrame({"날짜": pd.to_datetime(입찰[날짜], errors="coerce"),
                          "금액": pd.to_numeric(입찰[금액], errors="coerce") / 1e9})      # 달러 -> $B
        일별 = x.dropna().groupby("날짜")["금액"].sum()
        # 입찰이 없는 평일 = 0 (출처: 3.3절 "the daily-frequency issuance data has many zeros", 표 5 T = 1,367)
        입찰후보[f"{금액} / {날짜}"] = 일별.reindex(평일_넓게).fillna(0.0)
입찰선택 = 고르기(입찰후보, "입찰", "3개월 T-bill 입찰 ($B)")
Q = 입찰후보[입찰선택].rename("Q")


#%% [셀 4] SF 연준 USMPD: 통화정책 충격 (대체) + 표 5로 판별 ==================
# 논문: "changes in 3-month OIS rates in a 30-minute window around FOMC announcements (30 minutes
#        before to 30 minutes after)" (3.3절), LSEG 장중 자료 (2.1절) -> 유료
# [대체 자료] SF 연준 U.S. Monetary Policy Event-Study Database (USMPD)
#   출처: Acosta, Ajello, Bauer, Loria & Miranda-Agrippino (2025), SF 연준 WP 2025-30 / https://sffed.us/USMPD
#   논문과 다른 점 (출처: WP 2025-30)
#     ① 창: "starts 10 minutes before and ends 20 minutes after the release of the statement" (논문은 −30 ~ +30분)
#     ② 3개월 OIS 없음: OIS 는 "one- and two-year OIS rates" 뿐. 대신 연방기금 선물(FF1~FF6, MP1, MP2),
#        Eurodollar/SOFR 선물(ED1~ED8) 등이 있음
#     ③ 단위: 퍼센트포인트 -> × 100 = bp (표 5 단위 "bps")
#   후보 (단기 정책금리 기대를 재는 열, 3개월 OIS 와 만기가 가까운 순서로 넣음):
#     MP1(이번 회의 놀람), MP2(다음 회의), FF2, FF3(2·3번째 달 연방기금 선물), ED1(첫 분기 선물)
#     [제외] 3개월 T-bill 변화: 종속변수와 같은 자산이라 통제변수로 부적절
#   -> 표 5 (T 1,367, 평균 0.002, 표준편차 0.40, 최소 −11.2, 최대 4.00 bp)로 판별
# 사건: FOMC 성명(statement) (출처: 3.3절 "FOMC announcements" = 정책금리 발표 = 성명)
# FOMC 날이 아닌 평일 = 0 (출처: 표 5 T = 1,367)
#
# [사용자 준비] https://sffed.us/USMPD 에서 "Data for USMPD" 엑셀 파일을 받아
#   데이터 폴더에 "통화정책충격_USMPD.xlsx" 로 저장
USMPD경로 = 데이터폴더 / "통화정책충격_USMPD.xlsx"
시트들 = pd.read_excel(USMPD경로, sheet_name=None)
print("시트:", list(시트들))
성명시트 = [k for k in 시트들 if "statement" in k.lower()]
if len(성명시트) != 1:
    raise ValueError(f"성명(statement) 시트를 하나로 정하지 못했습니다: {list(시트들)} -> 시트 이름을 알려주세요")
U = 시트들[성명시트[0]]
print(f"'{성명시트[0]}' 열: {list(U.columns)}")
날짜열 = [c for c in U.columns if "date" in str(c).lower()]
if not 날짜열:
    raise ValueError("날짜 열을 찾지 못했습니다. 위 열 목록을 알려주세요")
U["날짜"] = pd.to_datetime(U[날짜열[0]], errors="coerce").dt.normalize()

MPS후보 = {}
for 열 in ["MP1", "MP2", "FF2", "FF3", "ED1"]:
    맞는열 = [c for c in U.columns if str(c).strip().upper() == 열]
    if not 맞는열:
        print(f"  {열}: 열 없음 -> 건너뜀")
        continue
    x = pd.to_numeric(U[맞는열[0]], errors="coerce") * 100          # 퍼센트포인트 -> bp
    일별 = x.groupby(U["날짜"]).sum(min_count=1)                    # 같은 날 사건이 둘이면 합 (예: 긴급 회의)
    MPS후보[열] = 일별.reindex(평일_넓게).fillna(0.0)
if not MPS후보:
    raise ValueError("후보 열이 하나도 없습니다. 위 열 목록을 알려주세요")
MPS선택 = 고르기(MPS후보, "MPS", "통화정책 충격 (bp) [대체 자료]")
MPS = MPS후보[MPS선택].rename("MPS")


#%% [셀 5] 회귀 자료 만들기 (식 (4)의 변수) ================================
# 출처: 3.3절 식 (4)와 그 아래 설명
#   (1) 종속변수·선행 통제: y^k_{t+h} − y^k_{t−1}, k = FFR, 6M, 1Y ("proximate, but potentially untreated tenors")
#   (2) 지난 변화: y3M_{t−1} − y3M_{t−5}
#   (3) Q_t, 1[Q_t > 0], MPS_t
#   (4) X_t (i) 5일 변화: FFR, 1M·3M·6M·1Y·2Y·10Y 금리, log BTC, log ETH, MMF 자산(뺌), ETF 유입(비교 사양만)
#           (ii) 수준과 5일 변화: VIX, log S&P 500, log 금, log 브렌트, log 달러 NEER, CESI(뺌), TGA, ON-RRP, MCI(뺌)
# 5일 변화 = 5 거래일 전 대비 (출처: 각주 13)
# [논문에 명시 없음] 5일 변화의 종류 -> 표 6 주석의 규칙을 따름
#   "log-differences for price-type series (S&P 500, gold, Brent, NEER, ..., Bitcoin, Ether, VIX)
#    and arithmetic differences for index- and level-type series"
#   -> VIX 는 수준 + 5일 로그 변화, 금리·TGA·ON-RRP 는 수준 차이
# ETF 유입은 이미 "5일 유입"(흐름)이므로 그대로 씀 (출처: 표 5 "5-day ETF flow")
def 오일(s):
    return s - s.shift(5)

X = pd.DataFrame(index=평일_넓게)
# [주의] 3개월 금리의 5일 변화 (y3M_t − y3M_{t−5})
#   논문 X_t 목록에는 3M 이 들어 있음 ("5-day changes in the FFR and Treasury yields across the curve
#   (1M, 3M, 6M, 1Y, 2Y, 10Y)"). 그런데 식 (4)의 지난 변화 [y3M_{t−1} − y3M_{t−5}] 와 함께 넣으면
#   (y3M_t − y3M_{t−5}) − (y3M_{t−1} − y3M_{t−5}) = y3M_t − y3M_{t−1} = h = 0 의 종속변수 그 자체
#   -> h = 0 효과가 계산상 정확히 0 이 됨. 논문은 h = 0 효과 0.71bp(유의)를 보고하므로
#      논문 실제 사양에는 이 변수가 (이 형태로는) 없었다고 보는 것이 자연스러움
#   [논문에 명시 없음] 처리 -> 아래 설정으로 고름 (사용자 확정 10/10: False - 넣지 않음)
#     False: 넣지 않음 (지난 변화가 3개월 금리의 과거 움직임을 이미 통제) / True: 논문 목록 그대로
세달_5일변화_넣기 = False
for 키 in ["FFR", "1개월", "3개월", "6개월", "1년", "2년", "10년"]:
    if 키 == "3개월" and not 세달_5일변화_넣기:
        continue
    X[f"Δ5_{키}"] = 오일(금리[키])
for 키 in ["BTC", "ETH"]:
    X[f"Δ5_log{키}"] = 오일(np.log(수준[키]))
X["VIX"] = 수준["VIX"]
X["Δ5_logVIX"] = 오일(np.log(수준["VIX"]))
for 키 in ["SP500", "금", "브렌트", "NEER"]:
    X[f"log{키}"] = np.log(수준[키])
    X[f"Δ5_log{키}"] = 오일(np.log(수준[키]))
for 키 in ["TGA", "ONRRP"]:
    X[키] = 금리[키]
    X[f"Δ5_{키}"] = 오일(금리[키])
X["Q"] = Q
X["Q있음"] = (Q > 0).astype(float)
X["MPS"] = MPS
X["지난변화_3개월"] = 금리["3개월"].shift(1) - 금리["3개월"].shift(5)

통제열 = list(X.columns)
y3 = 금리["3개월"]
선행키 = ["FFR", "6개월", "1년"]

# 확인 출력
print(f"통제변수 {len(통제열)}개 (+ 선행 통제 {len(선행키)}개 + 상수)")
print("뺀 변수 (구할 수 없음): MMF 자산 5일 변화, CESI 수준·변화, MCI 수준·변화 (사용자 확정 10/10)")
빈 = X.loc[평일].isna().sum()
print("표본 안에서 빈 값이 있는 통제변수:", dict(빈[빈 > 0]))
저장 = pd.concat([y3.rename("y3M"), Flow5, GIV5, X] + ([ETF5] if ETF5 is not None else []), axis=1)
저장.index.name = "date"
저장.to_csv(데이터폴더 / "4단계_회귀자료_평일.csv", encoding=CSV인코딩)
print("저장: 4단계_회귀자료_평일.csv")


#%% [셀 6] 함수: 국소 투영 OLS / IV + Newey-West ===========================
def NW공분산(점수, 시차):
    """
    Newey-West (1987) S = Γ0 + Σ_{l=1..L} (1 − l/(L+1)) (Γl + Γl')
      출처: Newey & West (1987) 바틀렛 가중치 / statsmodels HAC 와 같은 식 (아래 확인에서 일치 확인)
      빠진 날(휴장일)은 남은 관측치를 이어 붙여 계산 (출처: Datta & Du 2012, Fed IFDP 1060 "Equal Spacing")
    """
    S = 점수.T @ 점수
    for l in range(1, 시차 + 1):
        G = 점수[l:].T @ 점수[:-l]
        S += (1 - l / (시차 + 1)) * (G + G.T)
    return S


def p값(t):
    """양측 p 값 (정규분포). statsmodels HAC 기본(use_t=False)과 같음"""
    return erfc(abs(t) / sqrt(2))


def LP자료(h, 통제추가=None, 표본=None):
    """h 에 대한 종속변수 + 통제 (식 (4))"""
    d = pd.DataFrame({"y": y3.shift(-h) - y3.shift(1)}, index=평일_넓게)
    for 키 in 선행키:
        d[f"선행_{키}"] = 금리[키].shift(-h) - 금리[키].shift(1)
    d = pd.concat([d, X] + ([통제추가] if 통제추가 is not None else []), axis=1)
    return d.loc[표본 if 표본 is not None else 평일]


def 회귀(d, 유입, 도구=None, 겹침=4):
    """
    도구 없음: OLS / 도구 있음: 2SLS (내생변수 1개 = 유입, 도구 1개 = GIV, 정확 식별)
      β = (Z'X)^{-1} Z'y, 분산 = (Z'X)^{-1} S (X'Z)^{-1}, S = NW(Z × 잔차)
           출처: 3.3절 / 그림 7 오른쪽. 분산식은 정확 식별 IV 의 HAC 샌드위치 공식
           (이 셀 끝의 확인 코드에서 도구 = 유입일 때 statsmodels OLS 와 일치 확인)
    1단계 F = (GIV 계수 / HAC 표준오차)² (출처: 퀀트랩_BIS_2 [셀 3], 표 2 "HAC-robust F-statistic")
      시차 1 (표 2 주석) / 시차 4 (5일 유입은 이웃 관측치가 4일을 공유 - 같은 겹침 원리)
    """
    열 = [d, 유입.rename("유입")] + ([도구.rename("도구")] if 도구 is not None else [])
    z = pd.concat(열, axis=1, sort=True).dropna().sort_index()      # 날짜 순서 (NW 계산에 필요)
    Xm = sm.add_constant(z.drop(columns=["y"] + (["도구"] if 도구 is not None else [])))
    결과 = {"관측치": len(z)}
    시차들 = {"시차1": 1, "겹침": 겹침}
    if 도구 is None:
        for 이름, L in 시차들.items():
            r = sm.OLS(z["y"], Xm).fit(cov_type="HAC", cov_kwds={"maxlags": L})
            결과[f"표준오차_{이름}"] = r.bse["유입"]
        결과["계수"] = r.params["유입"]
        return 결과
    Zm = Xm.copy()
    Zm["유입"] = z["도구"]
    Xv, Zv, yv = Xm.values, Zm.values, z["y"].values
    ZX역 = np.linalg.inv(Zv.T @ Xv)
    b = ZX역 @ (Zv.T @ yv)
    u = yv - Xv @ b
    j = list(Xm.columns).index("유입")
    결과["계수"] = b[j]
    for 이름, L in 시차들.items():
        V = ZX역 @ NW공분산(Zv * u[:, None], L) @ ZX역.T
        결과[f"표준오차_{이름}"] = sqrt(V[j, j])
    for 이름, L in {"시차1": 1, "시차4": 4}.items():
        r1 = sm.OLS(z["유입"], Zm).fit(cov_type="HAC", cov_kwds={"maxlags": L})
        결과[f"1단계F_{이름}"] = r1.tvalues["유입"] ** 2
    return 결과


# 확인: 도구 = 유입 자신이면 2SLS = OLS -> 직접 만든 표준오차가 statsmodels 와 같은지
d확인 = LP자료(10)
가 = 회귀(d확인, Flow5, None, 겹침시차(10))
나 = 회귀(d확인, Flow5, Flow5, 겹침시차(10))
print(f"OLS(statsmodels) 계수 {가['계수']:.6f}, 표준오차 {가['표준오차_겹침']:.6f}")
print(f"직접 계산         계수 {나['계수']:.6f}, 표준오차 {나['표준오차_겹침']:.6f}")
assert np.isclose(가["계수"], 나["계수"]) and np.isclose(가["표준오차_겹침"], 나["표준오차_겹침"]), "불일치"
print("-> 일치")


#%% [셀 7] 그림 7 재현: 통제만 / 통제 + IV (주 분석 B, 비교 A) ================
# (B) 주 분석: ETF 빼고 2026-03-31 까지 / (A) 비교: ETF 넣고 ETF 자료 끝(2025-12-31)까지
#   (출처: 사용자 확정 10/10)
사양들 = {"B 주 분석 (ETF 없음)": (None, 평일)}
if ETF5 is not None:
    사양들["A 비교 (ETF 포함)"] = (ETF5, 평일[평일 <= ETF끝])

결과행 = []
for 사양, (추가, 표본) in 사양들.items():
    for h in 예측기간:
        d = LP자료(h, 추가, 표본)
        for 방법, 도구 in [("통제만", None), ("IV", GIV5)]:
            r = 회귀(d, Flow5.loc[표본], None if 도구 is None else 도구.loc[표본], 겹침시차(h))
            결과행.append({"사양": 사양, "방법": 방법, "h": h, **r})
    print(f"  {사양} 완료")
결과 = pd.DataFrame(결과행)
for 이름 in ["시차1", "겹침"]:
    결과[f"효과bp_{이름}_se"] = 결과[f"표준오차_{이름}"] * 충격 * 100
    결과[f"p_{이름}"] = [p값(b / s) for b, s in zip(결과["계수"], 결과[f"표준오차_{이름}"])]
결과["효과bp"] = 결과["계수"] * 충격 * 100
결과.to_csv(데이터폴더 / "4단계_그림7_결과.csv", index=False, encoding=CSV인코딩)
print("저장: 4단계_그림7_결과.csv")


def 별(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


보일h = [0, 5, 10, 13, 15, 20, 25, 30, 40]
for 사양 in 사양들:
    print(f"\n========== [{사양}] $3.5B 유입 뒤 3개월 금리 변화 (bp, 별 = 겹침 NW h+4) ==========")
    표 = []
    for 방법 in ["통제만", "IV"]:
        x = 결과.query("사양 == @사양 and 방법 == @방법").set_index("h")
        표.append({"방법": 방법, **{f"h={h}": f"{x.loc[h, '효과bp']:+.2f}{별(x.loc[h, 'p_겹침'])}" for h in 보일h}})
    print(pd.DataFrame(표).to_string(index=False))
    x = 결과.query("사양 == @사양 and 방법 == 'IV'")
    print(f"  관측치 {x['관측치'].min()} ~ {x['관측치'].max()}, 1단계 F: 시차1 {x['1단계F_시차1'].min():.1f} ~ "
          f"{x['1단계F_시차1'].max():.1f} / 시차4 {x['1단계F_시차4'].min():.1f} ~ {x['1단계F_시차4'].max():.1f}")
print("\n[논문 3.4절] 통제만: 10일 내 1bp 미만, 20일째 약 −1.5bp (대체로 유의하지 않음)")
print("             IV  : 당일 −0.71bp (유의), 10일 내 약 −4bp, 13일째 약 −5bp, 10~25일 −3 ~ −4bp")

# 그림 (출처: 그림 7 형식 - 두 패널, 68%·95% 구간, 세로축 %, 가로축 0 ~ 40일)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False          # 마이너스 기호 깨짐 방지 (맑은 고딕에 U+2212 없음)
for 사양, 파일 in [("B 주 분석 (ETF 없음)", "그림7_재현_3개월금리.png"), ("A 비교 (ETF 포함)", "그림7_재현_ETF포함비교.png")]:
    if 사양 not in 사양들:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, 방법, 제목 in zip(axes, ["통제만", "IV"], ["$3.5B 유입 효과 (통제만)", "$3.5B 유입 효과 (통제 + IV)"]):
        x = 결과.query("사양 == @사양 and 방법 == @방법").sort_values("h")
        b = x["계수"] * 충격                     # % 단위 (그림 7 세로축 "3M T-bill response (%)")
        se = x["표준오차_겹침"] * 충격
        ax.fill_between(x["h"], b - Z95 * se, b + Z95 * se, color="#9ecae1", alpha=0.6, lw=0)
        ax.fill_between(x["h"], b - Z68 * se, b + Z68 * se, color="#4292c6", alpha=0.6, lw=0)
        ax.plot(x["h"], b, color="#08306b")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(제목)
        ax.set_xlabel("기간 (거래일)")
        ax.set_xlim(0, 40)
    axes[0].set_ylabel("3개월 T-bill 금리 반응 (%)")
    fig.suptitle(f"그림 7 재현 - {사양} (진한/옅은 = 68%/95% NW 구간, 시차 h+4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(그림폴더 / 파일, dpi=150)
    plt.close(fig)
    print("저장: 그림/" + 파일)


#%% [셀 8] 재현 판정 (주 분석 B, 미리 정한 규칙) =============================
# 판정 규칙 (출처: 사용자 확정 10/10, 결과를 보기 전에 정함)
#   ① h = 0 의 IV 효과가 음수이고 5% 유의
#      근거: 3.4절 "a statistically significant contemporaneous effect of 0.71 basis points"
#   ② IV 효과의 저점(h = 0 ~ 40 중 가장 작은 값)이 10 ~ 25일 안에 있고, 저점에서 5% 유의
#      근거: 3.4절 "a trough of about 5 bps at 13 days, ... around 3 to 4 bps over horizons of 10 to 25 days"
#            "statistically significant across the trough region"
#   ③ 1단계 F ≥ 10 (모든 h 에서, 겹침 시차 4 기준)
#      근거: Staiger & Stock (1997) 경험칙 (논문 밖 문헌. 논문 표 2 는 F 값만 보고)
#   유의성은 겹침 NW 시차(h + 4) 기준 (사용자 확정 10/10). 시차 1 결과는 함께 출력만 함
#   크기(−5bp)는 판정에 넣지 않음: 통제변수 4개(MMF, CESI, MCI, OIS 원자료)가 다르기 때문 (사용자 확정 10/10)
x = 결과.query("사양 == 'B 주 분석 (ETF 없음)' and 방법 == 'IV'").set_index("h")
저점h = int(x["효과bp"].idxmin())
규칙 = [
    ("① h=0 음수·5% 유의", (x.loc[0, "효과bp"] < 0) and (x.loc[0, "p_겹침"] < 유의수준),
     f"h=0: {x.loc[0, '효과bp']:+.2f}bp, p(겹침) {x.loc[0, 'p_겹침']:.3f} / p(시차1) {x.loc[0, 'p_시차1']:.3f}"),
    ("② 저점 10~25일 + 5% 유의", (10 <= 저점h <= 25) and (x.loc[저점h, "p_겹침"] < 유의수준),
     f"저점 h={저점h}: {x.loc[저점h, '효과bp']:+.2f}bp, p(겹침) {x.loc[저점h, 'p_겹침']:.3f} / p(시차1) {x.loc[저점h, 'p_시차1']:.3f}"),
    ("③ 1단계 F ≥ 10", x["1단계F_시차4"].min() >= 10,
     f"최소 F(시차4) {x['1단계F_시차4'].min():.1f}, 최소 F(시차1) {x['1단계F_시차1'].min():.1f}"),
]
판정표 = pd.DataFrame(규칙, columns=["규칙", "충족", "값"])
print("[재현 판정] 주 분석 B, IV")
print(판정표.to_string(index=False))
최종 = "재현됨 (세 규칙 모두 충족)" if 판정표["충족"].all() else \
    "재현 안 됨: " + ", ".join(판정표.loc[~판정표["충족"], "규칙"])
print(f"\n=> {최종}")
print(f"   크기 비교 (판정 아님): 우리 h=0 {x.loc[0, '효과bp']:+.2f} / h=13 {x.loc[13, '효과bp']:+.2f} bp"
      f"  vs 논문 h=0 −0.71 / h=13 약 −5")
판정표.assign(최종=최종).to_csv(데이터폴더 / "4단계_판정.csv", index=False, encoding=CSV인코딩)
print("저장: 4단계_판정.csv")


#%% [셀 9] 점검: 3개월 금리의 사전 추세 (기록용, 판정에 안 씀) ==================
# 질문: GIV 로 도구한 유입이 "유입 이전"의 3개월 금리 변화까지 설명하는가?
#   설명한다면 GIV 가 유입을 거치지 않고 금리 쪽 사정과 얽혀 있을 수 있음 -> IV 결과 해석 주의
# 출처: Stock & Watson (2018, Economic Journal) - LP-IV 도구는 과거·미래 충격과도 무관해야 함
#   [논문에 명시 없음] 논문에는 이 점검이 없음 (사용자 확정 10/10 - 기록용으로 추가)
# 결과변수: j 번째 앞 5 거래일 변화 = y3M_{t−5j} − y3M_{t−5j−5}
#   j = 0: y_t − y_{t−5} = 유입이 일어나는 5일과 같은 기간 (유입 효과가 섞임 -> 표에는 보이되 판정 대상 아님)
#   j = 1 ~ 4: 유입 기간(t−4 ~ t) 이전의 4주
#   [논문에 명시 없음] 4주(20 거래일) - 문헌 근거 없이 정한 길이. 기록용 점검이라 판정에 영향 없음
# 통제 두 가지: "통제 없음"(상수만) / "통제 있음"(X_t, Q, MPS)
#   뺀 것: 선행 통제, 지난 변화, 3개월 금리 5일 변화 - 결과변수와 기간이 겹치거나(j = 0 에서는 결과변수와 같음)
# NW 시차 4: 결과변수 5일 + 유입 5일이 겹치지 않으므로 이웃 관측치는 4일까지 공유
사전행 = []
for j in range(0, 5):
    yj = (y3.shift(5 * j) - y3.shift(5 * j + 5)).rename("y")
    for 통제이름, 통제표 in [("통제 없음", pd.DataFrame(index=평일_넓게)),
                            ("통제 있음", X.drop(columns=["지난변화_3개월", "Δ5_3개월"], errors="ignore"))]:
        d = pd.concat([yj, 통제표], axis=1).loc[평일]
        r = 회귀(d, Flow5.loc[평일], GIV5.loc[평일], 4)
        사전행.append({"j": j, "통제": 통제이름, "효과bp": r["계수"] * 충격 * 100,
                     "p_시차4": p값(r["계수"] / r["표준오차_겹침"]), "관측치": r["관측치"]})
사전 = pd.DataFrame(사전행)
사전.to_csv(데이터폴더 / "4단계_사전추세.csv", index=False, encoding=CSV인코딩)
print("[사전 추세] IV: $3.5B 유입과 그 전 5일 변화 (bp, 별 = NW 시차 4)")
for 통제이름 in ["통제 없음", "통제 있음"]:
    x = 사전.query("통제 == @통제이름").set_index("j")
    print(f"  <{통제이름}>  " + "  ".join(
        f"{'유입 중' if j == 0 else f'{j}주 전'} {x.loc[j, '효과bp']:+.2f}{별(x.loc[j, 'p_시차4'])}" for j in range(5)))
걸림 = 사전.query("j >= 1 and p_시차4 < @유의수준")
print(f"  j = 1 ~ 4 중 5% 유의: {len(걸림)}칸 / 8칸 (기록용)")
print("저장: 4단계_사전추세.csv")
