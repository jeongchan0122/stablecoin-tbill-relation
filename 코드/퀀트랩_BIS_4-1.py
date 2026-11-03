# -*- coding: utf-8 -*-
"""
퀀트랩_BIS_4-1.py
=====================================================================
[논문] Ahmed & Aldasoro (2026), "Stablecoins and safe asset prices",
       BIS Working Paper No 1270 (2026년 6월 개정판)
[단계] 복제 4-1단계: 기간 확장 - 2026-03-31 이후 최신 자료까지 늘려도 그림 7 결과가 유지되는가

[출처 표기 규칙]  (퀀트랩_BIS_1~4 와 같음)
  # 출처: 논문 ○○절 / 각주 ○ / 그림 ○ 주석 / 부록 ○   -> 논문 근거
  # 출처: 사용자 지시 / 사용자 확정 (날짜)               -> 연구자가 정한 조건
  # 출처: 퀀트랩_BIS_○.py                              -> 앞 단계에서 확정한 사항
  # [논문에 명시 없음]                                 -> 논문에 방법이 없어 정해야 했던 부분
  # [대체 자료]                                        -> 논문 자료를 구할 수 없어 바꾼 자료

[확정된 설계] (출처: 사용자 확정 10/10)
  - 자료를 모두 최신까지 다시 받음. 파일 이름은 "확장_" 으로 시작 -> 기존 CSV(BIS_1~4)는 건드리지 않음
  - GIV 는 퀀트랩_BIS_2 와 같은 규칙으로 늘린 패널에서 새로 계산
  - 기준 재확인: 새로 받은 자료를 2026-03-31 에서 잘라 BIS_4 결과와 비교 (자료 사후 수정의 영향 분리)
  - 사양: BIS_4 주 분석 B 와 같음 (ETF 없음 - CRSP 가 2025-12-31 에서 끝남, 3개월 금리 5일 변화 통제 없음)
  - 판정 (결과 보기 전에 확정)
      (가) 유지: 늘린 표본 전체에서 BIS_4 규칙 ①②③ 모두 충족
      (나) 변화: 식 (5) 상태 = 1[2026-04-01 이후], h = 0 과 h = 13 의 차이 계수가 5% 유의하면 "효과가 달라짐"
      (다) 검출력: 최소 검출 차이 = 2.8 × 차이 계수 표준오차 (Bloom 1995)
  - 사전 추세: BIS_4 [셀 9] 와 같은 점검 (기록용)

[입력] 인터넷: DefiLlama, FRED, Yahoo Finance, 미 재무부 Fiscal Data
       파일 : 데이터/확장_통화정책충격_USMPD.xlsx  (SF 연준 사이트에서 최신본을 받아 이 이름으로 저장)
              데이터/4단계_그림7_결과.csv          (퀀트랩_BIS_4 [셀 7] - 기준 재확인 비교용)
[출력] (모두 "확장_" 으로 시작)
  데이터/확장_합산시총_일별.csv, 확장_코인체인_패널_일별.csv, 확장_GIV_평일.csv, 확장_시장자료_원자료_일별.csv
  데이터/확장_기준재확인.csv, 확장_그림7_결과.csv, 확장_판정.csv, 확장_변화검정.csv, 확장_사전추세.csv
  그림/확장_그림7_전체기간.png, 확장_그림7_4월전후.png

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install requests pandas numpy statsmodels matplotlib openpyxl yfinance
=====================================================================
"""

#%% [셀 0] 패키지 + 설정 =================================================
import os
import warnings
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt

# FRED API 키 (출처: 사용자 지시) - 코드에 적지 않고 프로젝트 폴더의 FRED_키.txt 에서 읽음 (퀀트랩_BIS_1.py 와 같음)
_키파일 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "FRED_키.txt"   # GitHub에 올리지 않는 파일
FRED_API_KEY = os.environ.get("FRED_API_KEY") or (
    _키파일.read_text(encoding="utf-8").strip() if _키파일.exists() else "여기에_FRED_키_붙여넣기")
FRED주소 = "https://api.stlouisfed.org/fred/series/observations"
LLAMA = "https://stablecoins.llama.fi"              # 출처: 퀀트랩_BIS_1.py [셀 0]
FD주소 = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"

결과폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩"
데이터폴더 = 결과폴더 / "데이터"
그림폴더 = 결과폴더 / "그림"
그림폴더.mkdir(parents=True, exist_ok=True)
CSV인코딩 = "utf-8-sig"

# 기간
표본시작 = pd.Timestamp("2021-01-04")    # 출처: 논문 표 1 주석
기존끝 = pd.Timestamp("2026-03-31")      # 출처: 논문 표 1 주석 (BIS_1~4 의 표본끝)
상태시작 = pd.Timestamp("2026-04-01")    # 변화 검정의 상태 S = 1 시작일 (출처: 사용자 확정 10/10 - 논문 표본 이후)
패널시작 = pd.Timestamp("2021-01-01")    # 출처: 퀀트랩_BIS_1.py [셀 0] 시작일
받기시작 = pd.Timestamp("2020-12-01")    # 5일 변화용 (출처: 퀀트랩_BIS_3.py [셀 5])

# 코인 (출처: 논문 2.1절 + 각주 4, 퀀트랩_BIS_1.py [셀 0]·[셀 1])
코인목록 = ["USDT", "USDC", "TUSD", "BUSD", "FDUSD", "PYUSD", "RLUSD"]
ID_직접지정 = {"BUSD": "4"}              # 출처: 퀀트랩_BIS_1.py [셀 1] (BUSD 만 후보 3개)

# GIV 규칙 (출처: 퀀트랩_BIS_2.py [셀 0]·[셀 6] - 저자 답신(2026-10-12)으로 확정한 설정)
활성기준 = 1e6                            # 각주 18 "$1 million on both t − 1 and t"
최소N = 5                                 # 각주 18 "N_t ≥ 5"
윈저배수 = 8 * 1.4826                     # 각주 18 "±8 × 1.4826 × MAD"
#   설정: 날짜방식 가-3, MAD 범위 전체, 중심 0, MAD 기준 중앙값, MAD 표본 활성, N = t−1 활성 (출처: 저자 답신) 

# 시리즈 선택 (출처: 퀀트랩_BIS_4.py [셀 2]~[셀 4] 표 5 판별 결과, 10/10)
#   3개월 DGS3MO (값 차이 합 0.006), FFR EFFR (T 1,314 일치), TGA WTREGEN, ON-RRP RRPONTSYD,
#   입찰 = 낙찰 금액(total_accepted) / 발행일(issue_date), 통화정책 충격 = USMPD FF2 [대체 자료]
FRED_시리즈 = {"1개월": "DGS1MO", "3개월": "DGS3MO", "6개월": "DGS6MO", "1년": "DGS1", "2년": "DGS2",
             "10년": "DGS10", "FFR": "EFFR", "ONRRP": "RRPONTSYD", "TGA": "WTREGEN",
             "VIX": "VIXCLS", "SP500": "SP500", "NEER": "DTWEXBGS"}          # 뒤 3개: 퀀트랩_BIS_3.py [셀 5]
야후_시리즈 = {"BTC": "BTC-USD", "ETH": "ETH-USD",                            # 퀀트랩_BIS_3.py [셀 5]
             "금": "GC=F", "브렌트": "BZ=F"}                                    # 퀀트랩_BIS_3-1.py [셀 2] (선물)
입찰금액열, 입찰날짜열 = "total_accepted", "issue_date"
MPS열 = "FF2"

# 회귀 설정 (출처: 퀀트랩_BIS_4.py [셀 0] 과 같음)
예측기간 = list(range(0, 41))            # 그림 7 가로축
충격 = 3.5                               # 3.4절 "$3.5B"
유의수준 = 0.05
Z68, Z95 = 0.994458, 1.959964
def 겹침시차(h):                         # 사용자 확정 10/10 (BIS_4 와 같음)
    return h + 4
판정h = [0, 13]                           # 변화 검정 시점 (사용자 확정 10/10: 논문 강조 시점·BIS_4 저점)
MDE배수 = 2.8                             # 출처: Bloom (1995) - 5% 양측, 검출력 80%

if "붙여넣기" in FRED_API_KEY:
    raise ValueError("[셀 0]: 프로젝트 폴더에 FRED_키.txt 를 만들고 FRED API 키를 한 줄로 넣으세요 (또는 환경변수 FRED_API_KEY).")
오늘 = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
print("데이터 폴더:", 데이터폴더, "/ 오늘(UTC):", 오늘.date())


#%% [셀 1] DefiLlama: 합산 시총 + 코인 × 체인 패널 (최신까지) ==================
# 방법은 퀀트랩_BIS_1.py [셀 1]~[셀 3] 과 같음, 종료일만 없음 (출처: 사용자 지시 "현재까지")
# [논문에 명시 없음] 오늘(UTC) 날짜의 값은 하루가 끝나지 않은 값일 수 있어 뺌
목록 = requests.get(f"{LLAMA}/stablecoins", timeout=120).json()["peggedAssets"]
코인ID = {}
for 심볼 in 코인목록:
    if 심볼 in ID_직접지정:
        코인ID[심볼] = ID_직접지정[심볼]
        continue
    후보 = [a for a in 목록 if str(a.get("symbol", "")).upper() == 심볼]
    if len(후보) != 1:
        raise ValueError(f"{심볼}: 후보 {len(후보)}개 - 퀀트랩_BIS_1.py [셀 1] 처럼 ID_직접지정 필요")
    코인ID[심볼] = str(후보[0]["id"])
print("ID:", 코인ID)

# (가) 합산 시총 (출처: BIS_1 [셀 2] - totalCirculatingUSD 의 peggedUSD, 7개 합)
행들 = []
for 심볼, 번호 in 코인ID.items():
    for r in requests.get(f"{LLAMA}/stablecoincharts/all", params={"stablecoin": 번호}, timeout=120).json():
        행들.append({"date": pd.to_datetime(int(r["date"]), unit="s"), "symbol": 심볼,
                   "mcap": (r.get("totalCirculatingUSD") or {}).get("peggedUSD")})
합산 = pd.DataFrame(행들)
합산 = 합산[(합산["date"] >= 패널시작) & (합산["date"] < 오늘)]
합산 = 합산.pivot_table(index="date", columns="symbol", values="mcap", aggfunc="sum")
합산["Total"] = 합산[[c for c in 코인목록 if c in 합산.columns]].sum(axis=1)
합산.index.name = "date"
합산.to_csv(데이터폴더 / "확장_합산시총_일별.csv", encoding=CSV인코딩)

# (나) 코인 × 체인 패널 (출처: BIS_1 [셀 3] - chainBalances 의 circulating peggedUSD, 가격 $1 가정)
행들 = []
for 심볼, 번호 in 코인ID.items():
    자료 = requests.get(f"{LLAMA}/stablecoin/{번호}", timeout=120).json()
    for 체인, 내용 in (자료.get("chainBalances", {}) or {}).items():
        for t in (내용 or {}).get("tokens", []) or []:
            행들.append({"date": pd.to_datetime(int(t["date"]), unit="s"), "symbol": 심볼, "chain": 체인,
                       "mcap": (t.get("circulating") or {}).get("peggedUSD")})
패널 = pd.DataFrame(행들)
패널 = 패널[(패널["date"] >= 패널시작) & (패널["date"] < 오늘)]
패널["pair"] = 패널["symbol"] + " / " + 패널["chain"]
패널 = 패널.sort_values(["date", "symbol", "chain"]).reset_index(drop=True)[["date", "symbol", "chain", "pair", "mcap"]]
패널.to_csv(데이터폴더 / "확장_코인체인_패널_일별.csv", index=False, encoding=CSV인코딩)
print(f"합산 시총: ~ {합산.index.max().date()}, 패널: {len(패널):,}행 ~ {패널['date'].max().date()}")
print("저장: 확장_합산시총_일별.csv, 확장_코인체인_패널_일별.csv")


#%% [셀 2] FRED·Yahoo·Fiscal Data·USMPD (최신까지) ==========================
# 시리즈는 [셀 0] 의 선택 그대로 (BIS_3·3-1·4 와 같은 자료)
import yfinance as yf

시장 = {}
for 이름, 코드 in FRED_시리즈.items():
    자료 = requests.get(FRED주소, timeout=120, params={
        "series_id": 코드, "api_key": FRED_API_KEY, "file_type": "json",
        "observation_start": 받기시작.strftime("%Y-%m-%d")}).json()
    시장[이름] = pd.Series({pd.to_datetime(o["date"]): (np.nan if o["value"] == "." else float(o["value"]))
                         for o in 자료["observations"]})
시장["TGA"] = 시장["TGA"] / 1000        # 백만 달러 -> $B (BIS_4 [셀 2]: WTREGEN 단위 Millions)
for 이름, 코드 in 야후_시리즈.items():
    h = yf.Ticker(코드).history(start=받기시작.strftime("%Y-%m-%d"), auto_adjust=False)
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    시장[이름] = s[~s.index.duplicated(keep="last")]
시장 = pd.DataFrame(시장)
시장 = 시장[시장.index < 오늘]
시장.index.name = "date"
시장.to_csv(데이터폴더 / "확장_시장자료_원자료_일별.csv", encoding=CSV인코딩)
print("저장: 확장_시장자료_원자료_일별.csv")

# 3개월 T-bill 입찰 (출처: BIS_4 [셀 3], 선택 = 낙찰 금액 / 발행일)
행들, 쪽 = [], 1
while True:
    r = requests.get(FD주소, timeout=120, params={
        "filter": f"security_type:eq:Bill,security_term:eq:13-Week,record_date:gte:{받기시작.date()}",
        "page[size]": 1000, "page[number]": 쪽}).json()
    행들 += r["data"]
    if 쪽 >= r["meta"]["total-pages"]:
        break
    쪽 += 1
입찰 = pd.DataFrame(행들)
if "cash_management_bill_cmb" in 입찰.columns:
    입찰 = 입찰[입찰["cash_management_bill_cmb"] != "Yes"]
입찰일별 = (pd.DataFrame({"날짜": pd.to_datetime(입찰[입찰날짜열], errors="coerce"),
                       "금액": pd.to_numeric(입찰[입찰금액열], errors="coerce") / 1e9})
          .dropna().groupby("날짜")["금액"].sum())

# 통화정책 충격 [대체 자료] (출처: BIS_4 [셀 4], 선택 = 성명 FF2, 퍼센트포인트 × 100 = bp)
U = pd.read_excel(데이터폴더 / "확장_통화정책충격_USMPD.xlsx", sheet_name="Statements")
U["날짜"] = pd.to_datetime(U["Date"], errors="coerce").dt.normalize()
MPS일별 = (pd.to_numeric(U[MPS열], errors="coerce") * 100).groupby(U["날짜"]).sum(min_count=1)

print("마지막 날짜:")
for 이름 in 시장.columns:
    print(f"  {이름}: {시장[이름].dropna().index.max().date()}")
print(f"  입찰(발행일): {입찰일별.index.max().date()} / USMPD 마지막 FOMC: {MPS일별.dropna().index.max().date()}")


#%% [셀 3] 확장 표본 끝 정하기 + 평일 달력 =================================
# [논문에 명시 없음] 확장 표본의 끝 = 매일 값이 필요한 자료가 모두 있는 마지막 날
#   (금리·FFR·ON-RRP·VIX·S&P·NEER·BTC·ETH·금·브렌트·합산 시총·패널 중 가장 이른 마지막 날)
#   TGA 는 주간 자료라 앞값 채움 (BIS_4 [셀 2]) -> 마지막 관측 뒤 6일까지 허용
#   입찰·통화정책 충격은 "없는 날 = 0" 인 자료라 끝을 정하는 데 넣지 않음.
#     단, USMPD 마지막 FOMC 이후의 회의는 0 으로 들어가므로 아래에서 확인 출력
매일자료 = [c for c in 시장.columns if c != "TGA"]
끝후보 = {c: 시장[c].dropna().index.max() for c in 매일자료}
끝후보["합산 시총"] = 합산.index.max()
끝후보["패널"] = 패널["date"].max()
끝후보["TGA(+6일)"] = 시장["TGA"].dropna().index.max() + pd.Timedelta(days=6)
확장끝 = min(끝후보.values())
가장이른 = min(끝후보, key=끝후보.get)
print(f"확장 표본 끝: {확장끝.date()}  (가장 먼저 끝나는 자료: {가장이른})")
print(f"  USMPD 마지막 FOMC {MPS일별.dropna().index.max().date()} -> 그 뒤 {확장끝.date()} 까지 FOMC 가 있었다면 "
      "최신 USMPD 파일인지 확인 필요")

평일_넓게 = pd.bdate_range(받기시작, 확장끝)        # 출처: BIS_2 [셀 0] 평일 달력
평일_확장 = pd.bdate_range(표본시작, 확장끝)
평일_기존 = pd.bdate_range(표본시작, 기존끝)
print(f"평일: 기존 {len(평일_기존)}일 (표 5: 1,367) + 4월 이후 {len(평일_확장) - len(평일_기존)}일 = {len(평일_확장)}일")


#%% [셀 4] GIV 다시 계산 (퀀트랩_BIS_2 와 같은 규칙, 끝 날짜 두 가지) ============
# 아래 함수는 퀀트랩_BIS_2.py [셀 2]·[셀 3] 과 같은 계산 (선택 조합 하나만 남김)
#   성장률 = (MC_t − MC_{t−1}) / MC_{t−1} × 100   (BIS_2: [논문에 명시 없음] 문자 그대로의 성장률)
#   활성 = t−1, t 모두 $1M 초과 (각주 18)
#   패널 페어 = 표본 기간 중 한 번이라도 활성 (2.2절 226개 페어와 일치했던 정의)
#   윈저화 = 0 ± 8 × 1.4826 × MAD, 범위 "전체", MAD 표본 "활성" (각주 18, 저자 답신)
#   GIV_t = Σ (s_{i,t−1} − 1/N_t) ε_{i,t} (식 (2)), N_t = t−1 활성 페어 수 (저자 답신)
#     N_t < 5 면 빈 값 (각주 18) [저자 답신에 명시 없음] t−1 활성 N 에 적용 (퀀트랩_BIS_2.py [셀 3]과 같음)
#   5일 합 = 달력일 5개 합 -> 평일만 (가-3, 저자 답신 - 최근 달력일 5개(t−4 ~ t))
# [주의] 패널 페어와 윈저화 기준(MAD)을 "표본 전체"로 정하므로, 끝 날짜가 바뀌면 3월 이전 GIV 도 조금 바뀜
#        -> 끝 = 기존끝 (기준 재확인용) / 끝 = 확장끝 (확장 분석용) 두 벌을 따로 계산
def 성장률과_활성(시총표):
    전날 = 시총표.shift(1)
    g = (시총표 - 전날) / 전날.where(전날 > 0) * 100
    활성 = (시총표 > 활성기준) & (전날 > 활성기준)
    return g, 활성, 전날


def GIV만들기(끝):
    p = 패널[패널["date"] <= 끝]
    표 = p.pivot_table(index="date", columns="pair", values="mcap", aggfunc="sum")
    표 = 표.reindex(pd.date_range(표.index.min(), 끝, freq="D"))
    g, 활성, 전날 = 성장률과_활성(표)
    표 = 표[활성.columns[활성.any()]]                       # 패널 페어
    g, 활성, 전날 = 성장률과_활성(표)
    # 윈저화 (범위 전체, 중심 0, MAD 기준 중앙값, MAD 표본 활성)
    x = g.where(활성).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
    gw = pd.DataFrame(np.clip(g.values, - 윈저배수 * mad, 윈저배수 * mad), index=g.index, columns=g.columns) # 출처: 저자 답신 - 경계는 0 중심
    # GIV_t (식 (2))
    s = 전날.where(활성)
    s = s.div(s.sum(axis=1), axis=0)
    gg = gw.where(활성)
    N = (전날 > 활성기준).sum(axis=1) # 출처: 저자 답신 - N은 t−1 활성만으로 셈
    eps = gg.sub((s * gg).sum(axis=1), axis=0)
    giv = (s.sub(1 / N, axis=0) * eps).sum(axis=1)
    giv[N < 최소N] = np.nan   # [저자 답신에 명시 없음] N ≥ 5 를 t−1 활성 N 에 적용 (퀀트랩_BIS_2.py [셀 3]과 같음)
    평일들 = pd.bdate_range(표본시작, 끝)
    return giv.rolling(5).sum().reindex(평일들), len(표.columns)


GIV5_기존, 페어_기존 = GIV만들기(기존끝)
GIV5_확장, 페어_확장 = GIV만들기(확장끝)
MC = 합산["Total"].reindex(평일_넓게) / 1e9               # $B (BIS_2 [셀 1])
Flow5 = (MC - MC.shift(5)).rename("Flow5")                # 식 (1), 평일 5일

print(f"패널 페어: 기존끝까지 {페어_기존}개 (BIS_2: 226) / 확장끝까지 {페어_확장}개")
for 이름, giv5, 날 in [("기존끝", GIV5_기존, 평일_기존), ("확장끝", GIV5_확장, 평일_확장)]:
    d = pd.concat([Flow5.reindex(날), giv5.rename("GIV")], axis=1).dropna()
    r = sm.OLS(d["Flow5"], sm.add_constant(d["GIV"])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    print(f"  [1단계, {이름}] 계수 {r.params['GIV']:.3f}, F {r.tvalues['GIV'] ** 2:.1f}, R² {r.rsquared:.3f}, T {len(d)}"
          f"  (BIS_2: 0.974, 34.5, 0.081 / 표 2 (3)열: 1.04, 42.0, 0.097)")
GIV5_기존.index.name = "date"
pd.DataFrame({"GIV5_기존끝": GIV5_기존.reindex(평일_확장), "GIV5_확장끝": GIV5_확장}).rename_axis("date") \
    .to_csv(데이터폴더 / "확장_GIV_평일.csv", encoding=CSV인코딩)
겹침 = pd.concat([GIV5_기존, GIV5_확장], axis=1).dropna()
print(f"  같은 날짜에서 두 GIV 의 상관: {겹침.corr().iloc[0, 1]:.4f} (1 에 가까우면 끝 날짜의 영향 작음)")
print("저장: 확장_GIV_평일.csv")


#%% [셀 5] 회귀 자료 + 함수 (퀀트랩_BIS_4 [셀 5]·[셀 6] 과 같은 식) ===============
def 앞값채움(s):
    return s.reindex(s.index.union(평일_넓게)).sort_index().ffill().reindex(평일_넓게)


# 휴장일 처리 (출처: BIS_4 [셀 1]·[셀 2] - 표 5 의 T)
#   금리·FFR·ON-RRP: 채우지 않음 / VIX·S&P·NEER·금·브렌트·TGA: 앞값 채움 / BTC·ETH: 평일 값 그대로
금리 = pd.DataFrame({k: 시장[k].reindex(평일_넓게) for k in ["1개월", "3개월", "6개월", "1년", "2년", "10년", "FFR", "ONRRP"]})
금리["TGA"] = 앞값채움(시장["TGA"])
수준 = pd.DataFrame({k: 앞값채움(시장[k]) for k in ["VIX", "SP500", "NEER", "금", "브렌트"]})
for k in ["BTC", "ETH"]:
    수준[k] = 시장[k].reindex(평일_넓게)
Q = 입찰일별.reindex(평일_넓게).fillna(0.0)               # 입찰 없는 평일 = 0 (3.3절, 표 5)
MPS = MPS일별.reindex(평일_넓게).fillna(0.0)              # FOMC 아닌 평일 = 0 (표 5)


def 오일(s):
    return s - s.shift(5)


# X_t (출처: 3.3절, BIS_4 [셀 5] 와 같음. 3개월 금리 5일 변화는 넣지 않음 - 사용자 확정 10/10)
X = pd.DataFrame(index=평일_넓게)
for 키 in ["FFR", "1개월", "6개월", "1년", "2년", "10년"]:
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
y3 = 금리["3개월"]
선행키 = ["FFR", "6개월", "1년"]
S = pd.Series((평일_넓게 >= 상태시작).astype(float), index=평일_넓게, name="S")   # 상태 (4월 이후 = 1)


def LP자료(h, 표본, 끝=None):
    """
    식 (4) 의 종속변수 + 통제 (BIS_4 [셀 6] 과 같음)
    끝: 예측 끝 날짜(t+h)가 이 날을 넘는 관측치는 제외
        기준 재확인은 끝 = 2026-03-31 -> BIS_4 처럼 3월 31일 이후 금리를 쓰지 않음
        확장 분석은 끝 = 확장끝 (자료가 거기서 끝나므로 자동)
    """
    d = pd.DataFrame({"y": y3.shift(-h) - y3.shift(1)}, index=평일_넓게)
    if 끝 is not None:
        넘음 = pd.Series(평일_넓게, index=평일_넓게).shift(-h)
        d.loc[넘음.isna() | (넘음 > 끝), "y"] = np.nan
    for 키 in 선행키:
        d[f"선행_{키}"] = 금리[키].shift(-h) - 금리[키].shift(1)
    return pd.concat([d, X], axis=1).loc[표본]


def NW공분산(점수, 시차):
    """Newey-West (1987) 바틀렛 가중치 (BIS_4 [셀 6] 과 같음)"""
    S_ = 점수.T @ 점수
    for l in range(1, 시차 + 1):
        G = 점수[l:].T @ 점수[:-l]
        S_ += (1 - l / (시차 + 1)) * (G + G.T)
    return S_


def p값(t):
    return erfc(abs(t) / sqrt(2))


def 회귀(d, 내생, 도구=None, 겹침=4):
    """
    내생: 유입 열들 (DataFrame) / 도구: 같은 개수의 도구 열들 (정확 식별)
      BIS_4 [셀 6] 의 2SLS 를 내생변수 여러 개로 넓힌 것 (식 (5) 의 유입 × 상태 때문에)
      β = (Z'X)^{-1} Z'y, 분산 = (Z'X)^{-1} S (X'Z)^{-1}, S = NW(Z × 잔차)
    1단계 F: 내생변수마다, 제외된 도구 전체의 HAC Wald / 도구 개수
      (도구가 1개면 BIS_4 의 F = t² 와 같음)
    """
    z = pd.concat([d, 내생] + ([도구] if 도구 is not None else []), axis=1, sort=True).dropna().sort_index()
    외생열 = [c for c in d.columns if c != "y"]
    Xm = sm.add_constant(z[외생열 + list(내생.columns)])
    결과 = {"관측치": len(z)}
    시차들 = {"시차1": 1, "겹침": 겹침}
    if 도구 is None:
        for 이름, L in 시차들.items():
            r = sm.OLS(z["y"], Xm).fit(cov_type="HAC", cov_kwds={"maxlags": L})
            for c in 내생.columns:
                결과[(c, f"표준오차_{이름}")] = r.bse[c]
        for c in 내생.columns:
            결과[(c, "계수")] = r.params[c]
        return 결과
    Zm = Xm.copy()
    for c, zc in zip(내생.columns, 도구.columns):
        Zm[c] = z[zc]
    Xv, Zv, yv = Xm.values, Zm.values, z["y"].values
    ZX역 = np.linalg.inv(Zv.T @ Xv)
    b = ZX역 @ (Zv.T @ yv)
    u = yv - Xv @ b
    for 이름, L in 시차들.items():
        V = ZX역 @ NW공분산(Zv * u[:, None], L) @ ZX역.T
        for c in 내생.columns:
            j = list(Xm.columns).index(c)
            결과[(c, f"표준오차_{이름}")] = sqrt(V[j, j])
    for c in 내생.columns:
        결과[(c, "계수")] = b[list(Xm.columns).index(c)]
        r1 = sm.OLS(z[c], Zm).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        idx = [list(Zm.columns).index(k) for k in 내생.columns]           # 도구가 들어간 자리
        bb, VV = r1.params.values[idx], r1.cov_params().values[np.ix_(idx, idx)]
        결과[(c, "1단계F")] = float(bb @ np.linalg.solve(VV, bb)) / len(idx)
    return 결과


def 별(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def IRF(표본, giv5, 끝):
    """통제만 / IV 의 h = 0 ~ 40 IRF (BIS_4 [셀 7] 과 같은 계산)"""
    행 = []
    for h in 예측기간:
        d = LP자료(h, 표본, 끝)
        내생 = Flow5.loc[표본].to_frame("유입")
        for 방법, 도구 in [("통제만", None), ("IV", giv5.reindex(표본).to_frame("GIV"))]:
            r = 회귀(d, 내생, 도구, 겹침시차(h))
            줄 = {"방법": 방법, "h": h, "관측치": r["관측치"], "계수": r[("유입", "계수")],
                 "표준오차_시차1": r[("유입", "표준오차_시차1")], "표준오차_겹침": r[("유입", "표준오차_겹침")],
                 "1단계F": r.get(("유입", "1단계F"), np.nan)}
            행.append(줄)
    x = pd.DataFrame(행)
    x["효과bp"] = x["계수"] * 충격 * 100
    for 이름 in ["시차1", "겹침"]:
        x[f"p_{이름}"] = [p값(b / s) for b, s in zip(x["계수"], x[f"표준오차_{이름}"])]
    return x


# 확인: 도구 = 유입 자신이면 2SLS = OLS
d확인 = LP자료(10, 평일_확장)
가 = 회귀(d확인, Flow5.loc[평일_확장].to_frame("유입"), None, 14)
나 = 회귀(d확인, Flow5.loc[평일_확장].to_frame("유입"), Flow5.loc[평일_확장].to_frame("도구"), 14)
assert np.isclose(가[("유입", "계수")], 나[("유입", "계수")]) and \
       np.isclose(가[("유입", "표준오차_겹침")], 나[("유입", "표준오차_겹침")]), "불일치"
print("확인: 직접 만든 2SLS 표준오차 = statsmodels OLS (일치)")


#%% [셀 6] 기준 재확인: 새 자료를 2026-03-31 에서 잘라 BIS_4 와 비교 ==============
# 목적: 새로 받은 자료(DefiLlama 사후 수정 등)만으로 결과가 얼마나 달라지는지 먼저 봄
#   -> 여기서 차이가 크면, 확장 결과의 변화를 "기간" 탓으로 돌릴 수 없음
기준 = IRF(평일_기존, GIV5_기존, 기존끝)
기준.to_csv(데이터폴더 / "확장_기준재확인.csv", index=False, encoding=CSV인코딩)
보일h = [0, 5, 10, 13, 15, 20, 25, 30, 40]
옛 = pd.read_csv(데이터폴더 / "4단계_그림7_결과.csv", encoding=CSV인코딩)
옛 = 옛[옛["사양"] == "B 주 분석 (ETF 없음)"]
print("[기준 재확인] 2026-03-31 까지, IV / 통제만 (bp, 별 = 겹침 NW)")
표 = []
for 방법 in ["IV", "통제만"]:
    a = 기준.query("방법 == @방법").set_index("h")
    b = 옛.query("방법 == @방법").set_index("h")
    표.append({"구분": f"{방법} 새 자료", **{f"h={h}": f"{a.loc[h, '효과bp']:+.2f}{별(a.loc[h, 'p_겹침'])}" for h in 보일h}})
    표.append({"구분": f"{방법} BIS_4", **{f"h={h}": f"{b.loc[h, '효과bp']:+.2f}{별(b.loc[h, 'p_겹침'])}" for h in 보일h}})
print(pd.DataFrame(표).to_string(index=False))
F새 = 기준.query("방법 == 'IV'")["1단계F"].min()
F옛 = 옛.query("방법 == 'IV'")["1단계F_시차4"].min()
print(f"  1단계 F 최소 (시차 4): 새 자료 {F새:.1f} / BIS_4 {F옛:.1f}")
print("저장: 확장_기준재확인.csv")


#%% [셀 7] (가) 유지 판정: 확장 표본 전체 ===================================
# 규칙 (출처: 사용자 확정 10/10 - BIS_4 [셀 8] 과 같은 규칙)
#   ① h = 0 IV 효과 음수·5% 유의 ② 저점 10 ~ 25일 + 5% 유의 ③ 1단계 F ≥ 10 (모든 h)
#   유의성은 겹침 NW (h + 4)
확장 = IRF(평일_확장, GIV5_확장, 확장끝)
확장.to_csv(데이터폴더 / "확장_그림7_결과.csv", index=False, encoding=CSV인코딩)
print(f"[확장 표본] {표본시작.date()} ~ {확장끝.date()}, $3.5B 유입 뒤 3개월 금리 변화 (bp, 별 = 겹침 NW)")
표 = []
for 방법 in ["통제만", "IV"]:
    a = 확장.query("방법 == @방법").set_index("h")
    표.append({"방법": 방법, **{f"h={h}": f"{a.loc[h, '효과bp']:+.2f}{별(a.loc[h, 'p_겹침'])}" for h in 보일h}})
print(pd.DataFrame(표).to_string(index=False))

x = 확장.query("방법 == 'IV'").set_index("h")
저점h = int(x["효과bp"].idxmin())
규칙 = [
    ("① h=0 음수·5% 유의", (x.loc[0, "효과bp"] < 0) and (x.loc[0, "p_겹침"] < 유의수준),
     f"h=0: {x.loc[0, '효과bp']:+.2f}bp, p(겹침) {x.loc[0, 'p_겹침']:.3f} / p(시차1) {x.loc[0, 'p_시차1']:.3f}"),
    ("② 저점 10~25일 + 5% 유의", (10 <= 저점h <= 25) and (x.loc[저점h, "p_겹침"] < 유의수준),
     f"저점 h={저점h}: {x.loc[저점h, '효과bp']:+.2f}bp, p(겹침) {x.loc[저점h, 'p_겹침']:.3f}"),
    ("③ 1단계 F ≥ 10", x["1단계F"].min() >= 10, f"최소 F(시차4) {x['1단계F'].min():.1f}"),
]
판정표 = pd.DataFrame(규칙, columns=["규칙", "충족", "값"])
print("\n[(가) 유지 판정]")
print(판정표.to_string(index=False))
최종가 = "유지됨 (세 규칙 모두 충족)" if 판정표["충족"].all() else \
    "유지 안 됨: " + ", ".join(판정표.loc[~판정표["충족"], "규칙"])
print(f"=> {최종가}")


#%% [셀 8] (나)(다) 변화 검정: 식 (5), 상태 = 2026-04-01 이후 =====================
# 출처: 논문 3.5절 식 (5) "extending Equation (4) with a state-by-flow interaction and a state main effect"
#   y = ... + β0 Flow + β1 Flow·S + θ S + (식 (4) 의 나머지 통제)
#   β0 = 기준 상태(3월까지) 효과, β0 + β1 = 4월 이후 효과, β1 = 차이
# [논문에 명시 없음] 상태 의존 IRF 를 IV 로 추정했는지 논문이 밝히지 않음
#   -> 도구 = GIV, GIV × S (내생변수 Flow, Flow × S 와 같은 개수, 정확 식별) (사용자 확정 10/10)
# 판정 (나): h = 0, 13 에서 β1 이 5% 유의(겹침 NW)하면 "4월 이후 효과가 달라짐"
# 검출력 (다): 최소 검출 차이 = 2.8 × β1 표준오차 (Bloom 1995) -> 이보다 작은 변화는 판단 불가
변화행 = []
for h in 예측기간:
    d = pd.concat([LP자료(h, 평일_확장), S.loc[평일_확장]], axis=1)
    내생 = pd.DataFrame({"유입": Flow5, "유입xS": Flow5 * S}).loc[평일_확장]
    도구 = pd.DataFrame({"GIV": GIV5_확장, "GIVxS": GIV5_확장 * S.reindex(평일_확장)}).loc[평일_확장]
    r = 회귀(d, 내생, 도구, 겹침시차(h))
    b0, b1 = r[("유입", "계수")], r[("유입xS", "계수")]
    # β0 + β1 의 표준오차: 공분산이 필요 -> S 를 반대로 둔 회귀(기준 = 4월 이후)의 β0 로 직접 얻음
    d2 = d.copy()
    d2["S"] = 1 - d2["S"]
    내생2 = pd.DataFrame({"유입": Flow5, "유입xS": Flow5 * (1 - S)}).loc[평일_확장]
    도구2 = pd.DataFrame({"GIV": GIV5_확장, "GIVxS": GIV5_확장 * (1 - S.reindex(평일_확장))}).loc[평일_확장]
    r2 = 회귀(d2, 내생2, 도구2, 겹침시차(h))
    변화행.append({"h": h, "관측치": r["관측치"], "4월이후_관측치": int(d.dropna()["S"].sum()),
                 "기준bp": b0 * 충격 * 100, "기준se": r[("유입", "표준오차_겹침")] * 충격 * 100,
                 "이후bp": r2[("유입", "계수")] * 충격 * 100, "이후se": r2[("유입", "표준오차_겹침")] * 충격 * 100,
                 "차이bp": b1 * 충격 * 100, "차이se": r[("유입xS", "표준오차_겹침")] * 충격 * 100,
                 "1단계F_유입": r[("유입", "1단계F")], "1단계F_유입xS": r[("유입xS", "1단계F")]})
변화 = pd.DataFrame(변화행)
변화["차이p"] = [p값(b / s) for b, s in zip(변화["차이bp"], 변화["차이se"])]
변화["최소검출차이bp"] = MDE배수 * 변화["차이se"]
변화.to_csv(데이터폴더 / "확장_변화검정.csv", index=False, encoding=CSV인코딩)

print("[변화 검정] 식 (5), 상태 = 2026-04-01 이후 (bp, 겹침 NW)")
v = 변화.set_index("h")
print(v.loc[보일h, ["4월이후_관측치", "기준bp", "이후bp", "차이bp", "차이se", "차이p", "최소검출차이bp",
                  "1단계F_유입", "1단계F_유입xS"]].round(2).to_string())
판정나 = []
for h in 판정h:
    달라짐 = v.loc[h, "차이p"] < 유의수준
    판정나.append({"h": h, "차이bp": v.loc[h, "차이bp"], "p": v.loc[h, "차이p"], "최소검출차이bp": v.loc[h, "최소검출차이bp"],
                 "판정": "4월 이후 효과가 달라짐" if 달라짐 else
                 f"차이 유의하지 않음 - 약 {v.loc[h, '최소검출차이bp']:.1f}bp 보다 작은 변화는 판단 불가"})
판정나 = pd.DataFrame(판정나)
print("\n[(나)(다) 판정]")
print(판정나.round(3).to_string(index=False))
if (v[["1단계F_유입", "1단계F_유입xS"]].min() < 10).any():
    print("  [주의] 1단계 F 가 10 미만인 h 가 있음 - 그 h 의 변화 검정은 약한 도구 문제로 해석 주의")

pd.concat([판정표.assign(구분="(가) 유지", 최종=최종가),
           판정나.rename(columns={"판정": "값"}).assign(구분="(나)(다) 변화")], ignore_index=True) \
    .to_csv(데이터폴더 / "확장_판정.csv", index=False, encoding=CSV인코딩)
print("저장: 확장_그림7_결과.csv, 확장_변화검정.csv, 확장_판정.csv")


#%% [셀 9] 그림 =============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def 띠(ax, h, b, se, 색):
    ax.fill_between(h, b - Z95 * se, b + Z95 * se, color=색, alpha=0.25, lw=0)
    ax.fill_between(h, b - Z68 * se, b + Z68 * se, color=색, alpha=0.45, lw=0)
    ax.plot(h, b, color=색)


# (1) 확장 표본 전체 - 그림 7 형식
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, 방법 in zip(axes, ["통제만", "IV"]):
    a = 확장.query("방법 == @방법")
    띠(ax, a["h"], a["계수"] * 충격, a["표준오차_겹침"] * 충격, "#2171b5")
    b = 기준.query("방법 == @방법")
    ax.plot(b["h"], b["계수"] * 충격, color="black", ls="--", lw=1, label="2026-03-31 까지 (새 자료)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"$3.5B 유입 효과 ({방법})")
    ax.set_xlabel("기간 (거래일)")
    ax.set_xlim(0, 40)
    ax.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel("3개월 T-bill 금리 반응 (%)")
fig.suptitle(f"확장 표본 {표본시작.date()} ~ {확장끝.date()} (파랑, 68%/95% NW 구간) vs 3월까지 (점선)", fontsize=10)
fig.tight_layout()
fig.savefig(그림폴더 / "확장_그림7_전체기간.png", dpi=150)
plt.close(fig)

# (2) 식 (5): 3월까지 vs 4월 이후 (IV)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
띠(axes[0], v.index, v["기준bp"] / 100, v["기준se"] / 100, "#2171b5")
띠(axes[0], v.index, v["이후bp"] / 100, v["이후se"] / 100, "#e6550d")
axes[0].set_title("IV 효과: 파랑 = 2026-03 까지, 주황 = 2026-04 이후")
띠(axes[1], v.index, v["차이bp"] / 100, v["차이se"] / 100, "#636363")
axes[1].set_title("차이 (4월 이후 - 3월까지)")
for ax in axes:
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("기간 (거래일)")
    ax.set_xlim(0, 40)
axes[0].set_ylabel("3개월 T-bill 금리 반응 (%)")
fig.suptitle("식 (5) 상태 의존 IRF, 상태 = 2026-04-01 이후 (68%/95% NW 구간)", fontsize=10)
fig.tight_layout()
fig.savefig(그림폴더 / "확장_그림7_4월전후.png", dpi=150)
plt.close(fig)
print("저장: 그림/확장_그림7_전체기간.png, 그림/확장_그림7_4월전후.png")


#%% [셀 10] 점검: 사전 추세 (확장 표본, 기록용) ================================
# BIS_4 [셀 9] 와 같은 점검 (Stock & Watson 2018 - 도구의 과거 충격과의 무관성)
#   결과변수: y_{t−5j} − y_{t−5j−5}, j = 0 (유입 중, 판정 대상 아님), 1 ~ 4 / NW 시차 4
사전행 = []
for j in range(0, 5):
    yj = (y3.shift(5 * j) - y3.shift(5 * j + 5)).rename("y")
    for 통제이름, 통제표 in [("통제 없음", pd.DataFrame(index=평일_넓게)),
                            ("통제 있음", X.drop(columns=["지난변화_3개월"]))]:
        d = pd.concat([yj, 통제표], axis=1).loc[평일_확장]
        r = 회귀(d, Flow5.loc[평일_확장].to_frame("유입"), GIV5_확장.to_frame("GIV"), 4)
        사전행.append({"j": j, "통제": 통제이름, "효과bp": r[("유입", "계수")] * 충격 * 100,
                     "p_시차4": p값(r[("유입", "계수")] / r[("유입", "표준오차_겹침")]), "관측치": r["관측치"]})
사전 = pd.DataFrame(사전행)
사전.to_csv(데이터폴더 / "확장_사전추세.csv", index=False, encoding=CSV인코딩)
print("[사전 추세, 확장 표본] IV: $3.5B 유입과 그 전 5일 변화 (bp, 별 = NW 시차 4)")
for 통제이름 in ["통제 없음", "통제 있음"]:
    a = 사전.query("통제 == @통제이름").set_index("j")
    print(f"  <{통제이름}>  " + "  ".join(
        f"{'유입 중' if j == 0 else f'{j}주 전'} {a.loc[j, '효과bp']:+.2f}{별(a.loc[j, 'p_시차4'])}" for j in range(5)))
print(f"  j = 1 ~ 4 중 5% 유의: {len(사전.query('j >= 1 and p_시차4 < @유의수준'))}칸 / 8칸 (기록용, BIS_4: 1칸)")
print("저장: 확장_사전추세.csv")
