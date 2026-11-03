# -*- coding: utf-8 -*-
"""
퀀트랩_BIS_1.py
=====================================================================
[논문] Ahmed & Aldasoro (2026), "Stablecoins and safe asset prices",
       BIS Working Paper No 1270 (2026년 6월 개정판)
[단계] 복제 1단계: 데이터 수집 + 그림 2, 그림 3 재현

[출처 표기 규칙]  모든 설정·처리 옆에 아래 셋 중 하나를 적었습니다.
  # 출처: 논문 ○○절 / 각주 ○ / 그림 ○ 주석 / 부록 ○   -> 논문 근거
  # 출처: 사용자 지시                                  -> 연구자가 정한 조건
  # [논문에 명시 없음]                                 -> 논문에 방법이 안 나와 있어서
                                                         코드를 돌리려면 어쩔 수 없이
                                                         필요한 부분 (결과 숫자에 영향이
                                                         있는지 함께 적음)

[Spyder 사용법]
  - "#%%" 줄이 셀 구분선입니다. Shift+Enter 로 [셀 0]부터 순서대로 실행하세요.
  - 앞 셀에서 만든 변수를 뒤 셀이 사용합니다.

[필요한 패키지]  pip install requests pandas numpy matplotlib
=====================================================================
"""

#%% [셀 0] 패키지 + 설정 =================================================
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------
# FRED API 키
# 출처: 사용자 지시 (FRED API 사용)
# 키는 코드에 적지 않고 프로젝트 폴더의 FRED_키.txt 에서 읽음. 환경변수 FRED_API_KEY 가 있으면 그것이 먼저
# ---------------------------------------------------------------
_키파일 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "FRED_키.txt"   # GitHub에 올리지 않는 파일
FRED_API_KEY = os.environ.get("FRED_API_KEY") or (
    _키파일.read_text(encoding="utf-8").strip() if _키파일.exists() else "여기에_FRED_키_붙여넣기")

# ---------------------------------------------------------------
# 표본 시작일
# 출처: 논문 2.1절 "daily data from January 2021 to March 2026"
#       + 사용자 지시 "2021년 1월부터 현재까지"
# 종료일은 따로 정하지 않고, DefiLlama가 주는 마지막 날짜까지 씁니다.
# (출처: 사용자 지시 "현재까지")
# ---------------------------------------------------------------
시작일 = pd.Timestamp("2021-01-01")

# ---------------------------------------------------------------
# 표본 종료일 (둘 중 하나 선택)
#   None                         -> DefiLlama 마지막 날짜까지 (출처: 사용자 지시 "현재까지")
#   pd.Timestamp("2026-03-31")   -> 논문 표본과 동일
#       출처: 논문 표 1 주석 "Sample: January 4, 2021 to March 31, 2026"
#             논문 2.1절 "daily data from January 2021 to March 2026"
# 셀 2(합산 시총), 셀 3(체인 패널), 셀 5(FRED 금리) 모두에 똑같이 적용됩니다.
# ---------------------------------------------------------------
#종료일 = None
종료일 = pd.Timestamp("2026-03-31")   # 논문 표본으로 돌릴 때 이 줄의 # 을 지우세요

# ---------------------------------------------------------------
# 분석 대상 7개 스테이블코인
# 출처: 논문 2.1절 + 각주 4 (USDT, USDC, TUSD, BUSD, FDUSD, PYUSD, RLUSD)
#       + 사용자 지시 (시가총액을 CoinMarketCap 대신 DefiLlama에서 수집)
# ---------------------------------------------------------------
코인목록 = ["USDT", "USDC", "TUSD", "BUSD", "FDUSD", "PYUSD", "RLUSD"]

# ---------------------------------------------------------------
# 그림 2 왼쪽 패널의 "Other" 구성
# 출처: 논문 그림 2 주석 "'Other' (TUSD, BUSD, FDUSD, PYUSD, and RLUSD)"
# ---------------------------------------------------------------
기타코인 = ["TUSD", "BUSD", "FDUSD", "PYUSD", "RLUSD"]

# ---------------------------------------------------------------
# FRED 금리 만기
# 출처: 논문 2.1절 "1-month, 3-month, 6-month, 1-year, 2-year, and 10-year"
# [논문에 명시 없음] 논문은 "FRED"라고만 쓰고 시리즈 코드를 밝히지 않습니다.
#   여기서는 FRED의 만기별 국채 수익률(DGS 계열)을 썼습니다.
#   검증: 퀀트랩_BIS_4.py [셀 2]에서 논문 표 5(T, 평균, 표준편차, 최소, 최대)로 판별
#     3개월물: DGS3MO 가 표 5와 일치 (값 차이 합 0.006). 후보 DTB3 는 최댓값 5.36 으로 표 5의 5.63과 불일치
#     1M·6M·1Y·2Y·10Y: DGS 계열이 표 5와 값 차이 합 0.01 이하로 일치
# ---------------------------------------------------------------
FRED_시리즈 = {
    "1-month Treasury": "DGS1MO",
    "3-month Treasury": "DGS3MO",
    "6-month Treasury": "DGS6MO",
    "1-year Treasury": "DGS1",
    "2-year Treasury": "DGS2",
    "10-year Treasury": "DGS10",
}

# ---------------------------------------------------------------
# 저장 폴더
# 출처: 사용자 지시 (바탕화면/스테이블코인_퀀트랩 안에 저장, 폴더·파일명 한국어)
#
#   바탕화면/
#    └─ 스테이블코인_퀀트랩/
#        ├─ 데이터/   CSV 파일
#        └─ 그림/     PNG 파일
#
# 바탕화면 경로: 윈도우 바탕화면의 실제 폴더 = C:\Users\사용자이름\Desktop
#   (탐색기에서는 "바탕 화면"으로 보이지만 실제 폴더 이름은 Desktop)
# ---------------------------------------------------------------
바탕화면 = Path.home() / "Desktop"

결과폴더 = 바탕화면 / "스테이블코인_퀀트랩"
데이터폴더 = 결과폴더 / "데이터"
그림폴더 = 결과폴더 / "그림"
데이터폴더.mkdir(parents=True, exist_ok=True)
그림폴더.mkdir(parents=True, exist_ok=True)

# CSV를 엑셀에서 열어도 한글이 안 깨지게 저장하는 방식 (출처: 사용자 지시 - 한국어 파일)
CSV인코딩 = "utf-8-sig"

# DefiLlama 스테이블코인 API 주소 (출처: 사용자 지시 - DefiLlama 사용)
LLAMA = "https://stablecoins.llama.fi"
FRED = "https://api.stlouisfed.org/fred/series/observations"

print("결과 폴더:", 결과폴더)


#%% [셀 1] DefiLlama에서 7개 코인의 ID 확인 ================================
# DefiLlama는 코인마다 번호(ID)가 있어서, 심볼로 번호를 찾아야 데이터를 받을 수 있습니다.
# [논문에 명시 없음] API 사용 절차상 필요. 결과 숫자에는 영향 없음.
#
# 같은 심볼이 여러 개 나오면 코드가 마음대로 고르지 않고 멈춥니다.
# 그때 출력된 후보를 보고 아래 ID_직접지정 에 번호를 적은 뒤 다시 실행하세요.

# BUSD 만 후보가 3개: 4 Binance USD / 153 Binance Peg BUSD / 231 Bera USD
#   -> 4 선택 (출처: 논문 각주 4 "Binance USD (BUSD)" 와 이름 일치)
#      153 은 원래 BUSD 를 맡겨 두고 다른 체인에 발행한 포장 토큰이라 포함하면 이중 계산, 231 은 심볼만 같은 별개 코인
# 나머지 6개(USDT, USDC, TUSD, FDUSD, PYUSD, RLUSD)는 같은 심볼이 하나뿐이라 자동 확정
ID_직접지정 = {"BUSD": "4"}

목록 = requests.get(f"{LLAMA}/stablecoins", timeout=120).json()["peggedAssets"]

코인ID = {}
판단필요 = {}
for 심볼 in 코인목록:
    if 심볼 in ID_직접지정:
        코인ID[심볼] = str(ID_직접지정[심볼])
        continue
    후보 = [a for a in 목록 if str(a.get("symbol", "")).upper() == 심볼]
    if len(후보) == 1:
        코인ID[심볼] = str(후보[0]["id"])
    else:
        판단필요[심볼] = [(a["id"], a["name"], a.get("pegType"),
                        (a.get("circulating") or {}).get("peggedUSD")) for a in 후보]

print("확인된 ID:", 코인ID)
if 판단필요:
    for 심볼, 후보 in 판단필요.items():
        print(f"\n!! {심볼}: 후보 {len(후보)}개 -> ID_직접지정 에 하나를 적으세요")
        for c in 후보:
            print(f"   id={c[0]}, 이름={c[1]}, 페그={c[2]}, 현재유통량={c[3]}")
    raise SystemExit("ID를 직접 지정한 뒤 [셀 1]을 다시 실행하세요.")


#%% [셀 2] 코인별 시가총액 수집 → 7개 합산 ================================
# 출처: 논문 2.1절 "we collect market capitalization data ... on seven USD-backed
#       stablecoins ... We aggregate these series into a single market
#       capitalization measure"
# 출처: 논문 2.2절 식 (1) 아래 "MC_t ... product of the stablecoin price and
#       outstanding supply: P_t × S_t"
#   -> DefiLlama의 totalCirculatingUSD(유통량 × 가격 = 달러 가치)를 사용
# 출처: 사용자 지시 (CoinMarketCap 대신 DefiLlama)

행들 = []
for 심볼, 번호 in 코인ID.items():
    자료 = requests.get(f"{LLAMA}/stablecoincharts/all",
                      params={"stablecoin": 번호}, timeout=120).json()
    for r in 자료:
        행들.append({
            "date": pd.to_datetime(int(r["date"]), unit="s"),
            "symbol": 심볼,
            "mcap": (r.get("totalCirculatingUSD") or {}).get("peggedUSD"),
        })
    print(f"  {심볼}: {sum(1 for x in 행들 if x['symbol'] == 심볼):,}일")

합산_long = pd.DataFrame(행들)
합산_long = 합산_long[합산_long["date"] >= 시작일]
if 종료일 is not None:                                   # [셀 0]의 종료일 적용
    합산_long = 합산_long[합산_long["date"] <= 종료일]

# 날짜 × 코인 표로 바꾸고 7개를 더함
# 출처: 논문 각주 12 "RLUSD launched in late 2024 and is included from its inception"
#   -> 출시 전 코인은 값이 없으므로 합계에서 빠짐 (pandas sum 은 빈 칸을 건너뜀)
합산시총 = 합산_long.pivot_table(index="date", columns="symbol", values="mcap", aggfunc="sum")
합산시총 = 합산시총[[c for c in 코인목록 if c in 합산시총.columns]]
합산시총["Total"] = 합산시총[코인목록].sum(axis=1)
합산시총.index.name = "date"

합산시총.to_csv(데이터폴더 / "스테이블코인_합산시총_일별.csv", encoding=CSV인코딩)
print(f"\n기간: {합산시총.index.min().date()} ~ {합산시총.index.max().date()}")
print(f"마지막 날 합계: ${합산시총['Total'].iloc[-1] / 1e9:,.1f}B  (논문 2.1절: 2026년 3월 약 $270B)")


#%% [셀 3] 코인 × 블록체인 패널 수집 =====================================
# 7개 코인이 블록체인별로 얼마씩 발행돼 있는지를 날짜별로 모으는 셀
# 출처: 논문 2.1절 "we construct a daily stablecoin-chain panel from DeFiLlama"
# 출처: 논문 1장 "We use data on market capitalization at the stablecoin-chain
#       pair level from DeFiLlama"
# 출처: 부록 C.1 + 표 2 3열 "DeFiLlama-only panel"
#   -> Token Terminal 대체(표 4의 12개 페어)는 하지 않고 DefiLlama 원자료 그대로 사용
#   -> 따라서 표 4에서 지적된 이상값(예: TUSD/Ethereum 음수)도 그대로 둡니다.
#
# DefiLlama 체인별 자료는 달러 가치가 아니라 유통 "개수"(circulating)만 제공합니다.
# 출처: 논문 2.2절 "Equation (1) therefore makes the simplifying assumption that
#       stablecoins maintain their peg of P_t = $1"
#   -> 가격 = $1 가정에 따라 유통량을 그대로 시장가치로 사용
# 하지 않는 것
# 음수 값 제거, 빠진 날 채우기, 이상값 수정은 하지 않습니다. 
# 부록 C.1과 표 2 3열에 따르면 DeFiLlama-only 버전은 Token Terminal 대체 없이 DefiLlama 원자료를 그대로 쓰기 때문

행들 = []
for 심볼, 번호 in 코인ID.items():
    자료 = requests.get(f"{LLAMA}/stablecoin/{번호}", timeout=120).json()
    체인자료 = 자료.get("chainBalances", {}) or {}
    for 체인, 내용 in 체인자료.items():
        for t in (내용 or {}).get("tokens", []) or []:
            행들.append({
                "date": pd.to_datetime(int(t["date"]), unit="s"),
                "symbol": 심볼,
                "chain": 체인,
                "mcap": (t.get("circulating") or {}).get("peggedUSD"),
            })
    print(f"  {심볼}: 체인 {len(체인자료)}개")

패널 = pd.DataFrame(행들)
패널 = 패널[패널["date"] >= 시작일]
if 종료일 is not None:                                   # [셀 0]의 종료일 적용
    패널 = 패널[패널["date"] <= 종료일]
패널["pair"] = 패널["symbol"] + " / " + 패널["chain"]   # 논문 그림 3 표기 방식 "USDT / Tron"
패널 = 패널.sort_values(["date", "symbol", "chain"]).reset_index(drop=True)
패널 = 패널[["date", "symbol", "chain", "pair", "mcap"]]

패널.to_csv(데이터폴더 / "코인체인_패널_일별.csv", index=False, encoding=CSV인코딩)
print(f"\n저장: 코인체인_패널_일별.csv ({len(패널):,}행)")


#%% [셀 4] 하루 활성 페어 수 확인 (논문 수치와 비교) =======================
# 이 셀은 데이터를 바꾸지 않고 확인만 합니다.
# 출처: 논문 각주 18 "We classify a unit as active at t if its market
#       capitalization exceeds $1 million on both t − 1 and t"
# 비교값 출처: 논문 각주 17 "The median number of pairs active in the sample is 89,
#       ranging from 12 in the early thin period to 143 at peak"

넓게 = 패널.pivot_table(index="date", columns="pair", values="mcap", aggfunc="sum")  # 날짜 × 페어 표 (셀 9에서도 사용)
활성 = (넓게 > 1e6) & (넓게.shift(1) > 1e6)
활성수 = 활성.sum(axis=1)
# 비교 범위 = 표본 시작일(2021-01-04) 이후 (출처: 논문 표 1 주석 "Sample: January 4, 2021 to March 31, 2026")
#   패널은 2021-01-01부터 받았지만 1월 1~3일은 표본 밖. 1월 1일은 전날 자료가 없어 활성 0, 1월 2·3일 최솟값 11
표본활성수 = 활성수.loc["2021-01-04":]
print(f"하루 활성 페어 (2021-01-04 이후): 중앙값 {표본활성수.median():.0f}, "
      f"범위 {표본활성수.min()}~{표본활성수.max()}  (각주 17: 중앙값 89, 12~143)")

# DefiLlama가 과거 자료를 사후 수정하는 경우가 있어서, 저자들이 받은 시점과 지금 데이터가 조금 다른 정도
# (저자 답신 2026-10-12, 이메일: DefiLlama 자료는 가끔 과거 값까지 사후 수정됨)

#%% [셀 5] FRED 미국채 금리 수집 ==========================================
# 출처: 논문 2.1절 "We collect daily data on US market interest rates from FRED"
if "붙여넣기" in FRED_API_KEY:
    raise ValueError("[셀 0]: 프로젝트 폴더에 FRED_키.txt 를 만들고 FRED API 키를 한 줄로 넣으세요 (또는 환경변수 FRED_API_KEY).")

금리 = {}
for 이름, 코드 in FRED_시리즈.items():
    자료 = requests.get(FRED, timeout=120, params={
        "series_id": 코드, "api_key": FRED_API_KEY, "file_type": "json",
        "observation_start": 시작일.strftime("%Y-%m-%d"),
    }).json()
    # FRED는 휴장일 값을 "." 으로 줍니다 → 빈 값 처리 (FRED 자료 형식상 필요)
    금리[이름] = pd.Series(
        {pd.to_datetime(o["date"]): (np.nan if o["value"] == "." else float(o["value"]))
         for o in 자료["observations"]})
    print(f"  {코드}: {금리[이름].notna().sum():,}개")

국채금리 = pd.DataFrame(금리)
if 종료일 is not None:                                   # [셀 0]의 종료일 적용
    국채금리 = 국채금리[국채금리.index <= 종료일]
국채금리.index.name = "date"
국채금리.to_csv(데이터폴더 / "미국채_금리_일별.csv", encoding=CSV인코딩)
print("저장: 미국채_금리_일별.csv")

# 3개월물 = DGS3MO 로 확정 (퀀트랩_BIS_4.py [셀 2] 표 5 판별 결과, [셀 0] FRED 시리즈 주석 참고)

#%% [셀 6] 그림 2-1: 스테이블코인 시가총액 (누적 면적) =======================
# 출처: 논문 그림 2 왼쪽 패널 + 주석 "aggregate USD-backed stablecoin market
#       capitalization, broken out by USDT, USDC, and 'Other'"
# 색·쌓는 순서: 논문 그림 2 왼쪽 패널을 보고 맞춤 (아래 Other, 가운데 USDT, 위 USDC)
# [논문에 명시 없음] 그림 크기, 정확한 색상 코드 -> 결과 숫자에 영향 없음

fig, ax = plt.subplots(figsize=(4.2, 5.2))
ax.stackplot(합산시총.index,
             합산시총[기타코인].sum(axis=1) / 1e9,
             합산시총["USDT"] / 1e9,
             합산시총["USDC"] / 1e9,
             colors=["#c2eaf9", "#6fa6cd", "#1b1a6b"],
             labels=["Other", "USDT", "USDC"], linewidth=0)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", frameon=False)
ax.set_ylabel("Stablecoin market capitalization ($B)")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(color="#e6e6e6")
fig.savefig(그림폴더 / "그림2-1.png", dpi=200, bbox_inches="tight")
plt.show()


#%% [셀 7] 그림 2-2: 미국채 만기별 금리 ====================================
# 출처: 논문 그림 2 오른쪽 패널 + 주석 "daily US Treasury yields for several maturities"
# 색: 논문 그림 2 오른쪽 패널처럼 짧은 만기일수록 진한 파랑
# [논문에 명시 없음] 정확한 색상 코드 -> 결과 숫자에 영향 없음

fig, ax = plt.subplots(figsize=(4.2, 5.2))
for 열, 색 in zip(국채금리.columns,
                 ["#0b1a5c", "#0e3f8f", "#0b6cb3", "#0a93d6", "#1cb4ec", "#8ed1ee"]):
    s = 국채금리[열].dropna()
    ax.plot(s.index, s.values, color=색, lw=0.9, label=열)
ax.set_ylabel("Interest rate (%)")
ax.legend(loc="lower right", fontsize=7.5)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(color="#e6e6e6")
fig.savefig(그림폴더 / "그림2-2.png", dpi=200, bbox_inches="tight")
plt.show()


#%% [셀 8] 그림 3-1: 마지막 날 시총 상위 10개 코인-체인 페어 ================
# 출처: 논문 그림 3 주석 "The left panel reports the top 10 stablecoin-chain pairs
#       by market capitalization in billions of US dollars at the end of the panel.
#       Bars are colored by stablecoin issuer."
# 색: 논문 그림 3 왼쪽 패널 (USDT 남색, USDC 하늘색, PYUSD 노랑)
# [논문에 명시 없음] 논문 그림에 안 나오는 코인(TUSD·BUSD·FDUSD·RLUSD)은
#   색이 정해져 있지 않아 회색으로 둠 -> 결과 숫자에 영향 없음

코인색 = {"USDT": "#2b2a78", "USDC": "#7eaed3", "PYUSD": "#e8b33a"}

마지막날 = 패널["date"].max()
상위10 = (패널[패널["date"] == 마지막날]
        .sort_values("mcap", ascending=False).head(10).iloc[::-1])

fig, ax = plt.subplots(figsize=(5.2, 4.0))
막대 = ax.barh(상위10["pair"], 상위10["mcap"] / 1e9,
             color=[코인색.get(s, "#999999") for s in 상위10["symbol"]])
for b, v in zip(막대, 상위10["mcap"] / 1e9):
    ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.1f}",
            va="center", fontsize=7.5)
ax.set_xlabel("Market capitalization ($B)")
ax.set_title(f"Top 10 stablecoin-chain pairs by market cap\n{마지막날:%B %d, %Y}",
             loc="left", fontsize=10)
ax.tick_params(axis="y", labelsize=7.5)
ax.grid(axis="x", color="#e6e6e6")
fig.savefig(그림폴더 / "그림3-1.png", dpi=200, bbox_inches="tight")
plt.show()

print(상위10.iloc[::-1][["pair", "mcap"]].assign(mcap=lambda d: (d["mcap"] / 1e9).round(1))
      .to_string(index=False))

# USDC / Arbitrum이 논문값과 다른 이유(Appendix C 참고)
# USDC / Arbitrum은 저자들이 DefiLlama 대신 Token Terminal 자료로 바꾼 12개 페어 중 하나
# 바꾼 이유(분류 L)는 "DefiLlama 수준값이 외부 기준과 맞지 않음", 그게 뭔지 밝히진 않음

#%% [셀 9] 그림 3-2: 상위 k개 페어 점유율 ===================================
# 출처: 논문 그림 3 주석 "The right panel plots the market capitalization share of
#       the top-K stablecoin-chain pairs at daily frequency over time, for K = 1, 3, 5, 10."
# 출처: 논문 2.2절 "a single pair accounts for roughly one third ...; the top three
#       pairs together account for around 80%; and the top ten account for around 95%"
#   -> 분모 = 그날 패널 전체 페어 시총 합
# 매일 상위 몇 개 페어가 전체 시총의 몇 %를 차지하는지 계산해서 그림 3-2를 그리는 셀
# 논문 2.2절이 말하는 "시장이 몇 개 페어에 쏠려 있다"는 점, 즉 GIV를 쓰는 근거를 보여주는 그림

정렬 = -np.sort(-넓게.fillna(0).values, axis=1)   # 날짜마다 큰 순으로 정렬
누적 = np.cumsum(정렬, axis=1)
합계 = 정렬.sum(axis=1)
집중도 = pd.DataFrame({f"Top {k}": 누적[:, k - 1] / 합계 * 100 for k in (1, 3, 5, 10)},
                   index=넓게.index)
집중도.to_csv(데이터폴더 / "상위k_집중도.csv", encoding=CSV인코딩)

fig, ax = plt.subplots(figsize=(5.2, 4.0))
for 열, 색 in zip(집중도.columns, ["#0b1066", "#6a9fd0", "#bfe4f7", "#3f82b5"]):
    ax.plot(집중도.index, 집중도[열], color=색, lw=1.2, label=열)
ax.set_ylim(0, 102)
ax.set_ylabel("Share of total stablecoin market cap (%)")
ax.set_title("Top-k stablecoin-chain pair concentration", loc="left", fontsize=10)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(color="#e6e6e6")
fig.savefig(그림폴더 / "그림3-2.png", dpi=200, bbox_inches="tight")
plt.show()

print("평균 점유율(%):", 집중도.mean().round(1).to_dict(),
      " (2.2절: Top1 약 33, Top3 약 80, Top10 약 95)")
