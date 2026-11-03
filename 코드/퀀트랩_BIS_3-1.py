# -*- coding: utf-8 -*-
"""
퀀트랩_BIS_3-1.py
=====================================================================
[논문] Ahmed & Aldasoro (2026), "Stablecoins and safe asset prices",
       BIS Working Paper No 1270 (2026년 6월 개정판)
[단계] 복제 3-1단계: 배제 조건 ④(교란변수 상관, 표 6)의 "대체 자료" 분석 - 2차
       퀀트랩_BIS_3.py 에서 빠진 변수를 무료(또는 학교 구독) 자료로 대신해 같은 계산을 함
       (출처: 사용자 지시 - 유료 자료를 무료 자료로 대체하는 분석은 별도 파일로)

[중요] 이 파일의 결과는 논문과 "같은 자료"로 한 재현이 아닙니다.
       퀀트랩_BIS_3.py(논문 자료와 같은 FRED·Yahoo 변수)의 결과와 섞지 말고 따로 보고합니다.

[출처 표기 규칙]
  # 출처: 논문 ○○절 / 각주 ○ / 표 ○ 주석   -> 논문 근거
  # 출처: 사용자 지시                       -> 연구자가 정한 조건
  # 출처: 퀀트랩_BIS_2.py / 퀀트랩_BIS_3.py -> 앞 단계에서 논문 수치로 확정한 사항
  # [논문에 명시 없음]                      -> 논문에 방법이 안 나와 있는 부분 (이유·영향 함께)
  # [대체 자료]                             -> 논문과 다른 자료를 쓰는 부분 (무엇을 무엇으로 바꿨는지)

[다루는 변수]
  원래 변수 (논문 출처)                     ->  대체 자료
  log gold price (FRED·Bloomberg)           ->  Yahoo GC=F   (COMEX 금 선물 근월물)
  log Brent price (FRED·Bloomberg)          ->  Yahoo BZ=F   (브렌트유 선물 근월물)
  log BGCI (Bloomberg)                      ->  Nasdaq Crypto Index (NCI), Nasdaq 지수 사이트에서 받은 파일
                                                (Yahoo ^NCI 는 "possibly delisted; no price data found"로 받지 못함)
  ETF flow (ETF.com)                        ->  운용사 일별 파일(BIL·SPTS·SGOV) + CRSP(SHV·SHY·VGSH·SCHO)
  Bloomberg US FCI, Goldman FCI (Bloomberg) ->  OFR 금융스트레스지수 (OFR FSI)
[한계로 남기는 변수] (출처: 사용자 지시 - 구하기 어려운 자료는 한계로)
  MMF flow (iMoneyNet, 유료) - 무료 대체(ICI 주간 MMF 자산)도 구하지 못함

[입력] 모두 바탕화면/스테이블코인_퀀트랩/데이터/ 안에 아래 이름으로 저장 (출처: 사용자 지시 - 한국어 파일명)
  GIV_선택조합_평일.csv        (퀀트랩_BIS_2.py [셀 6])
  교란변수_원자료_일별.csv      (퀀트랩_BIS_3.py [셀 5], 브렌트 현물 비교용)
  ETF_CRSP_일별.csv             <- WRDS CRSP 에서 받은 파일 (BILSGOVSHVSHYVGSHSCHOSPTS.csv)
  ETF_BIL_운용사.xlsx           <- SPDR 에서 받은 navhist-us-en-bil.xlsx
  ETF_SPTS_운용사.xlsx          <- SPDR 에서 받은 navhist-us-en-spts.xlsx
  ETF_SGOV_운용사.xls           <- iShares 에서 받은 iShares-0-3-Month-Treasury-Bond-ETF_fund.xls
  OFR_금융스트레스지수.csv      <- OFR 에서 받은 fsi.csv
  암호자산지수_NCI_나스닥.xlsx  <- Nasdaq 지수 사이트에서 받은 EODHist_20210101-20260930_NCI.xlsx

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install pandas numpy statsmodels yfinance
=====================================================================
"""

#%% [셀 0] 패키지 + 설정 =================================================
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# 폴더 (출처: 사용자 지시 - 바탕화면/스테이블코인_퀀트랩, 한국어 파일명)
결과폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩"
데이터폴더 = 결과폴더 / "데이터"
CSV인코딩 = "utf-8-sig"

# 표본 기간 (출처: 논문 표 1 주석 "Sample: January 4, 2021 to March 31, 2026")
표본시작 = pd.Timestamp("2021-01-04")
표본끝 = pd.Timestamp("2026-03-31")

# 평일 달력 (출처: 퀀트랩_BIS_2.py [셀 0] - 표 1·표 5 의 T = 1,367 로 역산)
평일 = pd.bdate_range(표본시작, 표본끝)

# 5일 변화를 구하려면 표본 시작 전 5 거래일 값이 필요 -> 조금 앞에서부터 받음
# (출처: 퀀트랩_BIS_3.py [셀 5]와 같음, 계산상 필요)
받기시작 = pd.Timestamp("2020-12-01")
평일_넓게 = pd.bdate_range(받기시작, 표본끝)

# 논문 비교값
# 출처: 표 5 (T, 평균, 표준편차, 최소, 최대) - 로그 수준
표5 = {"금": (1367, 7.73, 0.30, 7.40, 8.58), "브렌트": (1367, 4.37, 0.16, 3.93, 4.85)}
# 출처: 표 1 - 5-day ETF flow ($B)
표1_ETF = (1362, 0.58, 1.59, -4.21, 9.41)
# 출처: 표 6 - (5일, 일별)
표6 = {"금": ("+0.035", "+0.047*"), "브렌트": ("-0.033", "-0.011"),
       "암호자산지수": ("-0.070", "-0.007"), "ETF 유입": ("+0.097**", "+0.056*"),
       # 금융여건지수 2개 (BBG = Bloomberg US FCI, GS = Goldman FCI) - OFR FSI 와 비교용
       "금융스트레스지수": ("BBG -0.036 / GS -0.039", "BBG +0.006 / GS +0.003")}

print("데이터 폴더:", 데이터폴더)
print("평일 수:", len(평일), " (표 1·표 5: 1,367)")


#%% [셀 1] GIV 불러오기 + 상관 계산 함수 =====================================
# GIV를 불러오고, 이후 모든 변수에 똑같이 쓸 상관 계산 함수 세 개를 정의하는 셀
# GIV (출처: 퀀트랩_BIS_2.py [셀 6] - 저자 답신(2026-10-12)으로 확정한 설정의 GIV, 평일 기준)
# 계산 방법은 퀀트랩_BIS_3.py [셀 7]과 완전히 같게 맞췄습니다. 그래야 논문 자료(퀀트랩_BIS_3.py)와 대체 자료(퀀트랩_BIS_3-1.py)의 결과를 같은 기준으로 비교
GIV = pd.read_csv(데이터폴더 / "GIV_선택조합_평일.csv", parse_dates=["date"], index_col="date",
                  encoding=CSV인코딩)

# 상관계수와 유의성 (출처: 퀀트랩_BIS_3.py [셀 7]과 완전히 같은 방법)
#   출처: 표 6 주석 "Newey–West HAC standard errors with one lag, with *, **, *** denoting
#         significance at the 10, 5 and 1% level"
#   [논문에 명시 없음] 유의성 계산의 회귀 방향 -> 두 방향 모두 표시 (퀀트랩_BIS_3.py 와 같음)
def 별표(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def HAC상관(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    za = (d.iloc[:, 0] - d.iloc[:, 0].mean()) / d.iloc[:, 0].std(ddof=0)
    zb = (d.iloc[:, 1] - d.iloc[:, 1].mean()) / d.iloc[:, 1].std(ddof=0)
    ρ = float((za * zb).mean())
    p = []
    for y_, x_ in [(zb, za), (za, zb)]:       # (교란변수 ~ GIV), (GIV ~ 교란변수)
        r = sm.OLS(y_, sm.add_constant(x_)).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        p.append(float(r.pvalues.iloc[1]))
    return ρ, p[0], p[1], len(d)


def 상관한줄(이름, 변화5, 변화1, 휴장일처리):
    """5일: GIV 5일 합 vs 5일 변화 / 일별: 일별 GIV vs 1일 변화 (출처: 표 6 주석 빈도 맞춤)"""
    ρ5, p5a, p5b, n5 = HAC상관(GIV["GIV_5일합"], 변화5)
    ρ1, p1a, p1b, n1 = HAC상관(GIV["GIV_일별"], 변화1)
    return {"휴장일처리": 휴장일처리, "변수": 이름,
            "5일_ρ": round(ρ5, 3), "5일_별(교란~GIV)": 별표(p5a), "5일_별(GIV~교란)": 별표(p5b),
            "5일_T": n5, "5일_표6": 표6[이름][0],
            "일별_ρ": round(ρ1, 3), "일별_별(교란~GIV)": 별표(p1a), "일별_별(GIV~교란)": 별표(p1b),
            "일별_T": n1, "일별_표6": 표6[이름][1]}


print(f"GIV: {GIV.index.min().date()} ~ {GIV.index.max().date()}, {len(GIV)}일")


#%% [셀 2] 금·브렌트유 선물(Yahoo) + 암호자산 지수(Nasdaq 파일) 받기 ==========
# 금, 브렌트유 선물, 암호자산 지수의 원자료를 모으는 셀입니다. 금과 브렌트는 Yahoo에서 자동으로 받고, 
# 암호자산 지수는 사용자님이 받으신 Nasdaq 파일에서 읽습니다.
# [대체 자료]
#   금       : 논문 출처 "FRED and Bloomberg"(2.1절). FRED 일별 금 가격(LBMA)은 중단되어 2021~2026 없음
#              -> Yahoo GC=F (COMEX 금 선물 근월물)
#   브렌트유 : 퀀트랩_BIS_3.py 는 FRED 현물(DCOILBRENTEU)을 썼는데 표 5와 최댓값이 달랐음
#              (4.89 대 4.85) -> 선물 가격인지 확인하려고 Yahoo BZ=F (브렌트유 선물 근월물)
#   BGCI     : Bloomberg Galaxy Crypto Index (Bloomberg, 유료)
#              -> Nasdaq Crypto Index (NCI) - 같은 목적(암호자산 시장 전체 지수)의 무료 지수
#              Yahoo ^NCI 는 자료가 없어("possibly delisted") Nasdaq 지수 사이트 파일을 씀 (아래 (나))
# [논문에 명시 없음] 가격 항목 -> 일별 종가(Close) (퀀트랩_BIS_3.py 의 비트코인·이더리움과 같음)
import yfinance as yf

# --- (가) Yahoo: 금, 브렌트유 선물 ---------------------------------------------
야후_대체 = {"금": "GC=F", "브렌트": "BZ=F"}
원수준 = {}
for 이름, 코드 in 야후_대체.items():
    h = yf.Ticker(코드).history(start=받기시작.strftime("%Y-%m-%d"),
                               end=(표본끝 + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                               auto_adjust=False)
    if h.empty:
        print(f"  !! {이름} ({코드}): Yahoo 에서 자료를 받지 못했습니다 -> 이 변수는 이후 셀에서 건너뜀")
        continue
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()   # 시간대 표시 제거 (날짜만 사용)
    s = s[~s.index.duplicated(keep="last")]
    원수준[이름] = s
    print(f"  {이름} ({코드}): {s.notna().sum():,}개, {s.index.min().date()} ~ {s.index.max().date()}")

# --- (나) Nasdaq 지수 사이트 파일: 암호자산 지수 (NCI) ---------------------------------
# 파일 구조 (직접 확인): 'History' 시트, 열 Trade Date / Index Value / Net Change / High / Low
#   Net Change·High·Low 는 전부 0 -> Index Value 만 사용
#   평일만 있음. 실제 값은 2021-04-26 부터 (파일 이름은 20210101 이지만 그 전 값이 없음)
#   -> 이 변수의 상관은 2021-04-26 이후로만 계산됨 (한계)
# 값이 0 인 날 (2021-11-26, 2022-03-25, 2026-09-30): 지수는 0 이 될 수 없으므로 자료 없음으로 처리
#   (0 을 그대로 두면 로그를 계산할 수 없음 - 자료 오류 처리, 결과 판단과 무관)
# [논문에 명시 없음] 일별 값의 기준 시각. NCI 는 24시간 계산되는 지수이고 파일에 시각이 없음.
#   논문의 BGCI 기준 시각도 밝혀져 있지 않음
NCI = pd.read_excel(데이터폴더 / "암호자산지수_NCI_나스닥.xlsx", sheet_name="History")
NCI["Trade Date"] = pd.to_datetime(NCI["Trade Date"])
NCI = NCI.set_index("Trade Date")["Index Value"].astype(float).sort_index()
영인날 = NCI.index[NCI <= 0]
NCI = NCI[NCI > 0]
원수준["암호자산지수"] = NCI
print(f"  암호자산지수 (NCI 파일): {len(NCI):,}개, {NCI.index.min().date()} ~ {NCI.index.max().date()}"
      f" / 값이 0 이라 뺀 날: {[d.date() for d in 영인날]}")

원수준 = pd.DataFrame(원수준)
원수준.index.name = "date"
원수준.to_csv(데이터폴더 / "대체교란변수_원자료_일별.csv", encoding=CSV인코딩)
print("저장: 대체교란변수_원자료_일별.csv")


#%% [셀 3] 평일 달력에 맞추기 + 표 5와 비교 (금·브렌트유) ======================
# [셀 2]의 원자료를 평일 달력에 맞추고, 금·브렌트가 논문과 같은 시계열인지 표 5로 검증하는 셀
# 휴장일 처리 두 가지 (출처: 퀀트랩_BIS_3.py [셀 6]과 같음)
#   "앞값 채움": 직전 거래일 값 / "안 채움": 원자료 그대로
#   근거: 표 5에서 log gold, log Brent 의 T 가 1,367(평일 전체) -> 논문은 휴장일도 채움
#   [논문에 명시 없음] 채운 방법 -> 두 가지 모두 계산
수준 = {"앞값 채움": 원수준.reindex(원수준.index.union(평일_넓게)).sort_index().ffill().reindex(평일_넓게),
        "안 채움": 원수준.reindex(평일_넓게)}

# 브렌트유 현물 (퀀트랩_BIS_3.py [셀 5]에서 받은 FRED DCOILBRENTEU) - 선물과 나란히 비교하기 위함
현물원자료 = pd.read_csv(데이터폴더 / "교란변수_원자료_일별.csv", parse_dates=["date"], index_col="date",
                    encoding=CSV인코딩)["Brent"]
현물 = 현물원자료.reindex(현물원자료.index.union(평일_넓게)).sort_index().ffill().reindex(평일_넓게)

비교행 = []
for 이름, 계열 in [("금 (GC=F 선물)", 수준["앞값 채움"].get("금")),
                  ("브렌트 (BZ=F 선물)", 수준["앞값 채움"].get("브렌트")),
                  ("브렌트 (FRED 현물)", 현물)]:
    if 계열 is None:
        continue
    s = np.log(계열.loc[평일])                              # 표 5는 로그 수준으로 보고
    논문값 = 표5["금"] if 이름.startswith("금") else 표5["브렌트"]
    비교행.append([이름, s.notna().sum(), s.mean(), s.std(), s.min(), s.max(), *논문값])
비교 = pd.DataFrame(비교행, columns=["변수", "T", "평균", "표준편차", "최소", "최대",
                                "표5_T", "표5_평균", "표5_표준편차", "표5_최소", "표5_최대"])
print("[표 5와 비교] 앞값 채움 기준, 로그 수준")
print(비교.round(2).to_string(index=False))
# 암호자산 지수는 표 5에 BGCI 통계가 없어서 비교할 수 없음 (출처: 표 5 - BGCI 행 없음)

#결론
# 브렌트 선물은 표 5와 다섯 값이 모두 같고, 현물은 다릅니다. 금도 평균만 0.01 차이입니다. 
# 그래서 두 변수는 논문과 같은 시계열일 가능성이 높습니다. 이 셀은 검증용으로 출력만 하고 파일은 저장하지 않습니다. 
# 채운 자료는 변수(수준)로 남아 [셀 4]가 바로 씁니다.


#%% [셀 4] 금·브렌트유 선물·암호자산 지수와 GIV 의 상관 =====================
# 금, 브렌트유 선물, 암호자산 지수와 GIV의 상관을 계산해서 표 6과 비교하는 셀
# 계산 방법 (출처: 표 6 주석 - 퀀트랩_BIS_3.py [셀 7]과 같음)
#   가격형 변수는 로그 차분: "log-differences for price-type series (... gold, Brent, ... BGCI ...)"
#   5일 변화 = 5 거래일 전 대비 (출처: 각주 13)
#   표본: 평일 2021-01-04 ~ 2026-03-31 (출처: 표 6 주석 "January 2021 to March 2026")
상관행 = []
for 채움, 표 in 수준.items():
    로그 = np.log(표)
    for 이름 in [c for c in ["금", "브렌트", "암호자산지수"] if c in 표.columns]:
        변화5 = (로그[이름] - 로그[이름].shift(5)).reindex(평일)
        변화1 = (로그[이름] - 로그[이름].shift(1)).reindex(평일)
        상관행.append(상관한줄(이름, 변화5, 변화1, 채움))

상관_야후 = pd.DataFrame(상관행)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 20)
for 채움 in 수준:
    print(f"\n[{채움}]")
    print(상관_야후[상관_야후["휴장일처리"] == 채움].drop(columns="휴장일처리").to_string(index=False))

# [저자 답신 반영 후 결과 - 반영 전과 같음]
# 결론은 "금, 브렌트유, 암호자산 시장의 움직임과 GIV는 거의 무관하다"입니다. 이 결론은 논문 표 6과 같습니다.
# GIV가 원자재 시장(금, 유가)이나 암호자산 시장 전체의 흐름을 따라 움직인다면, 금리에 영향을 주는 다른 경로가 섞여 배제 조건이 의심받게 됩니다. 
# [셀 4]는 그런 경로가 거의 없다는 것을 보여줍니다.

#%% [셀 5] ETF 유입 - 운용사 일별 파일 + CRSP ==================================
# ETF 유입을 만드는 셀입니다. 논문은 유료인 ETF.com 자료를 썼는데, 우리는 무료 운용사 파일과 학교 구독 CRSP로 일별 발행주식수를 구해 유입을 근사합니다. 
# 논문 표 1과 비교해서 제대로 만들어졌는지도 확인

# [대체 자료] 논문: "flows into the seven largest short-term Treasury ETFs from ETF.com" (2.1절)
#   ETF 7개 (출처: 각주 15 "The tickers for the ETFs are BIL, SGOV, SHV, SHY, VGSH, SCHO, and SPTS")
#   ETF.com 은 유료 -> 아래 두 자료로 일별 발행주식수를 구해 유입을 근사
#     운용사 파일 (무료): BIL·SPTS (SPDR NAV History), SGOV (iShares Data Download "Historical" 시트)
#     CRSP (WRDS, 학교 구독): SHV·SHY·VGSH·SCHO
#   이렇게 나눈 이유 (보내주신 파일을 직접 확인한 결과)
#     CRSP 에서 SHV·SHY·VGSH 는 발행주식수가 거의 매일 바뀌지만, BIL·SGOV·SCHO·SPTS 는 5년간
#     56~61번만 바뀜(대략 월 1회). 운용사 파일은 BIL·SPTS·SGOV 의 발행주식수를 매일 제공.
#     SCHO 는 운용사(Schwab)가 일별 발행주식수를 제공하지 않아 CRSP 뿐 (사용자 확인)
#
# 유입 근사식
#   [논문에 명시 없음] ETF.com 의 유입 계산 방법은 논문에 없음.
#   유입(t) = (발행주식수(t) − 발행주식수(t−1)) × 1주당 가치(t)   단위 $B (출처: 표 1 "($B)")
#   1주당 가치: 운용사 파일은 NAV, CRSP 는 NAV 가 없어 종가(DlyPrc) -> 자료마다 있는 값을 씀
#
# 계산 구성 3가지 (출처: 사용자 지시 - 논문에 없는 처리는 가능한 경우를 모두 계산)
#   "7개 (운용사+CRSP)"  : BIL·SPTS·SGOV 운용사 + SHV·SHY·VGSH·SCHO CRSP  (SCHO 만 월 1회 갱신)
#   "6개 (SCHO 제외)"    : 위에서 SCHO 를 뺀 것 (모두 일별 갱신)
#   "7개 (CRSP만)"       : 7개 모두 CRSP (1차본 방식, 비교용)
# 기간: CRSP 가 2025-12-31 에서 끝나므로 세 구성 모두 2021-01-04 ~ 2025-12-31
import xml.etree.ElementTree as ET

# --- (가) CRSP ---------------------------------------------------------------
CRSP = pd.read_csv(데이터폴더 / "ETF_CRSP_일별.csv", parse_dates=["DlyCalDt"])
중복 = CRSP.duplicated().sum()
CRSP = CRSP.drop_duplicates()     # 모든 열이 완전히 같은 행 (SHY 2021-12-01, VGSH 2021-12-23 각 1행) - 지워도 정보 손실 없음
CRSP = CRSP.sort_values(["Ticker", "DlyCalDt"])
CRSP["유입_B"] = CRSP.groupby("Ticker")["ShrOut"].diff() * 1e3 * CRSP["DlyPrc"] / 1e9   # ShrOut 은 천 주 단위 (CRSP 형식)
CRSP유입 = CRSP.pivot(index="DlyCalDt", columns="Ticker", values="유입_B")
CRSP발행 = CRSP.pivot(index="DlyCalDt", columns="Ticker", values="ShrOut") * 1e3
CRSP끝 = CRSP["DlyCalDt"].max()
print(f"CRSP: 완전 중복 행 {중복}개 제거, 마지막 날 {CRSP끝.date()}")

# --- (나) SPDR (BIL, SPTS) - 파일 구조: 시트 'navhist', 4번째 줄이 머리글(Date, NAV, Shares Outstanding, ...)
def SPDR읽기(파일):
    x = pd.read_excel(데이터폴더 / 파일, header=3).iloc[:, :3]
    x["Date"] = pd.to_datetime(x["Date"], format="%d-%b-%Y", errors="coerce")
    x = x.dropna(subset=["Date"]).set_index("Date").sort_index()   # 맨 아래 안내 문구 줄은 날짜가 아니라서 빠짐
    return pd.DataFrame({"NAV": x["NAV"].astype(float), "발행주식수": x["Shares Outstanding"].astype(float)})

# --- (다) iShares (SGOV) - 파일 구조: Excel 2003 XML 형식, 'Historical' 시트
#     (As Of, NAV per Share, Ex-Dividends, Shares Outstanding)
#     파일 전체는 XML 문법 오류가 있는 부분(다른 시트의 링크)이 있어서 'Historical' 시트 부분만 잘라 읽음
def iShares읽기(파일):
    ns = "urn:schemas-microsoft-com:office:spreadsheet"
    raw = (데이터폴더 / 파일).read_text(encoding="utf-8")
    a = raw.index('<ss:Worksheet ss:Name="Historical">')
    b = raw.index("</ss:Worksheet>", a) + len("</ss:Worksheet>")
    ws = ET.fromstring(f'<ss:Workbook xmlns:ss="{ns}">' + raw[a:b] + "</ss:Workbook>")
    행들 = [[(c.find(f"{{{ns}}}Data").text if c.find(f"{{{ns}}}Data") is not None else None)
            for c in r.findall(f"{{{ns}}}Cell")] for r in ws.iter(f"{{{ns}}}Row")]
    x = pd.DataFrame(행들[1:], columns=행들[0])
    x.index = pd.to_datetime(x["As Of"], format="%b %d, %Y")
    # 값이 "--" 인 날은 숫자가 없음 -> 빈 값 (파일에서 2024-03-29(성금요일 휴장), 2024-03-31(일요일) 두 날)
    x = pd.DataFrame({"NAV": pd.to_numeric(x["NAV per Share"], errors="coerce"),
                      "발행주식수": pd.to_numeric(x["Shares Outstanding"], errors="coerce")}).dropna().sort_index()
    return x


운용사 = {"BIL": SPDR읽기("ETF_BIL_운용사.xlsx"), "SPTS": SPDR읽기("ETF_SPTS_운용사.xlsx"),
          "SGOV": iShares읽기("ETF_SGOV_운용사.xls")}
운용사유입 = pd.DataFrame({t: x["발행주식수"].diff() * x["NAV"] / 1e9 for t, x in 운용사.items()})

# --- 점검: 운용사 파일과 CRSP 의 발행주식수가 같은 ETF 인지 (같은 날끼리 비율) ----------
print("\n[점검] 운용사 파일 vs CRSP 발행주식수 (같은 날짜끼리)")
for t, x in 운용사.items():
    j = pd.concat([x["발행주식수"], CRSP발행[t]], axis=1, keys=["운용사", "CRSP"]).dropna()
    j = j.loc[표본시작:CRSP끝]
    변경 = int((x["발행주식수"].loc[표본시작:표본끝].diff() != 0).sum())
    print(f"  {t}: 운용사 {x.index.min().date()}~{x.index.max().date()}, 표본 내 발행주식수 바뀐 날 {변경}일"
          f" / 운용사÷CRSP 비율 중앙값 {(j['운용사'] / j['CRSP']).median():.4f}")
print("  (비율이 1 근처면 같은 ETF. CRSP 는 월 1회 갱신이라 날짜별로는 차이가 남)")

# --- 구성별 일별 유입 합 -------------------------------------------------------
모든유입 = pd.concat([운용사유입.add_suffix("_운용사"), CRSP유입.add_suffix("_CRSP")], axis=1)
구성 = {"7개 (운용사+CRSP)": ["BIL_운용사", "SPTS_운용사", "SGOV_운용사", "SHV_CRSP", "SHY_CRSP", "VGSH_CRSP", "SCHO_CRSP"],
        "6개 (SCHO 제외)": ["BIL_운용사", "SPTS_운용사", "SGOV_운용사", "SHV_CRSP", "SHY_CRSP", "VGSH_CRSP"],
        "7개 (CRSP만)": [f"{t}_CRSP" for t in ["BIL", "SPTS", "SGOV", "SHV", "SHY", "VGSH", "SCHO"]]}
ETF평일 = pd.bdate_range(표본시작, min(표본끝, CRSP끝))

ETF유입 = {}
for 이름, 열 in 구성.items():
    합 = 모든유입[열].sum(axis=1, min_count=len(열))      # 하나라도 없는 날은 빈 값
    # 휴장일 두 가지 (출처: 퀀트랩_BIS_3.py [셀 6]과 같음)
    #   앞값 채움: 발행주식수를 직전 거래일 값으로 채우는 것과 같음 -> 휴장일 유입 = 0
    #   안 채움 : 휴장일은 빈 값. 5일 합은 5일 중 하루라도 빈 값이면 빈 값 (T 가 줄어듦)
    ETF유입[(이름, "앞값 채움")] = 모든유입[열].reindex(평일_넓게).fillna(0.0).sum(axis=1).where(평일_넓게 <= CRSP끝)
    print(f"  [{이름}] 일부 ETF만 빈 평일: {(모든유입[열].isna().any(axis=1) & 모든유입[열].notna().any(axis=1)).reindex(ETF평일, fill_value=False).sum()}일")
    ETF유입[(이름, "안 채움")] = 합.reindex(평일_넓게)
pd.DataFrame({f"{a}_{b}": s for (a, b), s in ETF유입.items()}).rename_axis("date").to_csv(데이터폴더 / "ETF유입_일별.csv", encoding=CSV인코딩)

# 표 1과 비교 (앞값 채움) - 5일 유입 = 5 거래일 유입의 합 (출처: 표 6 주석 "5-day sum")
print("\n[표 1과 비교] 5일 ETF 유입 ($B), 앞값 채움 기준")
행 = []
for 이름 in 구성:
    f5 = ETF유입[(이름, "앞값 채움")].rolling(5).sum().reindex(ETF평일)
    행.append([이름, f5.notna().sum(), f5.mean(), f5.std(), f5.min(), f5.max(), *표1_ETF])
print(pd.DataFrame(행, columns=["구성", "T", "평균", "표준편차", "최소", "최대",
                               "표1_T", "표1_평균", "표1_표준편차", "표1_최소", "표1_최대"]).round(2).to_string(index=False))
print(f"  ※ 본 연구는 {ETF평일[0].date()} ~ {ETF평일[-1].date()} (CRSP 자료 한계), 논문은 2026-03-31 까지")

# 결론은 "무료 운용사 파일과 CRSP를 조합하면 논문이 쓴 유료 ETF.com 자료와 거의 같은 ETF 유입을 만들 수 있다"
# 세 가지 근거
# 표 1과 거의 일치합니다. 7개(운용사+CRSP) 구성의 5일 유입은 평균 0.52, 표준편차 1.54, 최소 −4.12, 최대 9.49입니다. 
#논문 표 1은 0.58, 1.59, −4.21, 9.41입니다. 네 값 모두 가깝습니다.
# 일별 발행주식수가 핵심이었습니다. CRSP만 쓰면 BIL·SGOV·SPTS·SCHO가 월 1회만 갱신돼서 최댓값이 15.22로 튀었습니다. 
# 운용사 일별 파일로 바꾸자 논문과 맞아졌습니다. 유입의 타이밍이 정확해야 논문 자료를 재현할 수 있다는 뜻입니다.
# 자료가 올바른지 교차 확인했습니다. 운용사 파일과 CRSP의 발행주식수 비율이 1.00~1.03이라 같은 ETF가 맞습니다.

#%% [셀 6] ETF 유입과 GIV 의 상관 ============================================
# [셀 5]에서 만든 ETF 유입과 GIV의 상관을 계산해서 표 6과 비교하는 셀
# 계산 방법 (출처: 표 6 주석)
#   "ETF flows are already in daily-flow form, so we use the 5-day sum for the 5-day column
#    and the daily flow for the 1-day column"
# 표본: 2021-01-04 ~ 2025-12-31 (CRSP 자료 한계)
상관행 = []
for (이름, 채움), 유입 in ETF유입.items():
    r = 상관한줄("ETF 유입", 유입.rolling(5).sum().reindex(ETF평일), 유입.reindex(ETF평일), 채움)
    r["변수"] = f"ETF 유입 {이름}"
    상관행.append(r)
상관_ETF = pd.DataFrame(상관행)
print(상관_ETF.to_string(index=False))

# [저자 답신 반영 후 결과 - 반영 전과 같음]
# 결론 **"ETF 유입과 GIV의 상관은 작고, 논문과 같은 방향이다"**입니다.
# 논문 기준을 만족합니다. 모든 구성에서 |ρ| ≤ 0.086으로 0.10 이하입니다.
# 표 1과 맞았던 구성이 표 6과도 맞습니다. 7개(운용사+CRSP)와 6개는 논문처럼 5일·일별 모두 양(+)의 상관이고, 5일 쪽이 논문처럼 통계적으로 유의합니다. 
# 크기는 논문보다 조금 작습니다(0.055~0.086 대 0.097).
# 타이밍이 틀리면 결과도 틀립니다. 표 1과 맞지 않았던 CRSP만 구성은 부호가 **반대(−)**로 나옵니다. 
# [셀 5]에서 운용사 일별 파일로 바꾼 것이 결과를 좌우했다는 뜻이고, 우리 대체 자료가 제대로 만들어졌다는 간접 근거입니다.

#%% [셀 7] OFR 금융스트레스지수와 GIV 의 상관 ================================
# OFR 금융스트레스지수(FSI)와 GIV의 상관을 계산하는 셀입니다. 논문의 Bloomberg US FCI와 Goldman FCI(둘 다 Bloomberg, 유료)를 대신

# [대체 자료] 논문: Bloomberg US FCI, Goldman FCI (2.3절, 표 6) - Bloomberg 유료
#   -> OFR Financial Stress Index (미 재무부 금융조사국 OFR, 무료, 미국 거래일마다 일별 계산)
#   파일 열: Date, OFR FSI(전체), Credit, Equity valuation, Safe assets, Funding, Volatility,
#            United States, Other advanced economies, Emerging markets
# [논문에 명시 없음] 어느 열을 쓸지 -> 두 가지 모두 계산 (출처: 사용자 지시)
#   "OFR FSI"       : 전 세계 금융시장 스트레스 전체
#   "United States" : 그중 미국 시장 기여분 (Bloomberg US FCI 가 미국 지표라서 함께 봄)
# 변화 계산 (출처: 표 6 주석 "arithmetic differences for index- and level-type series
#   (Bloomberg US FCI, Goldman FCI, ...)") -> 금융여건지수와 같은 지수형이므로 단순 차분
# 부호 주의: OFR FSI 는 값이 클수록 스트레스가 큼 (OFR 설명: "positive when stress levels are above
#   average"). 논문의 두 지수와 부호 방향이 같다는 보장이 없어서 상관은 "크기"만 비교
FSI원자료 = pd.read_csv(데이터폴더 / "OFR_금융스트레스지수.csv", parse_dates=["Date"], index_col="Date")
FSI원자료 = FSI원자료[["OFR FSI", "United States"]]
print(f"OFR FSI: {FSI원자료.index.min().date()} ~ {FSI원자료.index.max().date()}")

FSI수준 = {"앞값 채움": FSI원자료.reindex(FSI원자료.index.union(평일_넓게)).sort_index().ffill().reindex(평일_넓게),
           "안 채움": FSI원자료.reindex(평일_넓게)}
상관행 = []
for 채움, 표 in FSI수준.items():
    for 열 in ["OFR FSI", "United States"]:
        변화5 = (표[열] - 표[열].shift(5)).reindex(평일)       # 단순 차분 (표 6 주석)
        변화1 = (표[열] - 표[열].shift(1)).reindex(평일)
        r = 상관한줄("금융스트레스지수", 변화5, 변화1, 채움)
        r["변수"] = f"OFR {열}"
        상관행.append(r)
상관_FSI = pd.DataFrame(상관행)
print(상관_FSI.to_string(index=False))

# [저자 답신 반영 후 결과 - 반영 전과 같음]
# 결론 **"금융시장 전반의 스트레스와 GIV는 사실상 무관하다"**입니다.
# 상관이 극히 작습니다. 모든 경우 |ρ| ≤ 0.016이고, 유의한 것도 없습니다. 논문의 두 지수(최대 0.039)보다도 작습니다.
# 설정과 무관합니다. 전체와 미국, 앞값 채움과 안 채움 모두 0에 가깝습니다.
# 배제 조건에서 중요한 이유: 논문은 금융여건을 위험선호가 암호자산 시장과 국채시장 사이를 오가는 핵심 경로 중 하나로 봅니다
# (2.3절 "the principal channels through which risk sentiment could transmit"). 이 경로로 GIV가 오염되지 않았다는 근거입니다.


#%% [셀 8] 결과 모아 저장 + 요약 ============================================
# [셀 4], [셀 6], [셀 7]의 결과를 하나로 모아 저장하고, 요약과 한계를 출력하는 셀입니다. 새로 계산하는 건 없습니다.
대체상관 = pd.concat([상관_야후, 상관_ETF, 상관_FSI], ignore_index=True)
대체상관.to_csv(데이터폴더 / "배제조건_대체자료_교란변수상관.csv", index=False, encoding=CSV인코딩)

print("[대체 자료 - 교란변수 상관 요약] 앞값 채움 기준")
t = 대체상관[대체상관["휴장일처리"] == "앞값 채움"]
for _, r in t.iterrows():
    print(f"  {r['변수']:<24s} 5일 {r['5일_ρ']:+.3f}{r['5일_별(교란~GIV)']:<3s} (표 6 {r['5일_표6']})"
          f"   일별 {r['일별_ρ']:+.3f}{r['일별_별(교란~GIV)']:<3s} (표 6 {r['일별_표6']})")
print(f"  최대 |ρ|: 5일 {t['5일_ρ'].abs().max():.3f}, 일별 {t['일별_ρ'].abs().max():.3f}"
      f"  (논문 2.3절 기준: 12개 변수 모두 |ρ| ≤ 0.10)")
# 한계 (출처: 사용자 지시 - 구하지 못한 자료와 기간 제약은 한계로 기록)
print("\n[한계]")
print("  MMF 유입 (iMoneyNet): 무료 대체 자료(ICI 주간 MMF 자산)도 구하지 못해 제외")
if "암호자산지수" in 원수준.columns:
    print(f"  암호자산 지수 (NCI): {원수준['암호자산지수'].first_valid_index().date()} 부터만 있어 그 이후로 계산")
print(f"  ETF 유입: CRSP 가 {CRSP끝.date()} 에서 끝나 그때까지만 계산")
print("  OFR FSI: Bloomberg·Goldman 두 지수를 하나의 대체 지수로 봄. 부호 방향이 같다는 보장이 없어 크기만 비교")
print("\n저장: 배제조건_대체자료_교란변수상관.csv")

# [저자 답신 반영 후 결과 - 반영 전과 같음]
# 결론 : 유료 자료를 무료 자료로 대신해도, GIV가 교란변수와 거의 무관하다는 논문의 결론이 그대로 유지된다"
# 모든 변수의 |ρ|가 0.10 이하입니다. 최대는 ETF 6개 구성의 0.086입니다.
# 부호를 비교할 수 있는 4개 변수는 8개 상관이 모두 표 6과 같은 방향입니다.
# 검증 수준이 다른 세 종류가 섞여 있습니다.
# 금·브렌트: 표 5와 사실상 일치해서 논문과 같은 시계열일 가능성이 높습니다.
# ETF: 표 1과 일치해서 논문 자료에 가까운 대체입니다.
# 암호자산 지수, OFR FSI: 검증할 방법이 없는 보조 증거입니다.