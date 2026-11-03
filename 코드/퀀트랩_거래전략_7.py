# -*- coding: utf-8 -*-
"""
퀀트랩_거래전략_7.py
=====================================================================
[연구] 거래전략 연구 (설계: 퀀트랩_거래전략_0.py -> 데이터/거래전략_설계.csv)
[단계] 7단계: 진짜 표본 외 - 판정 E (부호만 기록)

[무엇을 하나]
  2026-04-01 이후 = 논문도 우리 설계도 보지 않은 구간에서 4·5단계를 그대로 이어 감
  ① 예측 (설계 "E_예측구간"): t = 2026-04-01 ~ t+h ≤ 금리 자료 마지막 날
       추정 = 신호 첫날부터 끝날(s+h) ≤ t 인 모든 관측치 (4단계 확장창 규칙 그대로, 2026-03 이전 자료도 씀)
  ② 기록할 부호 (설계 "E_부호항목", "E_진짜표본외" = 부호만 기록, 판정 없음)
       표본 외 R² (기준 = 신호 뺀 모형)                    - 4단계: 음수
       Clark-West t                                       - 4단계: 양수
       전략 타이밍 = 갈아탄 날 평균 초과수익 − 모든 날 평균 - 5단계: −0.01bp (0.666 − 0.675)
  예측 112개 (2026-04-01 ~ 09-21), 13일씩 겹쳐 독립 구간은 약 9개 -> p값·성립 여부는 의미 없어 부호만 남김
  이 코드는 자료가 쌓인 뒤 다시 돌리면 같은 규칙으로 표본만 늘어남

[출처 표기 규칙]  (퀀트랩_거래전략_0.py 와 같음)
  출처: 퀀트랩_BIS_○ / 퀀트랩_거래전략_○   출처: 사용자 확정 (날짜)
  출처: 설계 "항목이름" -> 데이터/거래전략_설계.csv 의 해당 줄

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  거래전략_설계.csv, 거래전략_예측자료_평일.csv, 4단계_TB입찰_원자료.csv,
  확장_시장자료_원자료_일별.csv (1·3·6개월 금리), 거래전략_4단계_표본외.csv, 거래전략_5단계_결과.csv
[출력]
  데이터/거래전략_7단계_진짜표본외.csv
  그림/거래전략_그림8_진짜표본외.png

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install pandas numpy scipy statsmodels matplotlib
=====================================================================
"""

#%% [셀 0] 패키지 + 폴더 + 설계 읽기 =======================================
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm

# 폴더 (출처: 사용자 지시 - 바탕화면/스테이블코인_퀀트랩, 한국어 파일명)
결과폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩"
데이터폴더 = 결과폴더 / "데이터"
그림폴더 = 결과폴더 / "그림"
그림폴더.mkdir(parents=True, exist_ok=True)
CSV인코딩 = "utf-8-sig"

# 설계 읽기 (출처: 퀀트랩_거래전략_0.py)
설계 = pd.read_csv(데이터폴더 / "거래전략_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
for k in ["E_부호항목", "E_예측구간"]:
    assert k in 설계.index, f'설계에 "{k}" 줄이 없습니다. 퀀트랩_거래전략_0.py 를 고쳐 다시 실행하세요.'
진짜시작 = pd.Timestamp(설계["진짜표본외시작"])                           # 2026-04-01
주h = int(설계["주h"])                                                   # 13
NW더하기 = int(re.search(r"\+\s*(\d+)", 설계["NW시차규칙"]).group(1))     # 4
print("데이터 폴더:", 데이터폴더)
print(f"설계: 진짜 표본 외 시작 {진짜시작.date()}, 주h {주h}")


#%% [셀 1] 자료 읽기 (퀀트랩_거래전략_4 [셀 1] + 퀀트랩_거래전략_5 [셀 1]) ==================
자료 = pd.read_csv(데이터폴더 / "거래전략_예측자료_평일.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
평일 = 자료.index
신호 = 자료["거래신호_GIV5"]
스프레드 = 자료["스프레드_bp"]
FOMC = 자료["FOMC성명일"]
자료끝 = 스프레드.last_valid_index()                                     # 금리 자료 마지막 날 (설계 "E_예측구간")

입찰 = pd.read_csv(데이터폴더 / "4단계_TB입찰_원자료.csv", encoding=CSV인코딩, dtype=str)
if "cash_management_bill_cmb" in 입찰.columns:
    입찰 = 입찰[입찰["cash_management_bill_cmb"] != "Yes"]
발표열 = [c for c in 입찰.columns if "announc" in c.lower() and c.lower().endswith("_date")]   # 날짜 열만
if not 발표열 or "offering_amt" not in 입찰.columns:
    raise ValueError(f"발표일 또는 제시금액 열이 없습니다. 열 목록을 알려주세요: {list(입찰.columns)}")
입찰 = pd.DataFrame({"발표일": pd.to_datetime(입찰[발표열[0]], errors="coerce"),
                     "발행일": pd.to_datetime(입찰["issue_date"], errors="coerce"),
                     "금액": pd.to_numeric(입찰["offering_amt"], errors="coerce") / 1e9}).dropna()
assert len(입찰) > 0, "입찰 자료가 0건입니다. 발표일·발행일·금액 열을 확인하세요."

시장 = pd.read_csv(데이터폴더 / "확장_시장자료_원자료_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
금리 = pd.DataFrame({1 / 12: 시장["1개월"], 0.25: 시장["3개월"], 0.5: 시장["6개월"]}).reindex(평일) / 100

# 통제변수 자료가 결과 기간 끝까지 덮는지 확인 (덮지 못하면 FOMC·입찰이 0 으로 잘못 들어감)
print(f"금리 자료 끝 {자료끝.date()}, 입찰 발행일 끝 {입찰['발행일'].max().date()}, "
      f"FOMC 성명일 자료 끝 {FOMC[FOMC > 0].index.max().date()}")
print("  [확인] FOMC 자료 끝 뒤 ~ 금리 자료 끝 사이에 FOMC 회의가 있었다면 그 회의는 0 으로 들어감")


#%% [셀 2] 예측 자료 + 확장창 예측 (퀀트랩_거래전략_4 와 같은 정의, 끝 = 자료끝) ============
날열 = pd.Series(평일, index=평일)
통제 = ["지난5일", "FOMC", "입찰"]


def 예측자료(h):
    x = 스프레드
    끝날 = 날열.shift(-h)
    d = pd.DataFrame({"y": x.shift(-h) - x, "신호": 신호, "지난5일": x - x.shift(5)}, index=평일)
    d["FOMC"] = (sum(FOMC.shift(-k) for k in range(1, h + 1)) > 0).astype(float)
    d["입찰"] = [입찰.loc[(입찰["발표일"] <= t) & (입찰["발행일"] > t) & (입찰["발행일"] <= e), "금액"].sum()
                 if pd.notna(e) else np.nan for t, e in zip(평일, 끝날)]
    d["끝날"] = 끝날
    d = d[끝날 <= 자료끝]                                     # 설계 "E_예측구간": t+h ≤ 금리 자료 마지막 날
    return d.dropna()


def 맞추고예측(학습, 열, 오늘):
    X = np.column_stack([np.ones(len(학습))] + [학습[c].values for c in 열])
    b = np.linalg.lstsq(X, 학습["y"].values, rcond=None)[0]
    return float(np.r_[1.0, 오늘[열].values] @ b)


d = 예측자료(주h)
행 = []
for t in d.index[d.index >= 진짜시작]:
    학습 = d[d["끝날"] <= t]                                   # t일에 결과가 알려진 것만 (설계 "표본외_자료규칙")
    행.append({"date": t, "판날": d.at[t, "끝날"], "실제": d.at[t, "y"], "학습관측치": len(학습),
              "예측_큰": 맞추고예측(학습, ["신호"] + 통제, d.loc[t]),
              "예측_기준": 맞추고예측(학습, 통제, d.loc[t])})
예측 = pd.DataFrame(행).set_index("date")
print(f"진짜 표본 외 예측 {len(예측)}개 ({예측.index.min().date()} ~ {예측.index.max().date()}), "
      f"학습 관측치 {예측['학습관측치'].iloc[0]} -> {예측['학습관측치'].iloc[-1]}")


#%% [셀 3] 부호 ① 표본 외 R², ② Clark-West t (퀀트랩_거래전략_4 [셀 3] 과 같은 계산) ==========
e큰, e기 = 예측["실제"] - 예측["예측_큰"], 예측["실제"] - 예측["예측_기준"]
R2 = 1 - (e큰 ** 2).sum() / (e기 ** 2).sum()
f = e기 ** 2 - (e큰 ** 2 - (예측["예측_기준"] - 예측["예측_큰"]) ** 2)
CW = sm.OLS(f.values, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": 주h + NW더하기}).tvalues[0]


#%% [셀 4] 부호 ③ 전략 타이밍 (퀀트랩_거래전략_5 [셀 2]·[셀 3] 과 같은 계산) =================
def 보간금리(날, 만기):
    줄 = 금리.loc[날].dropna()
    return np.nan if 줄.empty else float(np.interp(만기, 줄.index.values, 줄.values))


def 보유수익률(산날, 판날, n):
    """로그 보유수익률 (설계 "수익률계산", Campbell, Lo & MacKinlay 1997 10장)"""
    τ = (판날 - 산날).days / 365
    return n * 금리.at[산날, n] - (n - τ) * 보간금리(판날, n - τ)


초과 = []
for t, 줄 in 예측.iterrows():
    빔 = 금리.loc[t, [0.25, 0.5]].isna().any() or 금리.loc[줄["판날"]].isna().all()   # 설계 "금리빈날"
    초과.append(np.nan if 빔 else (보유수익률(t, 줄["판날"], 0.25) - 보유수익률(t, 줄["판날"], 0.5)) * 1e4)
예측["초과_bp"] = 초과
예측["갈아탐"] = (예측["예측_큰"] < 0) & 예측["초과_bp"].notna()            # 설계 "전략규칙"
유효 = 예측["초과_bp"].notna()
타이밍 = 예측.loc[예측["갈아탐"], "초과_bp"].mean() - 예측.loc[유효, "초과_bp"].mean()


#%% [셀 5] 판정 E: 부호 기록 + 저장 =============================================
# 비교 대상: 4단계(2025-01 ~ 2026-03) 표본 외, 5단계 백테스트
옛4 = pd.read_csv(데이터폴더 / "거래전략_4단계_표본외.csv", encoding=CSV인코딩)
옛4 = 옛4[(옛4["결과변수"] == "주(스프레드)") & (옛4["h"] == 주h)].iloc[0]
옛5 = pd.read_csv(데이터폴더 / "거래전략_5단계_결과.csv", encoding=CSV인코딩).iloc[0]
옛타이밍 = None
try:                                                         # 5단계 거래 파일에서 같은 방식으로 다시 계산
    거래5 = pd.read_csv(데이터폴더 / "거래전략_5단계_거래.csv", encoding=CSV인코딩)
    거래5 = 거래5[~거래5["금리빔"]]
    옛타이밍 = 거래5.loc[거래5["갈아탐"], "초과_bp"].mean() - 거래5["초과_bp"].mean()
except (FileNotFoundError, KeyError):
    pass


def 부호(v):
    return "+" if v > 0 else ("−" if v < 0 else "0")


기록 = pd.DataFrame([
    {"항목": "표본 외 R² (기준 = 신호 뺀 모형)", "진짜표본외": R2, "부호": 부호(R2),
     "4·5단계": 옛4["R2_신호뺀모형"], "앞부호": 부호(옛4["R2_신호뺀모형"])},
    {"항목": "Clark-West t", "진짜표본외": CW, "부호": 부호(CW),
     "4·5단계": 옛4["CW_t_신호뺀모형"], "앞부호": 부호(옛4["CW_t_신호뺀모형"])},
    {"항목": "전략 타이밍 (갈아탄 날 − 모든 날, bp)", "진짜표본외": 타이밍, "부호": 부호(타이밍),
     "4·5단계": 옛타이밍, "앞부호": 부호(옛타이밍) if 옛타이밍 is not None else "-"}])
기록["같은방향"] = 기록["부호"] == 기록["앞부호"]
기록.insert(0, "예측수", len(예측))
기록.insert(1, "기간", f"{예측.index.min().date()} ~ {예측['판날'].max().date()}")
기록.to_csv(데이터폴더 / "거래전략_7단계_진짜표본외.csv", index=False, encoding=CSV인코딩)

print(f"[판정 E] 진짜 표본 외 {예측.index.min().date()} ~ {예측['판날'].max().date()}, 예측 {len(예측)}개, "
      f"갈아탄 날 {int(예측['갈아탐'].sum())}일 - 부호만 기록 (판정 없음, 설계 \"E_진짜표본외\")")
for _, 줄 in 기록.iterrows():
    앞 = f"{줄['4·5단계']:+.3f}" if pd.notna(줄["4·5단계"]) else "없음"
    print(f"  {줄['항목']}: {줄['진짜표본외']:+.3f} ({줄['부호']})  |  4·5단계 {앞} ({줄['앞부호']})  "
          f"-> {'같은 방향' if 줄['같은방향'] else '다른 방향'}")
print("저장: 거래전략_7단계_진짜표본외.csv")


#%% [셀 6] 그림 ==============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
누적 = (e기 ** 2 - e큰 ** 2).cumsum()                     # 4단계 그림5 왼쪽과 같은 방식 (Goyal & Welch 2008)
axes[0].plot(누적.index, 누적.values, color="#2171b5", lw=1)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title(f"진짜 표본 외 h = {주h}: 누적 (기준 오차² - 큰 모형 오차²)", loc="left", fontsize=10)
axes[0].set_ylabel("bp²")
axes[1].scatter(예측["예측_큰"], 예측["실제"], s=6, color="#2171b5")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("예측 스프레드 변화 (bp, 큰 모형)")
axes[1].set_ylabel("실제 스프레드 변화 (bp)")
axes[1].set_title("예측 vs 실제", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(그림폴더 / "거래전략_그림8_진짜표본외.png", dpi=150)
plt.close(fig)
print("저장: 그림/거래전략_그림8_진짜표본외.png")
