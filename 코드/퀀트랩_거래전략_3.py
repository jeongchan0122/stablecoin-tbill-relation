# -*- coding: utf-8 -*-
"""
퀀트랩_거래전략_3.py
=====================================================================
[연구] 거래전략 연구 (설계: 퀀트랩_거래전략_0.py -> 데이터/거래전략_설계.csv)
[단계] 3단계: 표본 내 예측 회귀 - 판정 B (참고용)

[무엇을 하나]
  2단계에서 표준오차만 꺼냈던 같은 회귀에서 이번에는 계수(b)를 봄
    (t → t+h 결과변수 변화) = 상수 + b × 신호_t + 지난5일 + FOMC + 입찰,  Newey-West 시차 h + 4
  판정 B (설계 "B_표본내"): b < 0 이고 5% 유의
    -> 참고용. 논문이 효과를 찾은 기간과 겹치는 표본이라 여기서 좋게 나와도 주 결론으로 쓰지 않음
       주 판정은 4단계(표본 외, C)와 5단계(비용, D)

[2단계 판정 A 결과 (2026-10-15)] 검정력 70.8% -> 조건부 진행
  -> 유의하지 않으면: "신호 1SD 당 약 1.5bp 보다 작은 효과는 판단 불가"로 씀
  -> 유의하면: 검정력이 낮은 표본의 유의한 추정치는 커 보이기 쉬움 (Type M 오차, Gelman & Carlin 2014)

[출처 표기 규칙]  (퀀트랩_거래전략_0.py 와 같음)
  출처: 논문 ○○절 / 각주 ○ / 표 ○      출처: 퀀트랩_BIS_○ / 퀀트랩_거래전략_○
  출처: 사용자 확정 (날짜)               출처: 설계 "항목이름" -> 데이터/거래전략_설계.csv 의 해당 줄

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  거래전략_설계.csv                (퀀트랩_거래전략_0.py)
  거래전략_예측자료_평일.csv        (퀀트랩_거래전략_2 [셀 7]) - 신호, 스프레드, 3개월 금리, FOMC 성명일
  4단계_TB입찰_원자료.csv           (퀀트랩_BIS_4 [셀 3])     - 입찰 통제
  거래전략_2단계_검정력.csv         (퀀트랩_거래전략_2 [셀 5]) - 같은 회귀인지 확인 + 가정 효과·MDE 비교
[출력]
  데이터/거래전략_3단계_표본내.csv
  그림/거래전략_그림4_표본내계수.png

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
표본끝 = pd.Timestamp(설계["표본끝"])                                    # 2026-03-31
주h = int(설계["주h"])                                                   # 13
참고h = [int(x) for x in 설계["참고h"].split(",")]                       # 10 ~ 25
NW더하기 = int(re.search(r"\+\s*(\d+)", 설계["NW시차규칙"]).group(1))     # "h + 4" 에서 4
유의수준 = float(설계["유의수준"])                                        # 0.05
z = norm.ppf(1 - 유의수준 / 2)                                            # 1.96
print("데이터 폴더:", 데이터폴더)
print(f"설계: 주h {주h}, NW 시차 h + {NW더하기}, 양측 {유의수준:.0%}, 표본끝 {표본끝.date()}")


#%% [셀 1] 자료 읽기 ===========================================================
# 신호·금리·FOMC 성명일 (출처: 퀀트랩_거래전략_2 [셀 7] 이 저장한 평일 자료 - 같은 재료를 그대로 씀)
자료 = pd.read_csv(데이터폴더 / "거래전략_예측자료_평일.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
평일 = 자료.index
신호 = 자료["거래신호_GIV5"]
금리 = pd.DataFrame({"스프레드": 자료["스프레드_bp"], "3개월": 자료["3개월_bp"]})
FOMC = 자료["FOMC성명일"]

# 3개월 T-bill 입찰 (출처: 퀀트랩_거래전략_2 [셀 2] 와 같은 읽기 - 13주물, CMB 제외)
입찰 = pd.read_csv(데이터폴더 / "4단계_TB입찰_원자료.csv", encoding=CSV인코딩, dtype=str)
if "cash_management_bill_cmb" in 입찰.columns:
    입찰 = 입찰[입찰["cash_management_bill_cmb"] != "Yes"]
발표열 = [c for c in 입찰.columns if "announc" in c.lower() and c.lower().endswith("_date")]   # 날짜 열만
if not 발표열 or "offering_amt" not in 입찰.columns:
    raise ValueError(f"발표일 또는 제시금액 열이 없습니다. 열 목록을 알려주세요: {list(입찰.columns)}")
입찰 = pd.DataFrame({"발표일": pd.to_datetime(입찰[발표열[0]], errors="coerce"),
                     "발행일": pd.to_datetime(입찰["issue_date"], errors="coerce"),
                     "금액": pd.to_numeric(입찰["offering_amt"], errors="coerce") / 1e9}).dropna()   # 달러 -> $B
assert len(입찰) > 0, "입찰 자료가 0건입니다. 발표일·발행일·금액 열을 확인하세요."
print(f"평일 자료 {평일.min().date()} ~ {평일.max().date()}, 입찰 {len(입찰)}건 (발표일 열 '{발표열[0]}')")


#%% [셀 2] 예측 자료 + 회귀 함수 (퀀트랩_거래전략_2 [셀 3] 과 같은 정의) ==========
# 결과 (출처: 설계 "결과기간") y = X_{t+h} − X_t, h 는 평일 수
# 통제 (출처: 설계 "통제변수_정의") 지난5일 = X_t − X_{t−5}, FOMC = (t, t+h] 성명일, 입찰 = 발표 ≤ t·발행 (t, t+h]
# 표본 (출처: 설계 "검정력_표본") t 는 신호 첫날부터, t+h ≤ 표본끝
결과열 = {"주": "스프레드", "보조": "3개월"}       # 출처: 설계 "주결과변수", "보조결과변수"
날열 = pd.Series(평일, index=평일)


def 예측자료(h, 결과):
    x = 금리[결과]
    끝날 = 날열.shift(-h)
    d = pd.DataFrame({"y": x.shift(-h) - x, "신호": 신호, "지난5일": x - x.shift(5)}, index=평일)
    d["FOMC"] = (sum(FOMC.shift(-k) for k in range(1, h + 1)) > 0).astype(float)
    d["입찰"] = [입찰.loc[(입찰["발표일"] <= t) & (입찰["발행일"] > t) & (입찰["발행일"] <= e), "금액"].sum()
                 if pd.notna(e) else np.nan for t, e in zip(평일, 끝날)]
    d = d[끝날 <= 표본끝]
    return d.dropna()


def 회귀(d, h):
    """예측 회귀 (Newey-West 시차 h + 4, 출처: 설계 "NW시차규칙"). 이번 단계부터 계수를 꺼냄"""
    X = sm.add_constant(d[["신호", "지난5일", "FOMC", "입찰"]])
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + NW더하기})
    b, se = r.params["신호"], r.bse["신호"]
    return {"계수_bp": b, "SE_bp": se, "p": 2 * norm.sf(abs(b / se)), "관측치": int(r.nobs), "R2": r.rsquared}


#%% [셀 3] 판정 B (주결과변수, h = 13 하나로만) ==================================
d13 = 예측자료(주h, 결과열["주"])
신호SD = d13["신호"].std()                     # 1SD 환산용 (퀀트랩_거래전략_2 [셀 4] 와 같음)
주 = 회귀(d13, 주h)

# 확인: 2단계와 같은 회귀인지 (표준오차가 같아야 함)
검정력표 = pd.read_csv(데이터폴더 / "거래전략_2단계_검정력.csv", encoding=CSV인코딩)
이단계 = 검정력표[(검정력표["결과변수"] == f"주({결과열['주']})") & (검정력표["h"] == 주h)].iloc[0]
assert np.isclose(주["SE_bp"], 이단계["SE_bp"]), "2단계와 표준오차가 다릅니다 (자료나 정의가 바뀌었는지 확인)"
print(f"확인: 2단계와 같은 표준오차 {주['SE_bp']:.3f} -> 같은 회귀")

판정B = (주["계수_bp"] < 0) and (주["p"] < 유의수준)                    # 출처: 설계 "B_표본내"
print(f"\n[판정 B] 주결과변수 스프레드, h = {주h} (표본 {d13.index.min().date()} ~ {d13.index.max().date()}, 관측치 {주['관측치']})")
print(f"  계수 {주['계수_bp']:+.3f} bp / GIV 1 단위 = {주['계수_bp'] * 신호SD:+.2f} bp / 신호 1 SD, "
      f"SE {주['SE_bp']:.3f}, p {주['p']:.3f}, R² {주['R2']:.3f}")
print(f"  -> {'성립 (b < 0, 5% 유의)' if 판정B else '성립 안 함'}  [참고용 - 주 결론에 쓰지 않음]")
print(f"  비교: 가정 효과 {이단계['가정효과_1SD_bp']:+.2f} bp / 1 SD, MDE {이단계['MDE_1SD_bp']:.2f} bp / 1 SD (2단계)")
if 판정B:
    print(f"  [해석 주의] 검정력 {이단계['검정력']:.1%} 표본에서 유의 -> 크기는 실제보다 커 보일 수 있음 (Type M, Gelman & Carlin 2014)")
else:
    print(f"  [해석] '효과 없음'이 아니라 '신호 1SD 당 약 {이단계['MDE_1SD_bp']:.1f}bp 보다 작은 효과는 판단 불가' (2단계 조건부 진행)")


#%% [셀 4] 참고: h = 10 ~ 25, 주·보조 결과변수 (판정에 안 씀) ======================
# 출처: 설계 "참고h" - 그림으로만 보여 주고 판정에는 안 씀
행 = []
for 이름, 결과 in 결과열.items():
    for h in 참고h:
        d = d13 if (h == 주h and 이름 == "주") else 예측자료(h, 결과)
        r = 회귀(d, h)
        행.append({"결과변수": f"{이름}({결과})", "h": h, **r,
                  "계수_1SD_bp": r["계수_bp"] * 신호SD, "SE_1SD_bp": r["SE_bp"] * 신호SD})
표본내 = pd.DataFrame(행)
표본내.to_csv(데이터폴더 / "거래전략_3단계_표본내.csv", index=False, encoding=CSV인코딩)

pd.set_option("display.width", 200)
print("[참고] 표본 내 계수, 신호 1SD 당 bp (판정은 [셀 3] 의 주결과변수 h = 13 하나)")
보일 = 표본내[표본내["h"].isin([10, 13, 20, 25])]
print(보일[["결과변수", "h", "관측치", "계수_1SD_bp", "SE_1SD_bp", "p"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("저장: 거래전략_3단계_표본내.csv")


#%% [셀 5] 그림 ==============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
가정 = 검정력표.set_index(["결과변수", "h"])["가정효과_1SD_bp"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (이름, 묶음) in zip(axes, 표본내.groupby("결과변수", sort=False)):
    ax.fill_between(묶음["h"], 묶음["계수_1SD_bp"] - z * 묶음["SE_1SD_bp"],
                    묶음["계수_1SD_bp"] + z * 묶음["SE_1SD_bp"], color="#2171b5", alpha=0.2, label="95% 구간")
    ax.plot(묶음["h"], 묶음["계수_1SD_bp"], marker="o", ms=3, color="#2171b5", label="표본 내 계수")
    ax.plot(묶음["h"], [가정.loc[(이름, h)] for h in 묶음["h"]], color="#e6550d", ls="--", lw=1,
            label="가정 효과 (2단계)")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(주h, color="#bbbbbb", lw=0.8)
    ax.set_title(이름, loc="left", fontsize=10)
    ax.set_xlabel("h (평일)")
axes[0].set_ylabel("신호 1SD 당 h일 변화 (bp)")
axes[0].legend(fontsize=8)
fig.suptitle(f"표본 내 예측 계수 ({d13.index.min().date()} ~, t+h ≤ {표본끝.date()}, NW 시차 h + {NW더하기})",
             fontsize=10)
fig.tight_layout()
fig.savefig(그림폴더 / "거래전략_그림4_표본내계수.png", dpi=150)
plt.close(fig)
print("저장: 그림/거래전략_그림4_표본내계수.png")
