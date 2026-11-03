# -*- coding: utf-8 -*-
"""
퀀트랩_거래전략_2.py
=====================================================================
[연구] 거래전략 연구 (설계: 퀀트랩_거래전략_0.py -> 데이터/거래전략_설계.csv)
[단계] 2단계: 검정력 점검 - "효과가 논문만큼 있다면, 우리 표본으로 잡아낼 수 있는가"

[왜 하나]
  표본이 짧으면 효과가 있어도 "유의하지 않음"이 나올 수 있음.
  결과(계수)를 보기 전에 "잡아낼 확률"을 먼저 계산해 두면,
  나중에 유의하지 않게 나왔을 때 "효과가 없다"인지 "표본이 작아 모른다"인지 구분할 수 있음.

[이 코드는 예측 회귀의 계수를 화면에도 파일에도 남기지 않습니다]  (출처: 설계 "A_검정력")
  검정력 계산에 필요한 것은 표준오차(SE)뿐 -> 회귀는 돌리지만 SE 만 꺼냄

[계산 방법]
  ① 예측 회귀: (t → t+h 결과변수 변화) = 상수 + b × 신호_t + 통제변수, Newey-West 시차 h + 4
               -> b 의 표준오차 SE 만 꺼냄
  ② 가정한 효과 δ(h) = "논문(우리 재현) 효과가 그대로 있다면 b 가 얼마일까"
       = 재현 IV 계수 (퀀트랩_BIS_4, 3개월 금리, $1B 당 %) × 100 (-> bp)
         × 실시간 1단계 계수 (퀀트랩_거래전략_1, GIV 1 단위당 유입 $B)
       단위: GIV 5일 합 1 단위당 bp
  ③ 검정력 = Φ(|δ|/SE − 1.96) + Φ(−|δ|/SE − 1.96)   (양측 5% z 검정의 검정력)
     최소 검출 효과(MDE) = 2.8 × SE                    (Bloom 1995)
  ④ 판정 (설계 "A_검정력"): 80% 이상 진행 / 50~80% 조건부 진행 / 50% 미만 보류
     판정은 주결과변수(스프레드), 주h(13) 하나로만 함

[출처 표기 규칙]  (퀀트랩_거래전략_0.py 와 같음)
  출처: 논문 ○○절 / 각주 ○ / 표 ○      출처: 저자 답신 (2026-10-12)
  출처: 퀀트랩_BIS_○ / 퀀트랩_거래전략_○  출처: 사용자 확정 (날짜)
  출처: 설계 "항목이름"                   -> 데이터/거래전략_설계.csv 의 해당 줄

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  거래전략_설계.csv                (퀀트랩_거래전략_0.py)
  거래전략_신호_평일.csv            (퀀트랩_거래전략_1 [셀 5]) - 거래 신호
  거래전략_1단계_점검.csv           (퀀트랩_거래전략_1 [셀 7]) - 실시간 1단계 계수
  4단계_그림7_결과.csv              (퀀트랩_BIS_4 [셀 7])     - 재현 IV 계수 (효과 크기 δ 용)
  확장_시장자료_원자료_일별.csv     (퀀트랩_BIS_4-1 [셀 2])   - DGS3MO, DGS6MO
  4단계_TB입찰_원자료.csv           (퀀트랩_BIS_4 [셀 3])     - 3개월 T-bill 입찰 (발표일·발행일·제시금액)
  확장_통화정책충격_USMPD.xlsx      (퀀트랩_BIS_4-1 에서 쓴 파일) - FOMC 성명 날짜
[출력]
  데이터/거래전략_예측자료_평일.csv  (신호, 스프레드, 3개월 금리, FOMC 성명일 - 3단계 이후에도 씀)
  데이터/거래전략_2단계_검정력.csv
  그림/거래전략_그림3_검정력.png

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install pandas numpy scipy statsmodels matplotlib openpyxl
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
# 2단계 세부 정의 4줄이 설계에 들어 있어야 함 (없으면 퀀트랩_거래전략_0.py 에 붙여 넣고 다시 실행)
for k in ["결과기간", "통제변수_정의", "검정력_가정효과", "검정력_표본"]:
    assert k in 설계.index, f'설계에 "{k}" 줄이 없습니다. 퀀트랩_거래전략_0.py 를 고쳐 다시 실행하세요.'
def 날짜(k): return pd.Timestamp(설계[k])
def 정수(k): return int(설계[k])
def 실수(k): return float(설계[k])

표본끝 = 날짜("표본끝")                        # 2026-03-31
주h = 정수("주h")                              # 13
참고h = [int(x) for x in 설계["참고h"].split(",")]   # 10 ~ 25
NW더하기 = int(re.search(r"\+\s*(\d+)", 설계["NW시차규칙"]).group(1))   # "h + 4" 에서 4
유의수준 = 실수("유의수준")                    # 0.05
MDE배수 = 실수("MDE배수")                      # 2.8
z = norm.ppf(1 - 유의수준 / 2)                 # 1.96 (양측 5%)
print("데이터 폴더:", 데이터폴더)
print(f"설계: 주h {주h}, 참고h {참고h[0]}~{참고h[-1]}, NW 시차 h + {NW더하기}, 양측 {유의수준:.0%} (z = {z:.2f})")


#%% [셀 1] 신호와 금리 읽기 → 평일 표 ======================================
# 신호 = t일 거래에 쓰는 실시간 GIV 5일 합 (출처: 퀀트랩_거래전략_1 [셀 4], 설계 "신호지연_달력일")
신호 = pd.read_csv(데이터폴더 / "거래전략_신호_평일.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)["거래신호_GIV5"]

# 금리 (출처: 설계 "금리시리즈" = DGS3MO, DGS6MO / 파일은 퀀트랩_BIS_4-1 [셀 2] 에서 받은 것)
#   FRED 단위는 % -> × 100 = bp
#   휴장일은 채우지 않음 (출처: 퀀트랩_BIS_4 [셀 1] - 금리는 앞값 채움 안 함)
시장 = pd.read_csv(데이터폴더 / "확장_시장자료_원자료_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
평일 = pd.bdate_range(신호.index.min(), 시장.index.max())       # 평일 달력 (출처: 퀀트랩_BIS_2 [셀 0])
금리 = pd.DataFrame({"3개월": 시장["3개월"], "6개월": 시장["6개월"]}).reindex(평일) * 100
금리["스프레드"] = 금리["3개월"] - 금리["6개월"]                 # 출처: 설계 "주결과변수" (3개월 − 6개월)

print(f"신호: {신호.first_valid_index().date()} ~ {신호.last_valid_index().date()}")
print(f"금리: {금리['스프레드'].first_valid_index().date()} ~ {금리['스프레드'].last_valid_index().date()}, "
      f"휴장 등으로 빈 평일 {int(금리['스프레드'].isna().sum())}일")


#%% [셀 2] 통제변수 재료: FOMC 성명일, 3개월 T-bill 입찰 ======================
# 정의 (출처: 설계 "통제변수", "통제변수_정의")
#   FOMC : (t, t+h] 안에 FOMC 성명일이 있으면 1  - 회의 일정은 미리 공개되므로 t일에 알 수 있음
#   입찰 : 발표일 ≤ t 이고 발행일이 (t, t+h] 인 13주물 제시금액 합 ($B) - t일에 이미 발표된 공급
# FOMC 성명일 (출처: 퀀트랩_BIS_4 [셀 4] - SF 연준 USMPD "Statements" 시트, 퀀트랩_BIS_4-1 과 같은 파일)
U = pd.read_excel(데이터폴더 / "확장_통화정책충격_USMPD.xlsx", sheet_name="Statements")
FOMC날 = pd.DatetimeIndex(pd.to_datetime(U["Date"], errors="coerce").dropna().dt.normalize().unique())
FOMC = pd.Series(평일.isin(FOMC날).astype(float), index=평일, name="FOMC성명일")

# 3개월 T-bill 입찰 (출처: 퀀트랩_BIS_4 [셀 3] - Fiscal Data, 13주물, CMB 제외)
입찰 = pd.read_csv(데이터폴더 / "4단계_TB입찰_원자료.csv", encoding=CSV인코딩, dtype=str)
if "cash_management_bill_cmb" in 입찰.columns:
    입찰 = 입찰[입찰["cash_management_bill_cmb"] != "Yes"]
발표열 = [c for c in 입찰.columns if "announc" in c.lower() and c.lower().endswith("_date")]   # 날짜 열만 (cusip 열 제외)
if not 발표열 or "offering_amt" not in 입찰.columns:
    raise ValueError(f"발표일 또는 제시금액 열이 없습니다. 열 목록을 알려주세요: {list(입찰.columns)}")
입찰 = pd.DataFrame({"발표일": pd.to_datetime(입찰[발표열[0]], errors="coerce"),
                     "발행일": pd.to_datetime(입찰["issue_date"], errors="coerce"),
                     "금액": pd.to_numeric(입찰["offering_amt"], errors="coerce") / 1e9}).dropna()   # 달러 -> $B
assert len(입찰) > 0, "입찰 자료가 0건입니다. 발표일·발행일·금액 열을 확인하세요."

print(f"FOMC 성명일: 마지막 {FOMC날.max().date()} (평일 표 안 {int(FOMC.sum())}회)")
print(f"입찰: {len(입찰)}건, 발행일 마지막 {입찰['발행일'].max().date()}, 열 '{발표열[0]}' 을 발표일로 씀")


#%% [셀 3] 예측 자료 만들기 (h 마다) =========================================
# 결과 (출처: 설계 "결과기간") y = X_{t+h} − X_t, h 는 평일 수 (출처: 퀀트랩_BIS_4 [셀 6] - 평일 달력에서 h 칸 뒤)
# 통제 (출처: 설계 "통제변수_정의")
#   지난 5일 변화 = X_t − X_{t−5} (결과변수 자기 자신. 주: 스프레드, 보조: 3개월)
#   FOMC, 입찰 = [셀 2] 정의
# 표본 (출처: 설계 "검정력_표본") t 는 신호 첫날부터, t+h 가 표본끝(2026-03-31) 이하인 날까지
#   (퀀트랩_BIS_4 [셀 6] 처럼 3월 31일 이후 금리를 쓰지 않음)
결과열 = {"주": "스프레드", "보조": "3개월"}       # 출처: 설계 "주결과변수", "보조결과변수"
날열 = pd.Series(평일, index=평일)


def 예측자료(h, 결과):
    x = 금리[결과]
    끝날 = 날열.shift(-h)                                     # t+h 의 날짜
    d = pd.DataFrame({"y": x.shift(-h) - x, "신호": 신호.reindex(평일), "지난5일": x - x.shift(5)}, index=평일)
    # FOMC: (t, t+h] 에 성명일 있으면 1 -> 앞으로 h 칸의 합이 0보다 큰가
    d["FOMC"] = (sum(FOMC.shift(-k) for k in range(1, h + 1)) > 0).astype(float)
    # 입찰: 발표일 ≤ t, t < 발행일 ≤ t+h 인 금액 합
    d["입찰"] = [입찰.loc[(입찰["발표일"] <= t) & (입찰["발행일"] > t) & (입찰["발행일"] <= e), "금액"].sum()
                 if pd.notna(e) else np.nan for t, e in zip(평일, 끝날)]
    d = d[끝날 <= 표본끝]                                     # t+h ≤ 표본끝
    return d.dropna()


def 표준오차만(d, h):
    """예측 회귀 b 의 Newey-West 표준오차만 돌려줌 (계수는 꺼내지 않음, 출처: 설계 "A_검정력")"""
    X = sm.add_constant(d[["신호", "지난5일", "FOMC", "입찰"]])
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + NW더하기})   # 출처: 설계 "NW시차규칙"
    return r.bse["신호"], int(r.nobs)


# 끝 쪽 자료가 표본끝 + h 를 덮는지 확인 (덮지 못하면 FOMC·입찰이 0 으로 잘못 들어감)
필요끝 = 날열[날열 <= 표본끝].iloc[-1] + pd.offsets.BDay(max(참고h))
print(f"필요한 마지막 날(표본끝 + 최대 h): {필요끝.date()}")
print(f"  FOMC 성명일 자료 끝 {FOMC날.max().date()}, 입찰 발행일 끝 {입찰['발행일'].max().date()}"
      " -> 둘 다 이 날 근처 이후여야 함 (USMPD 는 마지막 회의 뒤 다음 회의 전까지는 괜찮음)")

d13 = 예측자료(주h, 결과열["주"])
print(f"h = {주h} 주결과변수 자료: {d13.index.min().date()} ~ {d13.index.max().date()}, 관측치 {len(d13)}")


#%% [셀 4] 가정한 효과 δ(h) ==================================================
# 출처: 설계 "검정력_가정효과"
#   재현 IV 계수 = 퀀트랩_BIS_4 [셀 7] 주 분석(ETF 없음) IV, 3개월 금리, $1B 유입당 %
#     스프레드에도 같은 값을 씀: 논문 3.6절 그림 8 - 스프레드 사양은 기준보다 "slightly larger point estimates"
#     -> 3개월 기준값을 쓰면 효과를 약간 작게 잡는 쪽 (검정력을 부풀리지 않음)
#   실시간 1단계 계수 = 퀀트랩_거래전략_1 [셀 7] 논문 표본 / 실시간 (GIV 1 단위당 유입 $B)
그림7 = pd.read_csv(데이터폴더 / "4단계_그림7_결과.csv", encoding=CSV인코딩)
그림7 = 그림7[(그림7["사양"] == "B 주 분석 (ETF 없음)") & (그림7["방법"] == "IV")].set_index("h")["계수"]
점검1 = pd.read_csv(데이터폴더 / "거래전략_1단계_점검.csv", encoding=CSV인코딩).set_index("점검")["값"]
일단계계수 = 점검1["1단계계수 논문 표본 실시간"]


def δ(h):
    return 그림7.loc[h] * 100 * 일단계계수                    # GIV 1 단위당 bp


신호SD = d13["신호"].std()                                     # 표본 안 신호 표준편차 (1 SD 로 환산용)
print(f"실시간 1단계 계수 {일단계계수:.3f} ($B / GIV 1 단위), 신호 표준편차 {신호SD:.3f}")
print(f"h = {주h}: 재현 IV 효과 {그림7.loc[주h] * 100:+.2f} bp / $1B  ->  "
      f"δ = {δ(주h):+.2f} bp / GIV 1 단위 = {δ(주h) * 신호SD:+.2f} bp / 신호 1 SD")


#%% [셀 5] 검정력 계산 =======================================================
def 검정력(효과, se):
    """양측 z 검정의 검정력 (출처: 표준 정규 근사, Bloom 1995 의 MDE 와 같은 틀)"""
    k = abs(효과) / se
    return norm.cdf(k - z) + norm.cdf(-k - z)


행 = []
for 이름, 결과 in 결과열.items():
    for h in 참고h:
        d = d13 if (h == 주h and 이름 == "주") else 예측자료(h, 결과)
        se, n = 표준오차만(d, h)
        행.append({"결과변수": f"{이름}({결과})", "h": h, "관측치": n, "NW시차": h + NW더하기,
                  "SE_bp": se, "가정효과_bp": δ(h), "검정력": 검정력(δ(h), se),
                  "MDE_bp": MDE배수 * se,
                  "가정효과_1SD_bp": δ(h) * 신호SD, "MDE_1SD_bp": MDE배수 * se * 신호SD})
검정 = pd.DataFrame(행)
검정.to_csv(데이터폴더 / "거래전략_2단계_검정력.csv", index=False, encoding=CSV인코딩)

pd.set_option("display.width", 200)
print("[검정력] 단위 bp, 1SD = 신호 1 표준편차당 (계수는 계산에 쓰지 않음)")
보일 = 검정[검정["h"].isin([10, 13, 20, 25])]
print(보일[["결과변수", "h", "관측치", "SE_bp", "가정효과_1SD_bp", "MDE_1SD_bp", "검정력"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("저장: 거래전략_2단계_검정력.csv")


#%% [셀 6] 판정 A (주결과변수, h = 13 하나로만) ================================
# 출처: 설계 "A_검정력" - 80% 이상 진행 / 50~80% 조건부 진행 / 50% 미만 보류 (Cohen 1988 80% 관례)
주 = 검정[(검정["결과변수"] == f"주({결과열['주']})") & (검정["h"] == 주h)].iloc[0]
p = 주["검정력"]
판정 = "진행" if p >= 0.8 else ("조건부 진행 (검정력 부족을 보고서에 명시)" if p >= 0.5 else "보류 (탐색적 결과로만 다룸)")
print(f"[판정 A] 주결과변수 스프레드, h = {주h}: 검정력 {p:.1%} -> {판정}")
print(f"  가정 효과 {주['가정효과_1SD_bp']:+.2f} bp / 신호 1 SD, 이 표본으로 잡을 수 있는 최소 효과(MDE) "
      f"{주['MDE_1SD_bp']:.2f} bp / 신호 1 SD")


#%% [셀 7] 저장 + 그림 =======================================================
# 3단계 이후에도 쓰는 평일 자료 (h 와 무관한 재료만)
pd.concat([신호.reindex(평일).rename("거래신호_GIV5"), 금리[["스프레드", "3개월", "6개월"]].add_suffix("_bp"), FOMC],
          axis=1).rename_axis("date").to_csv(데이터폴더 / "거래전략_예측자료_평일.csv", encoding=CSV인코딩)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8, 3.8))
for (이름, 묶음), 색 in zip(검정.groupby("결과변수", sort=False), ["#2171b5", "#bbbbbb"]):
    ax.plot(묶음["h"], 묶음["검정력"] * 100, marker="o", ms=3, color=색, label=이름)
ax.axhline(80, color="black", ls="--", lw=0.8, label="80% (진행)")
ax.axhline(50, color="black", ls=":", lw=0.8, label="50% (보류 경계)")
ax.axvline(주h, color="#e6550d", lw=0.8, label=f"판정 h = {주h}")
ax.set_ylim(0, 100)
ax.set_xlabel("h (평일)")
ax.set_ylabel("검정력 (%)")
ax.set_title("검정력: 효과가 재현 결과만큼 있을 때 5% 에서 잡아낼 확률", loc="left", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(그림폴더 / "거래전략_그림3_검정력.png", dpi=150)
plt.close(fig)
print("저장: 거래전략_예측자료_평일.csv, 그림/거래전략_그림3_검정력.png")
