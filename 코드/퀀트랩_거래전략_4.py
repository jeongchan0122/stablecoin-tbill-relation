# -*- coding: utf-8 -*-
"""
퀀트랩_거래전략_4.py
=====================================================================
[연구] 거래전략 연구 (설계: 퀀트랩_거래전략_0.py -> 데이터/거래전략_설계.csv)
[단계] 4단계: 표본 외 예측 - 판정 C (주 판정 ① "시장이 정보를 늦게 반영하는가")

[무엇을 하나]
  "그때 이 신호를 실제로 썼다면 미리 맞혔을까"를 확인
  ① 예측 시점 t 마다, t일에 결과가 이미 알려진 관측치(s + h ≤ t)만으로 회귀를 다시 추정 (확장창)
       (출처: 설계 "표본외_자료규칙", "첫창끝_주")
  ② 두 모형으로 t → t+h 변화를 예측
       큰 모형  : 상수 + 신호 + 지난5일 + FOMC + 입찰   (3단계와 같은 식)
       기준 모형: 상수 +        지난5일 + FOMC + 입찰   (신호만 뺌, 출처: 설계 "기준예측")
  ③ 표본 외 R² = 1 − Σ(실제 − 큰 모형)² / Σ(실제 − 기준 모형)²   (Campbell & Thompson 2008)
     Clark-West 검정 (Clark & West 2007, 내포 모형 비교)
       f = (실제 − 기준)² − [(실제 − 큰)² − (기준 − 큰)²] 의 평균이 0보다 큰가
       표준오차: Newey-West 시차 h + 4 (출처: 설계 "C_표본외", "NW시차규칙"), 한쪽 5% (z > 1.645)
  ④ 판정 C (설계 "C_표본외"): 표본 외 R² > 0 그리고 Clark-West 한쪽 5% 유의
     판정은 주결과변수(스프레드), h = 13 하나로만. "변화 없음(0)" 비교는 참고 출력만 (설계 "기준예측_참고")

[2단계 판정 A] 검정력 70.8% (표본 내 기준) -> 조건부 진행. 표본 외 구간은 그보다 짧아 힘이 더 약함
  -> 성립하지 않으면 "효과 없음"이 아니라 "판단 불가"로 읽음

[출처 표기 규칙]  (퀀트랩_거래전략_0.py 와 같음)
  출처: 논문 ○○절 / 각주 ○ / 표 ○      출처: 퀀트랩_BIS_○ / 퀀트랩_거래전략_○
  출처: 사용자 확정 (날짜)               출처: 설계 "항목이름" -> 데이터/거래전략_설계.csv 의 해당 줄

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  거래전략_설계.csv                (퀀트랩_거래전략_0.py)
  거래전략_예측자료_평일.csv        (퀀트랩_거래전략_2 [셀 7])
  4단계_TB입찰_원자료.csv           (퀀트랩_BIS_4 [셀 3])
[출력]
  데이터/거래전략_4단계_표본외.csv        (h·결과변수별 표본 외 R², Clark-West)
  데이터/거래전략_4단계_예측_평일.csv     (h = 13, 날마다의 실제·예측 - 5단계 백테스트에서 씀)
  그림/거래전략_그림5_표본외.png

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
for k in ["기준예측_참고", "표본외_자료규칙", "표본외_예측구간"]:
    assert k in 설계.index, f'설계에 "{k}" 줄이 없습니다. 퀀트랩_거래전략_0.py 를 고쳐 다시 실행하세요.'
표본끝 = pd.Timestamp(설계["표본끝"])                                    # 2026-03-31
첫창끝 = pd.Timestamp(설계["첫창끝_주"])                                 # 2024-12-31
주h = int(설계["주h"])                                                   # 13
참고h = [int(x) for x in 설계["참고h"].split(",")]                       # 10 ~ 25
NW더하기 = int(re.search(r"\+\s*(\d+)", 설계["NW시차규칙"]).group(1))     # "h + 4" 에서 4
유의수준 = float(설계["유의수준"])                                        # 0.05
z한쪽 = norm.ppf(1 - 유의수준)                                            # 1.645 (한쪽 5%)
print("데이터 폴더:", 데이터폴더)
print(f"설계: 첫 추정창 끝 {첫창끝.date()}, 예측 t+h ≤ {표본끝.date()}, 주h {주h}, CW NW 시차 h + {NW더하기}, 한쪽 {유의수준:.0%}")
print(f"      기준 모형: {설계['기준예측']}")


#%% [셀 1] 자료 읽기 (퀀트랩_거래전략_3 [셀 1] 과 같음) ============================
자료 = pd.read_csv(데이터폴더 / "거래전략_예측자료_평일.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
평일 = 자료.index
신호 = 자료["거래신호_GIV5"]
금리 = pd.DataFrame({"스프레드": 자료["스프레드_bp"], "3개월": 자료["3개월_bp"]})
FOMC = 자료["FOMC성명일"]

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
print(f"평일 자료 {평일.min().date()} ~ {평일.max().date()}, 입찰 {len(입찰)}건")


#%% [셀 2] 예측 자료 (퀀트랩_거래전략_3 [셀 2] 와 같은 정의 + 결과가 끝나는 날 열) =====
결과열 = {"주": "스프레드", "보조": "3개월"}       # 출처: 설계 "주결과변수", "보조결과변수"
통제 = ["지난5일", "FOMC", "입찰"]                 # 출처: 설계 "통제변수_정의"
날열 = pd.Series(평일, index=평일)


def 예측자료(h, 결과):
    x = 금리[결과]
    끝날 = 날열.shift(-h)
    d = pd.DataFrame({"y": x.shift(-h) - x, "신호": 신호, "지난5일": x - x.shift(5)}, index=평일)
    d["FOMC"] = (sum(FOMC.shift(-k) for k in range(1, h + 1)) > 0).astype(float)
    d["입찰"] = [입찰.loc[(입찰["발표일"] <= t) & (입찰["발행일"] > t) & (입찰["발행일"] <= e), "금액"].sum()
                 if pd.notna(e) else np.nan for t, e in zip(평일, 끝날)]
    d["끝날"] = 끝날                                          # 이 관측치의 결과가 알려지는 날 (t+h)
    d = d[끝날 <= 표본끝]
    return d.dropna()


#%% [셀 3] 확장창 표본 외 예측 ==================================================
# 출처: 설계 "표본외_자료규칙" - t일 예측에는 끝날(s+h) ≤ t 인 관측치만 씀, 매일 다시 추정
#       설계 "표본외_예측구간" - t 는 첫창끝 다음 평일부터 (t+h ≤ 표본끝 은 예측자료() 에서 이미 적용)
def 맞추고예측(학습, 열, 오늘):
    """최소제곱 계수를 구해 오늘의 예측값을 돌려줌 (계수는 저장하지 않음)"""
    X = np.column_stack([np.ones(len(학습))] + [학습[c].values for c in 열])
    b = np.linalg.lstsq(X, 학습["y"].values, rcond=None)[0]
    return float(np.r_[1.0, 오늘[열].values] @ b)


def 표본외예측(d):
    행 = []
    for t in d.index[d.index > 첫창끝]:
        학습 = d[d["끝날"] <= t]                               # t일에 결과가 알려진 것만
        행.append({"date": t, "실제": d.at[t, "y"], "학습관측치": len(학습),
                  "예측_큰": 맞추고예측(학습, ["신호"] + 통제, d.loc[t]),
                  "예측_기준": 맞추고예측(학습, 통제, d.loc[t])})
    return pd.DataFrame(행).set_index("date")


def 평가(p, h):
    """표본 외 R² (Campbell & Thompson 2008), Clark-West (Clark & West 2007, NW 시차 h + 4)"""
    out = {"예측수": len(p), "첫예측": p.index.min().date(), "마지막예측": p.index.max().date()}
    for 이름, 기준 in [("신호뺀모형", p["예측_기준"]), ("0참고", pd.Series(0.0, index=p.index))]:
        e큰, e기 = p["실제"] - p["예측_큰"], p["실제"] - 기준
        R2 = 1 - (e큰 ** 2).sum() / (e기 ** 2).sum()
        f = e기 ** 2 - (e큰 ** 2 - (기준 - p["예측_큰"]) ** 2)
        r = sm.OLS(f.values, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": h + NW더하기})
        out[f"R2_{이름}"] = R2
        out[f"CW_t_{이름}"] = r.tvalues[0]
        out[f"CW_p_{이름}"] = norm.sf(r.tvalues[0])            # 한쪽
    return out


예측13 = 표본외예측(예측자료(주h, 결과열["주"]))
print(f"h = {주h} 스프레드: 예측 {len(예측13)}개 ({예측13.index.min().date()} ~ {예측13.index.max().date()}), "
      f"첫 예측의 학습 관측치 {예측13['학습관측치'].iloc[0]}, 마지막 {예측13['학습관측치'].iloc[-1]}")


#%% [셀 4] 판정 C (주결과변수, h = 13 하나로만) ==================================
주 = 평가(예측13, 주h)
판정C = (주["R2_신호뺀모형"] > 0) and (주["CW_p_신호뺀모형"] < 유의수준)    # 출처: 설계 "C_표본외"
print(f"[판정 C] 주결과변수 스프레드, h = {주h}, 예측 {주['예측수']}개 ({주['첫예측']} ~ {주['마지막예측']})")
print(f"  기준 = 신호 뺀 모형: 표본 외 R² {주['R2_신호뺀모형']:+.2%}, Clark-West t {주['CW_t_신호뺀모형']:+.2f}, "
      f"한쪽 p {주['CW_p_신호뺀모형']:.3f}")
print(f"  -> {'성립 (신호를 넣으면 표본 외 예측이 나아짐)' if 판정C else '성립 안 함 (판단 불가 - 효과 없음이 아님)'}")
print(f"  [참고] 기준 = 변화 없음(0): 표본 외 R² {주['R2_0참고']:+.2%}, Clark-West t {주['CW_t_0참고']:+.2f}, "
      f"한쪽 p {주['CW_p_0참고']:.3f}  (판정에 안 씀)")


#%% [셀 5] 참고: h = 10 ~ 25, 주·보조 결과변수 (판정에 안 씀) + 저장 ================
행, 보관 = [], {}
for 이름, 결과 in 결과열.items():
    for h in 참고h:
        p = 예측13 if (h == 주h and 이름 == "주") else 표본외예측(예측자료(h, 결과))
        if h == 주h:
            보관[이름] = p
        행.append({"결과변수": f"{이름}({결과})", "h": h, **평가(p, h)})
표본외 = pd.DataFrame(행)
표본외.to_csv(데이터폴더 / "거래전략_4단계_표본외.csv", index=False, encoding=CSV인코딩)

pd.set_option("display.width", 200)
print("[참고] 표본 외, 기준 = 신호 뺀 모형 (판정은 [셀 4] 의 주결과변수 h = 13 하나)")
보일 = 표본외[표본외["h"].isin([10, 13, 20, 25])]
print(보일[["결과변수", "h", "예측수", "R2_신호뺀모형", "CW_t_신호뺀모형", "CW_p_신호뺀모형", "R2_0참고"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# 5단계 백테스트용: h = 13 날마다 실제·예측 (예측_큰 < 0 이면 3개월물로 바꾸는 규칙, 설계 "전략규칙")
pd.concat({이름: p[["실제", "예측_큰", "예측_기준"]] for 이름, p in 보관.items()}, axis=1) \
    .to_csv(데이터폴더 / "거래전략_4단계_예측_평일.csv", encoding=CSV인코딩)
print("저장: 거래전략_4단계_표본외.csv, 거래전략_4단계_예측_평일.csv")


#%% [셀 6] 그림 ==============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

# 왼쪽: 누적 오차제곱 차이 (Goyal & Welch 2008 의 그림 방식) - 위로 올라가면 신호를 넣은 모형이 더 잘 맞힌 구간
누적 = ((예측13["실제"] - 예측13["예측_기준"]) ** 2 - (예측13["실제"] - 예측13["예측_큰"]) ** 2).cumsum()
axes[0].plot(누적.index, 누적.values, color="#2171b5", lw=1)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title(f"스프레드 h = {주h}: 누적 (기준 오차² - 큰 모형 오차²)", loc="left", fontsize=10)
axes[0].set_ylabel("bp²")

# 오른쪽: h 별 표본 외 R² (기준 = 신호 뺀 모형)
for (이름, 묶음), 색 in zip(표본외.groupby("결과변수", sort=False), ["#2171b5", "#bbbbbb"]):
    axes[1].plot(묶음["h"], 묶음["R2_신호뺀모형"] * 100, marker="o", ms=3, color=색, label=이름)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].axvline(주h, color="#e6550d", lw=0.8, label=f"판정 h = {주h}")
axes[1].set_xlabel("h (평일)")
axes[1].set_ylabel("표본 외 R² (%)")
axes[1].set_title("표본 외 R² (기준 = 신호 뺀 모형)", loc="left", fontsize=10)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(그림폴더 / "거래전략_그림5_표본외.png", dpi=150)
plt.close(fig)
print("저장: 그림/거래전략_그림5_표본외.png")


#%% [셀 7] 보조신호: GIV 가 버린 유입에도 예측 정보가 있는가 (해석용, 판정에 안 씀) ========
# 출처: 설계 "보조신호" (5일 유입을 GIV 부분과 나머지로 분해 - 예측력이 어디서 오는지 해석용, 4단계에서 계산)
#       세부 방법은 설계 "보조신호_방법"
# 왜: GIV 는 인과 효과를 재려고 유입 중 큰 페어의 개별 충격만 남긴 값 (1단계 R² 약 0.29)
#     예측에는 원인일 필요가 없으므로, GIV 가 버린 나머지 유입(공통 흐름)에도 예측 정보가 있을 수 있음
# 유입 신호 = 직전 5 달력일 합산 시가총액 변화 ($B) = MC(t−1) − MC(t−6)
#   GIV 신호와 같은 창(직전 5 달력일)·같은 하루 지연 (출처: 설계 "신호지연_달력일", 퀀트랩_거래전략_1 [셀 4])
# 분해 방법: GIV 와 유입을 한 모형에 같이 넣음 -> 유입의 계수는 "GIV 로 설명되지 않는 나머지 유입"의 효과
#   (Frisch-Waugh 정리. 날마다 따로 분해 계수를 추정할 필요가 없어 미래 정보가 섞이지 않음)
# 비교 (모두 [셀 3] 과 같은 확장창, 끝날 ≤ t 인 관측치만, 표본 외 R² + Clark-West, NW 시차 h + 4)
#   ① GIV  vs 기준(통제만)   - [셀 4] 원래 결과와 같아야 함
#   ② 유입 vs 기준(통제만)   - 유입 전체로 하면?
#   ③ GIV + 유입 vs GIV      - GIV 가 버린 나머지 유입이 예측을 더 낫게 하는가 (핵심)
합산 = pd.read_csv(데이터폴더 / "확장_합산시총_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)["Total"] / 1e9
합산 = 합산.reindex(pd.date_range(합산.index.min(), 합산.index.max(), freq="D"))   # 달력일 (주말 포함)
유입신호 = (합산.shift(1) - 합산.shift(6)).reindex(평일)                           # 직전 5 달력일 변화


def 표본외_열(d, 큰열, 기준열):
    """[셀 3] 표본외예측 과 같은 확장창 예측, 넣는 변수만 바꿀 수 있게 한 것"""
    행 = []
    for t in d.index[d.index > 첫창끝]:
        학습 = d[d["끝날"] <= t]
        행.append({"date": t, "실제": d.at[t, "y"],
                  "예측_큰": 맞추고예측(학습, 큰열, d.loc[t]), "예측_기준": 맞추고예측(학습, 기준열, d.loc[t])})
    return pd.DataFrame(행).set_index("date")


d보조 = 예측자료(주h, 결과열["주"]).assign(유입=유입신호).dropna()
print(f"보조신호 자료: {d보조.index.min().date()} ~ {d보조.index.max().date()}, 관측치 {len(d보조)} "
      f"(원래 {len(예측자료(주h, 결과열['주']))})")
비교들 = {"① GIV (원래)": (["신호"] + 통제, 통제),
          "② 유입 전체": (["유입"] + 통제, 통제),
          "③ GIV + 유입 vs GIV": (["신호", "유입"] + 통제, ["신호"] + 통제)}
보조행 = []
print("[보조신호] 스프레드 h = 13, 표본 외 (해석용 - 판정 C 는 [셀 4] 그대로)")
for 이름, (큰열, 기준열) in 비교들.items():
    r = 평가(표본외_열(d보조, 큰열, 기준열), 주h)
    보조행.append({"비교": 이름, "예측수": r["예측수"], "R2": r["R2_신호뺀모형"],
                  "CW_t": r["CW_t_신호뺀모형"], "CW_p": r["CW_p_신호뺀모형"]})
    print(f"  {이름}: 예측 {r['예측수']}개, 표본 외 R² {r['R2_신호뺀모형']:+.2%}, "
          f"Clark-West t {r['CW_t_신호뺀모형']:+.2f}, 한쪽 p {r['CW_p_신호뺀모형']:.3f}")
print("  (① 은 [셀 4] 와 같아야 함. 다르면 유입 자료가 빈 날 때문에 관측치가 줄어든 것)")
pd.DataFrame(보조행).to_csv(데이터폴더 / "거래전략_4단계_보조신호.csv", index=False, encoding=CSV인코딩)
print("저장: 거래전략_4단계_보조신호.csv")