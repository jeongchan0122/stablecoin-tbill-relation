# -*- coding: utf-8 -*-
"""
퀀트랩_거래전략_1.py
=====================================================================
[연구] 거래전략 연구 (설계: 퀀트랩_거래전략_0.py -> 데이터/거래전략_설계.csv)
[단계] 1단계: 실시간 GIV 만들기 - 그날 실제로 알 수 있었던 정보만으로 GIV 를 다시 계산

[왜 다시 만드나]
  재현용 GIV(퀀트랩_BIS_2, BIS_4-1)는 윈저화 기준(MAD)과 패널 페어를 2026년까지의 자료를 다 보고 정함.
  2022년의 투자자는 그 값을 알 수 없었으므로, 예측·거래 연구에 쓰면 결과가 실제보다 좋아질 수 있음.
  -> 이 코드는 날마다 "그날까지의 자료"로만 MAD 와 패널 페어를 정해 GIV 를 다시 계산함.

[이 코드는 금리(결과변수)를 전혀 읽지 않습니다]
  점검에 쓰는 것은 GIV 와 스테이블코인 유입뿐 -> 결과를 엿보지 않고 신호만 만듦

[출처 표기 규칙]  (퀀트랩_거래전략_0.py 와 같음)
  출처: 논문 ○○절 / 각주 ○ / 표 ○      출처: 저자 답신 (2026-10-12)
  출처: 퀀트랩_BIS_○                     출처: 사용자 확정 (날짜)
  출처: 설계 "항목이름"                   -> 데이터/거래전략_설계.csv 의 해당 줄
  [논문에 명시 없음]                      -> 논문에 없어 정해야 했던 부분 (이유·영향 함께)

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  거래전략_설계.csv              (퀀트랩_거래전략_0.py)
  확장_코인체인_패널_일별.csv    (퀀트랩_BIS_4-1 [셀 1], 2026-10-12 UTC 에 받은 자료)
  확장_합산시총_일별.csv         (퀀트랩_BIS_4-1 [셀 1]) - 1단계 F 점검용
  확장_GIV_평일.csv              (퀀트랩_BIS_4-1 [셀 4]) - 재현용(전체 표본) GIV 와 비교용
[출력]
  데이터/거래전략_실시간GIV_달력일.csv   (날마다의 MAD, 윈저화 경계, N, 일별 GIV, 5일 합)
  데이터/거래전략_신호_평일.csv          (거래에 쓰는 신호 = 직전 달력일의 GIV 5일 합)
  데이터/거래전략_1단계_점검.csv
  그림/거래전략_그림1_실시간MAD.png, 그림/거래전략_그림2_GIV비교.png

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install pandas numpy statsmodels matplotlib
=====================================================================
"""

#%% [셀 0] 패키지 + 폴더 + 설계 읽기 =======================================
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 폴더 (출처: 사용자 지시 - 바탕화면/스테이블코인_퀀트랩, 한국어 파일명)
결과폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩"
데이터폴더 = 결과폴더 / "데이터"
그림폴더 = 결과폴더 / "그림"
그림폴더.mkdir(parents=True, exist_ok=True)
CSV인코딩 = "utf-8-sig"

# 설계 읽기 (출처: 퀀트랩_거래전략_0.py - 이 코드는 설계값을 직접 적지 않고 파일에서 가져옴)
설계 = pd.read_csv(데이터폴더 / "거래전략_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
def 날짜(k): return pd.Timestamp(설계[k])
def 정수(k): return int(설계[k])
def 실수(k): return float(설계[k])

표본시작 = 날짜("표본시작")              # 2021-01-04
표본끝 = 날짜("표본끝")                  # 2026-03-31
패널시작 = 날짜("패널시작")              # 2021-01-01
비움일수 = 정수("처음비움_달력일")        # 값은 설계 파일에서 읽음 (달력일, 주말 포함)
신호지연 = 정수("신호지연_달력일")        # 1
활성기준 = 실수("활성기준_달러")          # 1,000,000
최소N = 정수("최소N")                    # 5
윈저배수 = 실수("윈저배수")              # 8 × 1.4826

# 처음 비우는 기간의 마지막 날: 2021-01-01 부터 달력일 "처음비움_달력일" 일 (출처: 설계 "처음비움_달력일")
비움끝 = 패널시작 + pd.Timedelta(days=비움일수 - 1)
print("데이터 폴더:", 데이터폴더)
print(f"설계: 표본 {표본시작.date()} ~ {표본끝.date()}, GIV 비움 ~ {비움끝.date()}, 신호 지연 {신호지연}일")


#%% [셀 1] 코인 × 체인 패널 읽기 → 날짜 × 페어 표 ==============================
# 입력 패널 = BIS_4-1 이 2026-10-12 에 받은 확장 패널 하나 (출처: 사용자 확정 10/15)
#   이유: BIS_1 패널(10/01 받음, 3월까지)과 이어 붙이면 받은 날짜가 달라 이음매에서 값이 튈 수 있음
#   (DefiLlama 는 과거 값을 사후 수정함 - 저자 답신)
패널 = pd.read_csv(데이터폴더 / "확장_코인체인_패널_일별.csv", parse_dates=["date"], encoding=CSV인코딩)
패널 = 패널[패널["date"] >= 패널시작]

# 날짜(행) × 페어(열) 표, 하루도 빠짐없는 달력일로 맞춤 (출처: 퀀트랩_BIS_2 [셀 2] 와 같음)
시총표 = 패널.pivot_table(index="date", columns="pair", values="mcap", aggfunc="sum")
시총표 = 시총표.reindex(pd.date_range(시총표.index.min(), 시총표.index.max(), freq="D"))
빈날 = 시총표.index[시총표.isna().all(axis=1)]
print(f"패널: {시총표.index.min().date()} ~ {시총표.index.max().date()}, 페어 {시총표.shape[1]}개")
print(f"자료가 통째로 없는 날: {len(빈날)}일 (0이면 영향 없음)")


#%% [셀 2] 성장률, 활성 여부, 실시간 패널 페어 ===============================
# 성장률과 활성 (출처: 퀀트랩_BIS_2 [셀 2] 와 같은 계산)
#   성장률 g = (오늘 − 어제) ÷ 어제 × 100 (%p)  [논문에 명시 없음: 성장률 공식 - BIS_2 와 같음]
#   활성 = 어제·오늘 모두 시총 $1M 초과 (출처: 논문 각주 18)
전날 = 시총표.shift(1)
g = (시총표 - 전날) / 전날.where(전날 > 0) * 100
활성 = (시총표 > 활성기준) & (전날 > 활성기준)

# 실시간 패널 페어 = 그날까지 한 번이라도 활성이었던 페어 (출처: 설계 "패널_실시간", 사용자 확정 10/15)
#   cummax: 위에서부터 내려가며 "지금까지 한 번이라도 True 였나"를 기록
#   (재현용 BIS_2 는 2026년까지 보고 정함 -> 여기서는 그날까지만 봄)
실시간패널 = 활성.astype(int).cummax().astype(bool)

print(f"활성 페어-일 합계: {int(활성.values.sum()):,}")
print(f"실시간 패널 페어 수: 2021-03-31 {int(실시간패널.loc['2021-03-31'].sum())}개, "
      f"2026-03-31 {int(실시간패널.loc['2026-03-31'].sum())}개 (재현용 BIS_2: 226개), "
      f"마지막 날 {int(실시간패널.iloc[-1].sum())}개")


#%% [셀 3] 실시간 윈저화 기준 (MAD) =========================================
# 윈저화: 튀는 성장률을 0 ± 8 × 1.4826 × MAD 로 자름
#   방법 (출처: 저자 답신) MAD 는 날짜·페어를 모두 모은 풀링 표본에서, 풀링 중앙값으로부터의 거리의 중앙값.
#        활성 관측치만 씀. 경계는 0 중심.
#   실시간 (출처: 설계 "MAD_실시간", 사용자 확정 10/15) 날마다 "그날까지(그날 포함)" 쌓인 활성 성장률로 다시 계산
#
# 계산 방법: 활성 성장률을 날짜 순서로 한 줄로 이어 붙이고, t일에는 처음부터 t일까지 부분만 써서
#   중앙값 -> 거리 -> 거리의 중앙값을 구함 (날마다 처음부터 다시 계산하므로 몇 초~수십 초 걸림)
값 = g.where(활성).values
하루값 = [줄[~np.isnan(줄)] for 줄 in 값]                  # 날마다의 활성 성장률
누적개수 = np.cumsum([len(v) for v in 하루값])
이어붙임 = np.concatenate(하루값)

MAD = np.full(len(g), np.nan)
for i, n in enumerate(누적개수):
    if n == 0:
        continue                                           # 첫날처럼 활성 페어가 없으면 계산 불가
    지금까지 = 이어붙임[:n]
    중앙값 = np.median(지금까지)
    MAD[i] = np.median(np.abs(지금까지 - 중앙값))
MAD = pd.Series(MAD, index=g.index, name="MAD")
경계 = (윈저배수 * MAD).rename("윈저화경계")

# 윈저화: 날마다 그날의 경계로 자름 (0 중심, 출처: 저자 답신)
gw = g.clip(lower=-경계, upper=경계, axis=0)

for d in ["2021-03-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2026-03-31"]:
    print(f"  {d}: 쌓인 활성 성장률 {int(누적개수[g.index.get_loc(pd.Timestamp(d))]):,}개, "
          f"윈저화 경계 ±{경계.loc[d]:.3f} %p")
print(f"  (재현용 BIS_3 [셀 1] 의 2026-03-31 까지 전체 기준: ±1.15 %p)")


#%% [셀 4] 실시간 일별 GIV → 5일 합 → 거래 신호 ================================
# 식 (2) (출처: 논문 2.3절, 퀀트랩_BIS_2 [셀 3] GIV_일별 과 같은 계산)
#   s   = 전날 점유율. 분모 = 그날 어제·오늘 모두 활성인 페어들의 전날 시총 합 (BIS_2 와 같음)
#   ḡ   = Σ s·g (규모가중 평균 성장률), ε = g − ḡ (페어 고유 충격)
#   GIV = Σ (s − 1/N)·ε
#   N   = t−1일 시총 $1M 초과인 페어 수 (출처: 저자 답신. "t−1 활성" 해석은 설계 "N해석")
#         단, 그날까지의 실시간 패널 페어 안에서 셈 (출처: 설계 "패널_실시간")
#   N < 5 면 빈 값 (출처: 각주 18)
s = 전날.where(활성)
s = s.div(s.sum(axis=1), axis=0)
gg = gw.where(활성)
N = ((전날 > 활성기준) & 실시간패널).sum(axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    ε = gg.sub((s * gg).sum(axis=1), axis=0)
    GIV일별 = (s.sub(1 / N, axis=0) * ε).sum(axis=1, min_count=1)
GIV일별[N < 최소N] = np.nan
GIV일별[GIV일별.index <= 비움끝] = np.nan                    # 처음 기간 비움 (출처: 설계 "처음비움_달력일")

# 5일 합 = 최근 달력일 5개(t−4 ~ t) 합 (출처: 저자 답신, BIS_2 "가-3")
#   5일 중 하루라도 빈 값이면 합도 빈 값
GIV5 = GIV일별.rolling(5).sum()

# 거래 신호 (출처: 설계 "신호지연_달력일")
#   t일 거래에 쓰는 값 = 직전 달력일(t−1)의 GIV 5일 합. 월요일 거래에는 일요일 값을 씀
신호_달력 = GIV5.shift(신호지연)
평일 = pd.bdate_range(GIV5.index.min(), GIV5.index.max())    # 평일 달력 (출처: 퀀트랩_BIS_2 [셀 0])
신호 = 신호_달력.reindex(평일)

print(f"일별 GIV: 값 있는 날 {GIV일별.notna().sum()}일, 첫날 {GIV일별.first_valid_index().date()}")
print(f"거래 신호(평일): 값 있는 날 {신호.notna().sum()}일, 첫날 {신호.first_valid_index().date()}, "
      f"마지막 {신호.last_valid_index().date()}")
print(f"N 이 5 미만이라 빈 날(비움 기간 뒤): {int(((N < 최소N) & (N.index > 비움끝)).sum())}일")


#%% [셀 5] 저장 =============================================================
달력표 = pd.DataFrame({"MAD": MAD, "윈저화경계": 경계, "N_t−1활성": N,
                      "실시간패널페어수": 실시간패널.sum(axis=1),
                      "GIV_일별_실시간": GIV일별, "GIV_5일합_실시간": GIV5})
달력표.index.name = "date"
달력표.to_csv(데이터폴더 / "거래전략_실시간GIV_달력일.csv", encoding=CSV인코딩)

평일표 = pd.DataFrame({"거래신호_GIV5": 신호,                       # 이 값으로 t일에 거래
                      "GIV5_당일_참고": GIV5.reindex(평일)})        # 같은 날 값 (거래에는 안 씀, 점검용)
평일표.index.name = "date"
평일표.to_csv(데이터폴더 / "거래전략_신호_평일.csv", encoding=CSV인코딩)
print("저장: 거래전략_실시간GIV_달력일.csv, 거래전략_신호_평일.csv")


#%% [셀 6] 점검 ① 재현용(전체 표본) GIV 와 비교 ===============================
# 비교 대상: BIS_4-1 이 같은 확장 패널로 계산한 전체 표본 GIV (확장_GIV_평일.csv 의 GIV5_확장끝)
#   같은 날 받은 자료라서 차이는 "미래 정보(MAD, 패널 페어)를 썼는가" 하나뿐
# 판단 기준은 두지 않음 - 숫자와 그림만 기록 (출처: 사용자 확정 10/15, 근거 없는 문턱값을 만들지 않음)
재현 = pd.read_csv(데이터폴더 / "확장_GIV_평일.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)["GIV5_확장끝"]
비교 = pd.concat([GIV5.reindex(평일).rename("실시간"), 재현.rename("재현용")], axis=1).dropna()

점검행 = [{"점검": "상관(전체)", "값": 비교.corr().iloc[0, 1], "관측치": len(비교)}]
print(f"같은 날 GIV 5일 합의 상관 (실시간 vs 재현용): 전체 {점검행[0]['값']:.4f} (관측치 {len(비교)})")
for 연도, 묶음 in 비교.groupby(비교.index.year):
    r = 묶음.corr().iloc[0, 1]
    점검행.append({"점검": f"상관({연도})", "값": r, "관측치": len(묶음)})
    print(f"  {연도}: {r:.4f} (관측치 {len(묶음)})")
차이 = (비교["실시간"] - 비교["재현용"]).abs()
점검행.append({"점검": "절대차이 중앙값", "값": 차이.median(), "관측치": len(비교)})
점검행.append({"점검": "절대차이 최대", "값": 차이.max(), "관측치": len(비교)})
print(f"절대 차이: 중앙값 {차이.median():.4f}, 최대 {차이.max():.4f} "
      f"(GIV 5일 합 표준편차 {비교['재현용'].std():.3f} 대비)")


#%% [셀 7] 점검 ② 1단계 F: 실시간 GIV 도 유입을 움직이는가 =====================
# 5일 유입 = 합산 시총(평일) − 5 평일 전 값, $B (출처: 논문 식 (1), 각주 13, 퀀트랩_BIS_2 [셀 1])
# 회귀: 5일 유입 = 상수 + 계수 × GIV 5일 합 (같은 날), Newey-West 시차 1 (출처: 논문 표 2 주석)
#   F = (계수 ÷ 표준오차)² (출처: 퀀트랩_BIS_2 [셀 3])
# 실시간 GIV 와 재현용 GIV 를 "같은 날짜들"에서 돌려 나란히 비교 (공정한 비교를 위해)
# 구간: 논문 표본(신호 첫날 ~ 2026-03-31) / 자료 전체
합산 = pd.read_csv(데이터폴더 / "확장_합산시총_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
MC = 합산["Total"].reindex(평일) / 1e9
유입5 = (MC - MC.shift(5)).rename("유입5")


def 일단계(giv5, 구간끝):
    d = pd.concat([유입5, giv5.rename("GIV")], axis=1).dropna()
    d = d[d.index <= 구간끝]
    r = sm.OLS(d["유입5"], sm.add_constant(d["GIV"])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return {"계수": r.params["GIV"], "F": r.tvalues["GIV"] ** 2, "R2": r.rsquared, "관측치": int(r.nobs)}


공통날 = 비교.index                                      # 두 GIV 가 모두 있는 평일
print("[1단계] 5일 유입 ← GIV 5일 합 (NW 시차 1)")
for 구간이름, 구간끝 in [("논문 표본", 표본끝), ("자료 전체", 공통날.max())]:
    for 이름 in ["실시간", "재현용"]:
        r = 일단계(비교.loc[공통날, 이름], 구간끝)
        print(f"  {구간이름} / {이름}: 계수 {r['계수']:.3f}, F {r['F']:.1f}, R² {r['R2']:.3f}, 관측치 {r['관측치']}")
        점검행.append({"점검": f"1단계F {구간이름} {이름}", "값": r["F"], "관측치": r["관측치"]})
        점검행.append({"점검": f"1단계계수 {구간이름} {이름}", "값": r["계수"], "관측치": r["관측치"]})
print("  (참고: 퀀트랩_BIS_2 [셀 6] 재현 1단계 F 34.5, 2021-01-11 ~ 2026-03-31)")

pd.DataFrame(점검행).to_csv(데이터폴더 / "거래전략_1단계_점검.csv", index=False, encoding=CSV인코딩)
print("저장: 거래전략_1단계_점검.csv")


#%% [셀 8] 그림 =============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False          # 맑은 고딕에 수학용 빼기(−)가 없어 깨지는 것 방지

# 그림 1: 실시간 윈저화 경계가 언제쯤 안정되는가 (출처: 사용자 확정 10/15 - MAD 그래프로 안정 여부 확인)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(경계.index, 경계.values, color="#2171b5", lw=1, label="실시간 경계 (그날까지의 자료)")
ax.axhline(1.15, color="black", ls="--", lw=0.8, label="재현용 경계 ±1.15 (2026-03 까지 전체)")
ax.axvspan(경계.index.min(), 비움끝, color="#bbbbbb", alpha=0.4, label=f"GIV 비움 (~{비움끝.date()})")
ax.set_ylabel("윈저화 경계 (%p)")
ax.set_title("실시간 윈저화 경계 = 8 × 1.4826 × MAD", loc="left", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(그림폴더 / "거래전략_그림1_실시간MAD.png", dpi=150)
plt.close(fig)

# 그림 2: 실시간 GIV 와 재현용 GIV (5일 합)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
axes[0].plot(비교.index, 비교["재현용"], color="#bbbbbb", lw=0.8, label="재현용 (전체 표본)")
axes[0].plot(비교.index, 비교["실시간"], color="#2171b5", lw=0.6, label="실시간")
axes[0].set_ylabel("GIV 5일 합 (%)")
axes[0].legend(fontsize=8)
axes[1].scatter(비교["재현용"], 비교["실시간"], s=3, color="#2171b5")
lim = [비교.min().min(), 비교.max().max()]
axes[1].plot(lim, lim, color="black", lw=0.8)
axes[1].set_xlabel("재현용")
axes[1].set_ylabel("실시간")
fig.suptitle(f"GIV 5일 합: 실시간 vs 재현용 (상관 {비교.corr().iloc[0, 1]:.4f})", fontsize=10)
fig.tight_layout()
fig.savefig(그림폴더 / "거래전략_그림2_GIV비교.png", dpi=150)
plt.close(fig)
print("저장: 그림/거래전략_그림1_실시간MAD.png, 그림/거래전략_그림2_GIV비교.png")

#%% [셀 9] 진단: 실시간 GIV 가 어디서 재현용과 갈라지는가 (금리 자료 안 씀) ===========
# 목적: 실시간 GIV 가 재현용과 갈라지는 날, 연도별 1단계 관계 확인 (출처: 사용자 확정 10/15)
#   처음 90일 비움에서 1단계 F 가 0 근처였던 원인 진단으로 시작 -> 신호를 2024년부터로 정한 근거 (설계 "신호기간_근거")
# 계산 방법은 [셀 4] 와 같음. 새로 정한 설정 없음 (출력만 함)
기여 = s.sub(1 / N, axis=0) * ε                     # 페어별 GIV 기여 = 식 (2)의 각 항
일별차이 = (비교["실시간"] - 비교["재현용"]).abs().sort_values(ascending=False)
print("[실시간 − 재현용 GIV 5일 합 차이가 큰 날 10개]")
for d in 일별차이.index[:10]:
    창 = 기여.loc[d - pd.Timedelta(days=4): d].sum()  # 5일 합의 창 (t−4 ~ t) 동안 페어별 기여 합
    p = 창.abs().idxmax()
    최대성장률 = g.loc[d - pd.Timedelta(days=4): d, p].abs().max()
    print(f"  {d.date()}: 실시간 {비교.loc[d, '실시간']:+.2f}, 재현용 {비교.loc[d, '재현용']:+.2f}"
          f" / 기여 가장 큰 페어 {p} ({창[p]:+.2f}), 그 페어 원래 성장률 최대 {최대성장률:.1f}%p,"
          f" 그날 경계 ±{경계.loc[d]:.2f}")

print("\n[연도별 1단계 F] 5일 유입 ← GIV 5일 합 (NW 시차 1)")
연도자료 = pd.concat([유입5, 비교], axis=1).dropna()
for 연도, 묶음 in 연도자료.groupby(연도자료.index.year):
    줄 = []
    for 이름 in ["실시간", "재현용"]:
        r = sm.OLS(묶음["유입5"], sm.add_constant(묶음[이름])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        줄.append(f"{이름} F {r.tvalues[이름] ** 2:.1f}")
    print(f"  {연도}: " + " / ".join(줄) + f"  (관측치 {len(묶음)})")