# -*- coding: utf-8 -*-
"""
퀀트랩_유입베타_3.py
=====================================================================
[연구] 유입 베타 연구: 사건 전 '스테이블코인 유입과 같이 움직이던 정도'로 고른 주식이
       GENIUS Act 소식에 더 크게 반응했는가
[단계] 3단계: 본 분석 - 묶음별 사건 반응과 판정

[이 코드가 하는 일]
  여기서 처음으로 사건 창 수익률을 봅니다.
  ① 묶음1~5 와 차이(묶음5 − 묶음1)마다 3부 모형(시장 + IBIT)을 추정 기간에 맞추고
     판정 사건(A,B,C,E)의 창 [0,+1] CAR 을 구함 (설계 "비정상수익률모형", "사건창")
  ② S = (방향 × CAR) 평균, t, p (설계 "판정값", "검정")
  ③ 판정: S_차이 > 0 이고 5% 유의 -> 성립 (설계 "판정_차이")
  ④ 보조(기록만): 묶음 1~5 의 S 가 순서대로 커지는지 - 묶음 번호와 S 의 순위 상관 (설계 "보조_순서")
  ⑤ 그림: 묶음별 S 와 95% 범위 / 사건별 차이 CAR

[근거]
  CAR·t·S 계산: 퀀트랩_사건연구_3 [셀 1] 과 같은 방법 (MacKinlay 1997)
  p값: t분포, 자유도 = 추정 기간 일수 − 계수 3개 / 순위 상관: 스피어만 (판정에는 안 씀)

[입력] 유입베타_설계.csv, 유입베타_묶음수익률_일별.csv, 유입베타_요인_일별.csv, 유입베타_2단계_검정력.csv,
       사건연구_추정기간.csv, 사건연구_사건창.csv
[출력] 유입베타_3단계_사건별.csv, 유입베타_3단계_판정.csv, 그림/유입베타_그림1_묶음별반응.png
[Spyder] "#%%" 셀 단위로 Shift+Enter
=====================================================================
"""

#%% [셀 0] 패키지 + 자료 읽기 =============================================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

결과폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩"
데이터폴더, 그림폴더 = 결과폴더 / "데이터", 결과폴더 / "그림"
CSV인코딩 = "utf-8-sig"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

설계 = pd.read_csv(데이터폴더 / "유입베타_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
묶음 = pd.read_csv(데이터폴더 / "유입베타_묶음수익률_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
요인 = pd.read_csv(데이터폴더 / "유입베타_요인_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
검정력 = pd.read_csv(데이터폴더 / "유입베타_2단계_검정력.csv", encoding=CSV인코딩).set_index("묶음")
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)

자료 = 묶음.join(요인[["시장", "IBIT"]])
대상들 = [f"묶음{g}" for g in range(1, 6)] + ["차이"]
판정사건 = 설계["판정사건"].split(",")
유의 = float(설계["유의수준"])
print(f"판정 사건 {판정사건}, 창 {설계['사건창']}, 유의수준 {유의}")


#%% [셀 1] 사건 연구 함수 ==================================================
def 사건연구(대상, 사건들=판정사건, 기간이름="주(250일)", 창=None, 표=자료):
    """대상(묶음 열)의 사건별 CAR 과 판정값 S. 퀀트랩_사건연구_3 [셀 1] 과 같은 계산.
    창: None 이면 사건창.csv 그대로, (앞, 뒤) 를 주면 각 사건 첫 0일 기준으로 다시 정함 (4단계 강건성용)"""
    기간 = 추정기간.loc[기간이름]
    추정 = 표.loc[기간["시작"]:기간["끝"]]
    X = np.column_stack([np.ones(len(추정)), 추정["시장"], 추정["IBIT"]])
    y = 추정[대상].to_numpy()
    계수, *_ = np.linalg.lstsq(X, y, rcond=None)                    # 설계 "비정상수익률모형" (3부 주모형)
    잔차 = y - X @ 계수
    자유도 = len(y) - X.shape[1]
    σ = np.sqrt(잔차 @ 잔차 / 자유도)

    거래일 = 표.index
    행 = []
    for _, e in 사건창[사건창["이름"].isin(사건들)].iterrows():
        시작, 끝 = pd.Timestamp(e["창시작"]), pd.Timestamp(e["창끝"])
        if 창 is not None:
            k0 = 거래일.get_loc(시작)
            k1 = 거래일.get_loc(끝) - 1                                 # 마지막 0일 (주 창이 [0,+1] 이므로)
            시작, 끝 = 거래일[k0 + 창[0]], 거래일[k1 + 창[1]]
        구간 = 표.loc[시작:끝]
        AR = 구간[대상].to_numpy() - np.column_stack([np.ones(len(구간)), 구간["시장"], 구간["IBIT"]]) @ 계수
        CAR, SE = AR.sum(), σ * np.sqrt(len(구간))
        행.append({"대상": 대상, "사건": e["이름"], "방향": int(e["방향"]), "창시작": 시작.date(), "창끝": 끝.date(),
                   "창길이": len(구간), "CAR_%": CAR * 100, "SE_%": SE * 100, "t": CAR / SE,
                   "p": 2 * stats.t.sf(abs(CAR / SE), 자유도)})
    사건별 = pd.DataFrame(행)
    S = (사건별["방향"] * 사건별["CAR_%"]).mean()                    # 설계 "판정값" (부결 A 는 부호 뒤집음)
    SE_S = np.sqrt((사건별["SE_%"] ** 2).sum()) / len(사건별)
    t = S / SE_S
    return 사건별, {"대상": 대상, "S_%": S, "SE_%": SE_S, "t": t, "p": 2 * stats.t.sf(abs(t), 자유도),
                    "최소검출효과_%": 2.8 * SE_S, "자유도": 자유도}


#%% [셀 2] 판정 ============================================================
사건표, 판정행 = [], []
for 대상 in 대상들:
    사건별, 요약 = 사건연구(대상)
    사건표.append(사건별)
    판정행.append(요약)
사건표 = pd.concat(사건표, ignore_index=True)
판정 = pd.DataFrame(판정행).set_index("대상")

# 2단계 검정력과 같은 표준오차인지 (같은 계산이라는 점검)
assert np.isclose(판정.loc["차이", "SE_%"], 검정력.loc["차이", "S_표준오차_%"]), "2단계 표준오차와 다릅니다"

# 판정 (출처: 설계 "판정_차이")
d = 판정.loc["차이"]
성립 = (d["S_%"] > 0) and (d["p"] < 유의)
판정["판정"] = ""
판정.loc["차이", "판정"] = "성립" if 성립 else "성립 안 함"

# 보조: 묶음 번호와 S 의 순위 상관 (출처: 설계 "보조_순서" - 기록만)
S묶음 = 판정.loc[[f"묶음{g}" for g in range(1, 6)], "S_%"]
ρ, ρp = stats.spearmanr(range(1, 6), S묶음)
순서대로 = bool(np.all(np.diff(S묶음.to_numpy()) > 0))

pd.set_option("display.width", 200)
f = lambda v: f"{v:+.3f}"
print("[사건별 CAR, %] (A 는 부결 - S 계산 때 부호를 뒤집음)")
print(사건표.pivot(index="사건", columns="대상", values="CAR_%")[대상들].to_string(float_format=f))
print("\n[사건별 t값]")
print(사건표.pivot(index="사건", columns="대상", values="t")[대상들].to_string(float_format=f))
print("\n[판정] S = 법이 한 걸음 나아갈 때 사건당 평균 비정상 수익률 (%)")
print(판정[["S_%", "SE_%", "t", "p", "최소검출효과_%", "판정"]].to_string(float_format=f))
print(f"\n=> 차이(묶음5 − 묶음1): S {d['S_%']:+.3f}%, p {d['p']:.3f} -> {판정.loc['차이', '판정']}")
if not 성립:
    print(f"   (최소 검출 효과 {d['최소검출효과_%']:.2f}% - 이보다 작은 차이는 이 자료로 판단 불가)")
print(f"[보조, 기록만] 묶음 번호와 S 의 순위 상관 {ρ:+.2f} (p {ρp:.3f}), 1->5 로 계속 커짐: {순서대로}")


#%% [셀 3] 그림 ============================================================
k = stats.t.ppf(1 - 유의 / 2, int(판정["자유도"].iloc[0]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# 왼쪽: 묶음별 S 와 95% 범위
x = np.arange(len(대상들))
색 = ["steelblue"] * 5 + ["black"]
for xi, 대상, c in zip(x, 대상들, 색):
    ax[0].errorbar(xi, 판정.loc[대상, "S_%"], yerr=k * 판정.loc[대상, "SE_%"], fmt="o", color=c, capsize=3)
ax[0].axhline(0, color="gray", lw=0.8)
ax[0].set_xticks(x)
ax[0].set_xticklabels(["1\n(유입베타 작음)", "2", "3", "4", "5\n(유입베타 큼)", "차이\n(5 - 1)"], fontsize=8)
ax[0].set_ylabel("S (%), 막대 = 95% 범위")
ax[0].set_title("묶음별 사건당 평균 반응 (A,B,C,E)", loc="left", fontsize=11)

# 오른쪽: 사건별 차이 CAR (부결 A 는 부호를 뒤집어 표시)
차 = 사건표[사건표["대상"] == "차이"].reset_index(drop=True)
값 = 차["방향"] * 차["CAR_%"]
위치 = np.arange(len(차))
ax[1].bar(위치, 값, yerr=k * 차["SE_%"], color=["indianred" if v < 0 else "seagreen" for v in 값], capsize=3)
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_xticks(위치)
ax[1].set_xticklabels([f"{s}\n(부호 뒤집음)" if s == "A" else s for s in 차["사건"]], fontsize=8)
ax[1].set_ylabel("방향 × CAR (%), 막대 = 95% 범위")
ax[1].set_title("사건별 차이 (묶음5 - 묶음1)", loc="left", fontsize=11)

fig.suptitle("유입 베타 묶음별 GENIUS Act 반응", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "유입베타_그림1_묶음별반응.png", dpi=150)
plt.show()


#%% [셀 4] 저장 ============================================================
판정["순위상관_보조"] = np.nan
판정.loc["차이", "순위상관_보조"] = ρ
사건표.to_csv(데이터폴더 / "유입베타_3단계_사건별.csv", index=False, encoding=CSV인코딩)
판정.to_csv(데이터폴더 / "유입베타_3단계_판정.csv", encoding=CSV인코딩)
print("저장: 유입베타_3단계_사건별.csv, 유입베타_3단계_판정.csv, 그림/유입베타_그림1_묶음별반응.png")
