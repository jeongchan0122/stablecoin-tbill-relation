# -*- coding: utf-8 -*-
"""
퀀트랩_유입베타_한국_2.py
=====================================================================
[연구] 유입 베타 연구 - 한국 확장: 같은 질문을 코스피·코스닥 주식으로 다시 봄
[단계] 한국 2단계: 판정 + 강건성 7가지

[이 코드가 하는 일]
  여기서 처음으로 한국 사건 창 수익률을 봅니다.
  ① 종목마다 '자기 시장 지수 + IBIT' 모형으로 비정상 수익률 -> 묶음별 그날 평균 (설계 "한국_묶음비정상수익률")
  ② 판정 사건(A, B1, B2, C, E)의 창 CAR -> S, t, p (설계 "판정값", "검정")
  ③ 판정: S_차이 > 0 이고 5% 유의 -> 성립 (설계 "판정_차이") / 보조: 묶음 1~5 순위 상관 (기록만)
  ④ 강건성 7가지 (설계 "강건성목록" - 미국 4단계와 같은 목록, 기록만)
  ⑤ 그림: 묶음별 S, 사건별 차이 CAR, 강건성

[근거]
  유입 베타·묶음: 퀀트랩_유입베타_한국_1 과 같은 계산 / CAR·S·t: MacKinlay (1997)
  표준편차 = 추정 기간 묶음 비정상 수익률의 표준편차, p값 자유도 = 추정 기간 일수 − 1
  ② 시가총액 가중의 가중치 = 한국 추정 기간 마지막 날(묶음을 나누는 날)의 시가총액 (미국 4단계와 같은 방식)
  ⑥ 추정 기간 120일 = 유입 베타와 비정상 수익률 모형을 모두 120일로 추정 (미국 4단계와 같은 방식)

[입력] 유입베타_설계.csv, 유입베타_한국_사건창·추정기간·수익률_일별·시가총액_일별·요인_일별·베타·검정력 .csv (한국 1단계)
[출력] 유입베타_한국_사건별.csv, 유입베타_한국_판정.csv, 유입베타_한국_강건성.csv,
       그림/유입베타_한국_그림1_묶음별반응.png, 그림/유입베타_한국_그림2_강건성.png
[Spyder] "#%%" 셀 단위로 Shift+Enter ([셀 1]이 종목별 회귀를 여러 번 돌려 1~2분 걸릴 수 있음)
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


def 읽기(이름, **kw):
    return pd.read_csv(데이터폴더 / 이름, encoding=CSV인코딩, **kw)


설계 = 읽기("유입베타_설계.csv", dtype=str).set_index("항목")["값"]
사건창 = 읽기("유입베타_한국_사건창.csv")
추정기간 = 읽기("유입베타_한국_추정기간.csv").set_index("구분")
수익률 = 읽기("유입베타_한국_수익률_일별.csv", index_col="date", parse_dates=["date"])
시총 = 읽기("유입베타_한국_시가총액_일별.csv", index_col="date", parse_dates=["date"])
요인 = 읽기("유입베타_한국_요인_일별.csv", index_col="date", parse_dates=["date"])
베타1단계 = 읽기("유입베타_한국_베타.csv", index_col="코드")
검정력 = 읽기("유입베타_한국_검정력.csv", index_col="묶음")

내시장 = 베타1단계["시장"].map({"KS": "코스피", "KQ": "코스닥"})     # 설계 "한국_시장수익률"
유의 = float(설계["유의수준"])
판정사건 = list(사건창.loc[사건창["판정에씀"], "이름"])                 # A, B1, B2, C, E (설계 "한국_사건창결과")
print(f"판정 사건 {판정사건}, 종목 {수익률.shape[1]:,}개")


#%% [셀 1] 계산 함수 + 종목별 비정상 수익률·유입 베타 ========================
def 기간(이름):
    return pd.Timestamp(추정기간.loc[이름, "시작"]), pd.Timestamp(추정기간.loc[이름, "끝"])


def 종목계산(기간이름="주(250일)", 비트코인=True):
    """종목마다 유입 베타·t (자기 시장 + IBIT + 유입) 와 전체 날짜의 비정상 수익률 (자기 시장 + IBIT)"""
    s, e = 기간(기간이름)
    AR = pd.DataFrame(index=수익률.index, columns=수익률.columns, dtype=float)
    행 = {}
    for c in 수익률.columns:
        y = 수익률[c]
        m = 요인[내시장[c]]
        추 = (y.index >= s) & (y.index <= e) & y.notna().to_numpy()
        # 비정상 수익률 모형 (3부 모형: 자기 시장 + IBIT)
        X = np.column_stack([np.ones(len(y)), m, 요인["IBIT"]])
        b, *_ = np.linalg.lstsq(X[추], y.to_numpy()[추], rcond=None)
        AR[c] = y.to_numpy() - X @ b
        # 유입 베타 (설계 "베타모형", ③ 은 IBIT 를 뺌)
        열 = [np.ones(len(y)), m] + ([요인["IBIT"]] if 비트코인 else []) + [요인["유입_B"]]
        Xb = np.column_stack(열)[추]
        g, *_ = np.linalg.lstsq(Xb, y.to_numpy()[추], rcond=None)
        r = y.to_numpy()[추] - Xb @ g
        SE = np.sqrt(r @ r / (len(r) - Xb.shape[1]) * np.linalg.inv(Xb.T @ Xb)[-1, -1])
        행[c] = (g[-1], g[-1] / SE)
    return AR, pd.DataFrame(행, index=["베타", "t"]).T


def 묶음AR(AR, 점수, 묶음수=5, 가중="동일", 기간이름="주(250일)"):
    """점수 순서로 묶음 -> 묶음별 그날 평균 비정상 수익률 + 차이(맨 위 − 맨 아래)"""
    번호 = pd.qcut(점수, 묶음수, labels=False) + 1
    out = {}
    for g in range(1, 묶음수 + 1):
        멤버 = 번호.index[번호 == g]
        A = AR[멤버]
        if 가중 == "동일":
            out[f"묶음{g}"] = A.mean(axis=1)
        else:                                            # 시가총액 가중 (묶음을 나누는 날 시가총액, 고정)
            w = 시총.loc[기간(기간이름)[1], 멤버]
            W = A.notna().mul(w, axis=1)
            out[f"묶음{g}"] = (A.fillna(0) * W).sum(axis=1) / W.sum(axis=1)
    d = pd.DataFrame(out)
    d["차이"] = d[f"묶음{묶음수}"] - d["묶음1"]
    return d


def 판정값(ar, 사건들=판정사건, 기간이름="주(250일)", 창=None, 사건표=False):
    """묶음 비정상 수익률 ar 의 사건별 CAR, S, t, p"""
    s, e = 기간(기간이름)
    추정 = ar.loc[s:e]
    σ, 자유도 = 추정.std(ddof=1), len(추정) - 1
    거래일 = ar.index
    행 = []
    for _, ev in 사건창[사건창["이름"].isin(사건들)].iterrows():
        a, b = pd.Timestamp(ev["창시작"]), pd.Timestamp(ev["창끝"])
        if 창 is not None:
            k0, k1 = 거래일.get_loc(a), 거래일.get_loc(b) - 1      # 첫 0일, 마지막 0일
            a, b = 거래일[k0 + 창[0]], 거래일[k1 + 창[1]]
        구간 = ar.loc[a:b]
        CAR, SE = 구간.sum() * 100, σ * np.sqrt(len(구간)) * 100
        행.append({"사건": ev["이름"], "방향": int(ev["방향"]), "창시작": a.date(), "창끝": b.date(), "창길이": len(구간),
                   "CAR_%": CAR, "SE_%": SE, "t": CAR / SE, "p": 2 * stats.t.sf(abs(CAR / SE), 자유도)})
    d = pd.DataFrame(행)
    S = (d["방향"] * d["CAR_%"]).mean()
    SE_S = np.sqrt((d["SE_%"] ** 2).sum()) / len(d)
    요약 = {"S_%": S, "SE_%": SE_S, "t": S / SE_S, "p": 2 * stats.t.sf(abs(S / SE_S), 자유도),
            "최소검출효과_%": 2.8 * SE_S, "자유도": 자유도}
    return (d, 요약) if 사건표 else 요약


AR주, 베타주 = 종목계산()
AR120, 베타120 = 종목계산("강건성(120일)")
_, 베타무비트 = 종목계산(비트코인=False)
assert np.allclose(베타주["베타"], 베타1단계.loc[베타주.index, "유입베타"]), "1단계 유입 베타와 다릅니다"
print("종목별 계산 끝 (1단계 유입 베타와 같음 확인)")


#%% [셀 2] 주 판정 ========================================================
주묶음 = 묶음AR(AR주, 베타주["베타"])
사건표, 판정행 = [], []
for 대상 in 주묶음.columns:
    d, 요약 = 판정값(주묶음[대상], 사건표=True)
    사건표.append(d.assign(대상=대상))
    판정행.append({"대상": 대상, **요약})
사건표 = pd.concat(사건표, ignore_index=True)
판정 = pd.DataFrame(판정행).set_index("대상")
assert np.isclose(판정.loc["차이", "SE_%"], 검정력.loc["차이", "S_표준오차_%"]), "1단계 검정력과 표준오차가 다릅니다"

d = 판정.loc["차이"]
판정["판정"] = ""
판정.loc["차이", "판정"] = "성립" if (d["S_%"] > 0 and d["p"] < 유의) else "성립 안 함"
S묶음 = 판정.loc[[f"묶음{g}" for g in range(1, 6)], "S_%"]
ρ, ρp = stats.spearmanr(range(1, 6), S묶음)

pd.set_option("display.width", 220)
f = lambda v: f"{v:+.3f}"
순서 = list(주묶음.columns)
print("[사건별 CAR, %] (A 는 부결 - S 계산 때 부호를 뒤집음)")
print(사건표.pivot(index="사건", columns="대상", values="CAR_%")[순서].to_string(float_format=f))
print("\n[사건별 t값]")
print(사건표.pivot(index="사건", columns="대상", values="t")[순서].to_string(float_format=f))
print("\n[판정] S = 법이 한 걸음 나아갈 때 사건당 평균 비정상 수익률 (%)")
print(판정[["S_%", "SE_%", "t", "p", "최소검출효과_%", "판정"]].to_string(float_format=f))
print(f"\n=> 한국 차이(묶음5 − 묶음1): S {d['S_%']:+.3f}%, p {d['p']:.3f} -> {판정.loc['차이', '판정']}")
print(f"[보조, 기록만] 순위 상관 {ρ:+.2f} (p {ρp:.3f}), 1->5 로 계속 커짐: {bool(np.all(np.diff(S묶음.to_numpy()) > 0))}")


#%% [셀 3] 강건성 7가지 ====================================================
# 출처: 설계 "강건성목록" (미국 4단계와 같은 목록), "강건성_판정" (기록만)
경우들 = {
    "주 결과 (5묶음, 동일가중, 판정 사건, [0,+1])": (주묶음["차이"], {}),
    "① 묶음 10개":       (묶음AR(AR주, 베타주["베타"], 묶음수=10)["차이"], {}),
    "② 시가총액 가중":    (묶음AR(AR주, 베타주["베타"], 가중="시총")["차이"], {}),
    "③ 비트코인 없이 베타": (묶음AR(AR주, 베타무비트["베타"])["차이"], {}),
    "④ D 포함":         (주묶음["차이"], {"사건들": list(사건창["이름"])}),
    "⑤ 창 [0,0]":       (주묶음["차이"], {"창": (0, 0)}),
    "⑥ 추정 기간 120일":  (묶음AR(AR120, 베타120["베타"], 기간이름="강건성(120일)")["차이"], {"기간이름": "강건성(120일)"}),
    "⑦ t값으로 줄 세우기": (묶음AR(AR주, 베타주["t"])["차이"], {}),
}
주S = d["S_%"]
행 = []
for 이름, (ar, 옵션) in 경우들.items():
    v = 판정값(ar, **옵션)
    행.append({"경우": 이름, **v, "주결과와_같은부호": np.sign(v["S_%"]) == np.sign(주S),
               "주결과대비_크기": v["S_%"] / 주S, "5%유의": v["p"] < 유의})
강건성 = pd.DataFrame(행).set_index("경우")
print("\n[강건성] 차이 묶음 S (%)")
print(강건성.drop(columns="자유도").to_string(float_format=f))


#%% [셀 4] 그림 ============================================================
k = stats.t.ppf(1 - 유의 / 2, int(판정["자유도"].iloc[0]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(순서))
for xi, 대상 in zip(x, 순서):
    ax[0].errorbar(xi, 판정.loc[대상, "S_%"], yerr=k * 판정.loc[대상, "SE_%"], fmt="o",
                   color="black" if 대상 == "차이" else "steelblue", capsize=3)
ax[0].axhline(0, color="gray", lw=0.8)
ax[0].set_xticks(x)
ax[0].set_xticklabels(["1\n(유입베타 작음)", "2", "3", "4", "5\n(유입베타 큼)", "차이\n(5 - 1)"], fontsize=8)
ax[0].set_ylabel("S (%), 막대 = 95% 범위")
ax[0].set_title("묶음별 사건당 평균 반응 (한국)", loc="left", fontsize=11)
차 = 사건표[사건표["대상"] == "차이"].reset_index(drop=True)
값 = 차["방향"] * 차["CAR_%"]
위치 = np.arange(len(차))
ax[1].bar(위치, 값, yerr=k * 차["SE_%"], color=["indianred" if v < 0 else "seagreen" for v in 값], capsize=3)
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_xticks(위치)
ax[1].set_xticklabels([f"{s}\n(부호 뒤집음)" if s == "A" else s for s in 차["사건"]], fontsize=8)
ax[1].set_ylabel("방향 × CAR (%), 막대 = 95% 범위")
ax[1].set_title("사건별 차이 (묶음5 - 묶음1)", loc="left", fontsize=11)
fig.suptitle("유입 베타 묶음별 GENIUS Act 반응 - 한국", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "유입베타_한국_그림1_묶음별반응.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(강건성))[::-1]
for yy, (이름, r) in zip(y, 강건성.iterrows()):
    kk = stats.t.ppf(1 - 유의 / 2, int(r["자유도"]))
    ax.errorbar(r["S_%"], yy, xerr=kk * r["SE_%"], fmt="o", color="black" if 이름.startswith("주") else "steelblue", capsize=3)
ax.axvline(0, color="gray", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(강건성.index, fontsize=8)
ax.set_xlabel("차이 묶음 S (%), 가로 막대 = 95% 범위")
ax.set_title("유입 베타 연구 강건성 - 한국 (검은 점 = 주 결과)", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "유입베타_한국_그림2_강건성.png", dpi=150)
plt.show()


#%% [셀 5] 저장 ============================================================
판정["순위상관_보조"] = np.nan
판정.loc["차이", "순위상관_보조"] = ρ
사건표.to_csv(데이터폴더 / "유입베타_한국_사건별.csv", index=False, encoding=CSV인코딩)
판정.to_csv(데이터폴더 / "유입베타_한국_판정.csv", encoding=CSV인코딩)
강건성.to_csv(데이터폴더 / "유입베타_한국_강건성.csv", encoding=CSV인코딩)
print("저장: 유입베타_한국_사건별.csv, 유입베타_한국_판정.csv, 유입베타_한국_강건성.csv, 그림 2개")
