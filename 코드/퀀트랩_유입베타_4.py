# -*- coding: utf-8 -*-
"""
퀀트랩_유입베타_4.py
=====================================================================
[연구] 유입 베타 연구: 사건 전 '스테이블코인 유입과 같이 움직이던 정도'로 고른 주식이
       GENIUS Act 소식에 더 크게 반응했는가
[단계] 4단계: 강건성 - 미리 정해 둔 7가지로 바꿔 봐도 3단계 결과(부호·크기)가 비슷한가

[이 코드가 하는 일]
  설계 "강건성목록" 7가지를 그대로 돌립니다. 판정 대상은 차이 묶음(맨 위 묶음 − 맨 아래 묶음)의 S.
    ① 묶음 10개             : 10개로 나누고 차이 = 묶음10 − 묶음1
    ② 시가총액 가중          : 묶음 안을 시가총액 가중 (가중치 = 묶음을 나누는 날 2025-04-23 의 시가총액, 고정)
    ③ 비트코인 없이 베타      : 유입 베타를 R = a + b·시장 + d·유입 으로 잼 (사건 반응 모형은 그대로 시장 + IBIT)
    ④ D 포함 (A~E)          : 3부와 같은 다섯 사건
    ⑤ 창 [0,0]              : 0일 하루만
    ⑥ 추정 기간 120일        : 유입 베타와 사건 반응 모형을 모두 같은 끝날의 120 거래일로 추정
    ⑦ t값으로 줄 세우기       : 유입 베타 대신 유입 베타의 t값 순서로 5개 묶음
  설계 "강건성_판정": 통과 기준 없음. 부호와 크기가 주 결과와 같은지만 기록 -> 판정은 바뀌지 않음.

[근거]
  유입 베타·묶음·S 계산: 퀀트랩_유입베타_2, _3 과 같은 계산 (MacKinlay 1997, Ang 외 2006)
  ② 가중치 날짜·⑥ 두 추정을 함께 120일로 하는 것은 이 코드에서 정한 방식 (설계 "묶음" 이 묶음을 그날 고정하므로 같은 날로 맞춤)

[입력] 유입베타_설계.csv, 유입베타_수익률_일별.csv, 유입베타_시가총액_일별.csv, 유입베타_요인_일별.csv (1단계),
       유입베타_3단계_판정.csv (3단계), 사건연구_추정기간.csv, 사건연구_사건창.csv
[출력] 유입베타_4단계_강건성.csv, 그림/유입베타_그림2_강건성.png
[Spyder] "#%%" 셀 단위로 Shift+Enter (종목별 회귀를 여러 번 돌려 [셀 2]가 1~2분 걸릴 수 있음)
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
수익률 = pd.read_csv(데이터폴더 / "유입베타_수익률_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
시총 = pd.read_csv(데이터폴더 / "유입베타_시가총액_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
요인 = pd.read_csv(데이터폴더 / "유입베타_요인_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
주결과 = pd.read_csv(데이터폴더 / "유입베타_3단계_판정.csv", index_col="대상", encoding=CSV인코딩)
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)

판정사건 = 설계["판정사건"].split(",")                      # A,B,C,E
유의 = float(설계["유의수준"])
print("강건성 목록:", 설계["강건성목록"])


#%% [셀 1] 계산 함수 (2·3단계와 같은 계산) ==================================
def 유입베타(기간이름="주(250일)", 비트코인=True):
    """종목별 유입 베타와 t값 (퀀트랩_유입베타_2 [셀 1] 과 같음). 비트코인=False 면 IBIT 를 빼고 잼"""
    기간 = 추정기간.loc[기간이름]
    열 = ["시장", "IBIT", "유입_B"] if 비트코인 else ["시장", "유입_B"]
    X전체 = 요인.loc[기간["시작"]:기간["끝"], 열].assign(상수=1.0)[["상수"] + 열].to_numpy()
    Y = 수익률.loc[기간["시작"]:기간["끝"]]
    행 = {}
    for p in Y.columns:
        y = Y[p].to_numpy()
        있음 = ~np.isnan(y)
        X, yv = X전체[있음], y[있음]
        계수, *_ = np.linalg.lstsq(X, yv, rcond=None)
        잔차 = yv - X @ 계수
        σ2 = 잔차 @ 잔차 / (len(yv) - X.shape[1])
        SE = np.sqrt(σ2 * np.linalg.inv(X.T @ X)[-1, -1])
        행[p] = (계수[-1], 계수[-1] / SE)                         # 마지막 계수 = 유입
    return pd.DataFrame(행, index=["베타", "t"]).T


def 차이수익률(점수, 묶음수=5, 가중="동일"):
    """점수 순서로 묶음수 개 묶음 -> (맨 위 − 맨 아래) 일별 수익률 (퀀트랩_유입베타_2 [셀 3] 과 같은 방식)"""
    번호 = pd.qcut(점수, 묶음수, labels=False) + 1
    out = {}
    for g in [1, 묶음수]:
        멤버 = 번호.index[번호 == g]
        R = 수익률[멤버]
        if 가중 == "동일":
            out[g] = R.mean(axis=1)
        else:                                                    # 시가총액 가중 (묶음 나누는 날 시가총액으로 고정)
            w = 시총.loc[pd.Timestamp(추정기간.loc["주(250일)", "끝"]), 멤버]
            W = R.notna().mul(w, axis=1)                         # 그날 수익률 있는 종목만 가중치를 나눠 가짐
            out[g] = (R.fillna(0) * W).sum(axis=1) / W.sum(axis=1)
    return out[묶음수] - out[1]


def 판정값(r, 사건들=판정사건, 기간이름="주(250일)", 창=None):
    """차이 수익률 r 의 S·t·p (퀀트랩_유입베타_3 [셀 1] 과 같은 계산)"""
    표 = 요인[["시장", "IBIT"]].assign(r=r)
    기간 = 추정기간.loc[기간이름]
    추정 = 표.loc[기간["시작"]:기간["끝"]]
    X = np.column_stack([np.ones(len(추정)), 추정["시장"], 추정["IBIT"]])
    계수, *_ = np.linalg.lstsq(X, 추정["r"].to_numpy(), rcond=None)
    잔차 = 추정["r"].to_numpy() - X @ 계수
    자유도 = len(잔차) - 3
    σ = np.sqrt(잔차 @ 잔차 / 자유도)
    거래일 = 표.index
    값, 분산 = [], []
    for _, e in 사건창[사건창["이름"].isin(사건들)].iterrows():
        시작, 끝 = pd.Timestamp(e["창시작"]), pd.Timestamp(e["창끝"])
        if 창 is not None:
            k0, k1 = 거래일.get_loc(시작), 거래일.get_loc(끝) - 1   # 첫 0일, 마지막 0일
            시작, 끝 = 거래일[k0 + 창[0]], 거래일[k1 + 창[1]]
        구간 = 표.loc[시작:끝]
        AR = 구간["r"].to_numpy() - np.column_stack([np.ones(len(구간)), 구간["시장"], 구간["IBIT"]]) @ 계수
        값.append(int(e["방향"]) * AR.sum() * 100)
        분산.append((σ * 100) ** 2 * len(구간))
    S, SE = np.mean(값), np.sqrt(np.sum(분산)) / len(값)
    return {"S_%": S, "SE_%": SE, "t": S / SE, "p": 2 * stats.t.sf(abs(S / SE), 자유도), "최소검출효과_%": 2.8 * SE,
            "자유도": 자유도}


#%% [셀 2] 주 결과 + 강건성 7가지 ===========================================
베타주 = 유입베타()
베타120 = 유입베타("강건성(120일)")
베타무비트 = 유입베타(비트코인=False)

경우들 = {
    "주 결과 (5묶음, 동일가중, A,B,C,E, [0,+1])": (차이수익률(베타주["베타"]), {}),
    "① 묶음 10개":              (차이수익률(베타주["베타"], 묶음수=10), {}),
    "② 시가총액 가중":           (차이수익률(베타주["베타"], 가중="시총"), {}),
    "③ 비트코인 없이 베타":       (차이수익률(베타무비트["베타"]), {}),
    "④ D 포함 (A~E)":           (차이수익률(베타주["베타"]), {"사건들": list(사건창["이름"])}),
    "⑤ 창 [0,0]":               (차이수익률(베타주["베타"]), {"창": (0, 0)}),
    "⑥ 추정 기간 120일":         (차이수익률(베타120["베타"]), {"기간이름": "강건성(120일)"}),
    "⑦ t값으로 줄 세우기":        (차이수익률(베타주["t"]), {}),
}
주S = 주결과.loc["차이", "S_%"]
행 = []
for 이름, (r, 옵션) in 경우들.items():
    v = 판정값(r, **옵션)
    행.append({"경우": 이름, **v, "주결과와_같은부호": np.sign(v["S_%"]) == np.sign(주S),
               "주결과대비_크기": v["S_%"] / 주S, "5%유의": v["p"] < 유의})
강건성 = pd.DataFrame(행).set_index("경우")
assert np.isclose(강건성.iloc[0]["S_%"], 주S), "주 결과가 3단계와 다릅니다 (계산 방식 점검)"

# 묶음을 다르게 나눈 경우, 주 결과와 맨 위·맨 아래 묶음 종목이 얼마나 겹치는지 (참고)
def 끝묶음(점수, n=5):
    번호 = pd.qcut(점수, n, labels=False) + 1
    return set(번호.index[번호 == n]), set(번호.index[번호 == 1])
위주, 아래주 = 끝묶음(베타주["베타"])
for 이름, 점수 in [("③ 비트코인 없이 베타", 베타무비트["베타"]), ("⑥ 추정 기간 120일", 베타120["베타"]),
                   ("⑦ t값으로 줄 세우기", 베타주["t"])]:
    위, 아래 = 끝묶음(점수)
    print(f"  {이름}: 주 결과와 맨 위 묶음 {len(위 & 위주) / len(위주):.0%}, 맨 아래 묶음 {len(아래 & 아래주) / len(아래주):.0%} 겹침")

pd.set_option("display.width", 240)
print("\n[4단계] 강건성 - 차이 묶음(맨 위 − 맨 아래)의 S (%), 법이 한 걸음 나아갈 때 사건당 평균")
print(강건성.drop(columns="자유도").to_string(float_format=lambda v: f"{v:+.3f}"))


#%% [셀 3] 그림 ============================================================
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(강건성))[::-1]
for yy, (이름, r) in zip(y, 강건성.iterrows()):
    k = stats.t.ppf(1 - 유의 / 2, int(r["자유도"]))
    c = "black" if 이름.startswith("주") else "steelblue"
    ax.errorbar(r["S_%"], yy, xerr=k * r["SE_%"], fmt="o", color=c, capsize=3)
ax.axvline(0, color="gray", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(강건성.index, fontsize=8)
ax.set_xlabel("차이 묶음 S (%), 가로 막대 = 95% 범위")
ax.set_title("유입 베타 연구 강건성: 검은 점 = 주 결과", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "유입베타_그림2_강건성.png", dpi=150)
plt.show()


#%% [셀 4] 저장 ============================================================
강건성.to_csv(데이터폴더 / "유입베타_4단계_강건성.csv", encoding=CSV인코딩)
print("저장: 유입베타_4단계_강건성.csv, 그림/유입베타_그림2_강건성.png")
