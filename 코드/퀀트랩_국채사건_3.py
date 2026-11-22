# -*- coding: utf-8 -*-
"""
퀀트랩_국채사건_3.py
=====================================================================
[연구] 국채 사건 연구: GENIUS Act 에 단기국채 스프레드와 스테이블코인 유입이 어떻게 반응했는가
[단계] 3단계: 본 분석
  1층 (기술적 비교, 판정은 재현 여부만)
    [셀 1] 원 논문 그림 6 재현 - 논문 사건 4개, ±15 평일, 사건 전날 = 0
    [셀 2] 같은 방식으로 GENIUS 사건 A~E 를 그림 (창이 겹쳐 판정에 안 씀)
  2층 (판정용)
    [셀 3] 짧은 창 [0,+1] 검정 - 판정 사건 A·B·C·E, 평균 조정, 판정과 해석표

[근거]
  그림 6 방식: 원 논문 3.2절, 그림 6 주석 / 재현 판정 ⓐⓑⓒ: 설계 "재현_판정"
  평균 조정·t값: MacKinlay (1997) / 판정 기준·해석표: 설계 "판정_", "해석표"
  p값: t분포, 자유도 = 추정 기간 하루 변화 개수 − 1 (평균 하나만 추정)

[입력] 국채사건_설계.csv, 국채사건_자료_평일.csv, 국채사건_재현사건.csv (1단계),
       사건연구_사건창.csv, 사건연구_추정기간.csv
[출력] 국채사건_3단계_재현경로.csv, 국채사건_3단계_GENIUS경로.csv,
       국채사건_3단계_사건별.csv, 국채사건_3단계_판정.csv,
       그림/국채사건_그림1_재현.png, 그림/국채사건_그림2_GENIUS.png
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
그림폴더.mkdir(parents=True, exist_ok=True)
CSV인코딩 = "utf-8-sig"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

설계 = pd.read_csv(데이터폴더 / "국채사건_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
자료 = pd.read_csv(데이터폴더 / "국채사건_자료_평일.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
재현 = pd.read_csv(데이터폴더 / "국채사건_재현사건.csv", encoding=CSV인코딩)
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")

거래일 = 자료.index
앞, 뒤 = map(int, 설계["재현_기간"].split(","))                # −15, +15
기준 = int(설계["재현_기준일"])                                 # −1 (사건 전날 = 0)
k축 = np.arange(앞, 뒤 + 1)


def 경로(영일, 방향):
    """사건 0일 기준 ±15 평일의 누적 변화 (전날 = 0), 방향을 곱해 유입 사건으로 맞춤"""
    i = 거래일.get_loc(pd.Timestamp(영일))
    구간 = 자료.iloc[i + 앞: i + 뒤 + 1][["시총_B", "스프레드_bp"]].to_numpy()
    전날 = 자료.iloc[i + 기준][["시총_B", "스프레드_bp"]].to_numpy()
    d = (구간 - 전날) * 방향
    return pd.DataFrame({"k": k축, "유입_B": d[:, 0], "스프레드_bp": d[:, 1]})


#%% [셀 1] 1층: 원 논문 그림 6 재현 ========================================
# 출처: 원 논문 3.2절, 그림 6 (사건 4개, 유출 사건은 부호를 뒤집어 유입으로 맞춤)
재현경로 = pd.concat([경로(r["0일"], int(r["방향"])).assign(사건=r["사건"]) for _, r in 재현.iterrows()],
                     ignore_index=True)
재현평균 = 재현경로.groupby("k")[["유입_B", "스프레드_bp"]].mean()

# 재현 판정 (설계 "재현_판정"): ⓐ +15 유입 > 0  ⓑ +15 스프레드 < 0  ⓒ |−15 값| < |+15 값| (각각)
끝, 처음 = 재현평균.loc[뒤], 재현평균.loc[앞]
판정a = 끝["유입_B"] > 0
판정b = 끝["스프레드_bp"] < 0
판정c = (abs(처음["유입_B"]) < abs(끝["유입_B"])) and (abs(처음["스프레드_bp"]) < abs(끝["스프레드_bp"]))
print("[1층] 원 논문 그림 6 재현 (평균선, 사건 전날 = 0)")
print(f"  +15일 누적 유입 {끝['유입_B']:+.2f} $B -> ⓐ {'충족' if 판정a else '불충족'}")
print(f"  +15일 스프레드 변화 {끝['스프레드_bp']:+.2f} bp -> ⓑ {'충족' if 판정b else '불충족'}")
print(f"  사건 전(−15일) 유입 {처음['유입_B']:+.2f} $B, 스프레드 {처음['스프레드_bp']:+.2f} bp -> ⓒ {'충족' if 판정c else '불충족'}")
print(f"  => 그림 6 {'재현됨' if (판정a and 판정b and 판정c) else '재현 안 됨'}")
print("\n사건별 +15일 값")
print(재현경로[재현경로["k"] == 뒤][["사건", "유입_B", "스프레드_bp"]].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for 사건, d in 재현경로.groupby("사건", sort=False):
    방향 = int(재현.set_index("사건").loc[사건, "방향"])
    ls = "--" if 방향 < 0 else "-"                                  # 원 논문처럼 유출 사건은 점선
    ax[0].plot(d["k"], d["유입_B"], ls, lw=0.9, alpha=0.6, label=사건)
    ax[1].plot(d["k"], d["스프레드_bp"] / 100, ls, lw=0.9, alpha=0.6)
ax[0].plot(재현평균.index, 재현평균["유입_B"], color="navy", lw=2.4, label="평균")
ax[1].plot(재현평균.index, 재현평균["스프레드_bp"] / 100, color="navy", lw=2.4)
for a, t, y in [(ax[0], "스테이블코인 유입 ($B)", "$B"), (ax[1], "3개월-6개월 스프레드 (%)", "%p")]:
    a.axvline(0, color="gray", lw=0.8); a.axhline(0, color="black", lw=0.8)
    a.set_title(t, loc="left", fontsize=11); a.set_xlabel("사건 전후 평일")
ax[0].legend(fontsize=8)
fig.suptitle("원 논문 그림 6 재현 (유출 사건은 부호를 뒤집음, 사건 전날 = 0)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "국채사건_그림1_재현.png", dpi=150)
plt.show()


#%% [셀 2] 1층: GENIUS 사건을 그림 6 방식으로 (기술적 비교) ================
# 출처: 설계 "그림6_GENIUS", "그림6_D표시" - 창이 겹쳐 사건별 구분이 안 되므로 판정에 안 씀
G경로 = pd.concat([경로(e["창시작"], int(e["방향"])).assign(사건=e["이름"]) for _, e in 사건창.iterrows()],
                  ignore_index=True)
G평균 = G경로.groupby("k")[["유입_B", "스프레드_bp"]].mean()
print("\n[1층] GENIUS 사건 (그림 6 방식, 기술적 비교) +15일 값")
print(G경로[G경로["k"] == 뒤][["사건", "유입_B", "스프레드_bp"]].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
print(f"  평균: 유입 {G평균.loc[뒤, '유입_B']:+.2f} $B, 스프레드 {G평균.loc[뒤, '스프레드_bp']:+.2f} bp "
      f"(참고: 논문 사건 평균 {끝['유입_B']:+.2f} $B, {끝['스프레드_bp']:+.2f} bp)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for 사건, d in G경로.groupby("사건", sort=False):
    이름 = f"{사건} (FOMC 겹침)" if 사건 == "D" else (f"{사건} (부결, 부호 뒤집음)" if 사건 == "A" else 사건)
    ax[0].plot(d["k"], d["유입_B"], lw=0.9, alpha=0.6, ls=":" if 사건 == "D" else "-", label=이름)
    ax[1].plot(d["k"], d["스프레드_bp"] / 100, lw=0.9, alpha=0.6, ls=":" if 사건 == "D" else "-")
ax[0].plot(G평균.index, G평균["유입_B"], color="darkred", lw=2.4, label="GENIUS 평균")
ax[1].plot(G평균.index, G평균["스프레드_bp"] / 100, color="darkred", lw=2.4)
ax[0].plot(재현평균.index, 재현평균["유입_B"], color="navy", lw=1.6, ls="--", label="논문 사건 평균")
ax[1].plot(재현평균.index, 재현평균["스프레드_bp"] / 100, color="navy", lw=1.6, ls="--")
for a, t in [(ax[0], "스테이블코인 유입 ($B)"), (ax[1], "3개월-6개월 스프레드 (%)")]:
    a.axvline(0, color="gray", lw=0.8); a.axhline(0, color="black", lw=0.8)
    a.set_title(t, loc="left", fontsize=11); a.set_xlabel("사건 전후 평일")
ax[0].legend(fontsize=7)
fig.suptitle("GENIUS Act 사건을 그림 6 방식으로 (기술적 비교, 사건 창이 서로 겹침)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "국채사건_그림2_GENIUS.png", dpi=150)
plt.show()


#%% [셀 3] 2층: 짧은 창 검정 + 판정 ========================================
# 출처: 설계 "사건창", "스프레드변화", "유입변화", "평균조정", "검정", "판정값", "판정_", "해석표"
주 = 추정기간.loc["주(250일)"]
하루변화 = 자료[["스프레드_bp", "시총_B", "3개월_bp"]].diff().loc[주["시작"]:주["끝"]].dropna()
μ, σ = 하루변화.mean(), 하루변화.std(ddof=1)
자유도 = len(하루변화) - 1
유의 = float(설계["유의수준"])
판정사건 = 설계["판정사건"].split(",")

행 = []
for _, e in 사건창.iterrows():
    시작, 끝날 = pd.Timestamp(e["창시작"]), pd.Timestamp(e["창끝"])
    i0, i1 = 거래일.get_loc(시작), 거래일.get_loc(끝날)
    L = i1 - i0 + 1                                                   # 창 길이 (국채 거래일)
    for 열, 이름 in [("스프레드_bp", "스프레드"), ("시총_B", "유입"), ("3개월_bp", "3개월금리")]:
        변화 = 자료[열].iloc[i1] - 자료[열].iloc[i0 - 1]               # 창 끝 − 창 시작 전날
        비정상 = 변화 - μ[열] * L                                     # 평균 조정
        SE = σ[열] * np.sqrt(L)
        행.append({"사건": e["이름"], "방향": int(e["방향"]), "대상": 이름, "창길이": L,
                   "변화": 변화, "비정상변화": 비정상, "SE": SE, "t": 비정상 / SE,
                   "p": 2 * stats.t.sf(abs(비정상 / SE), 자유도)})
사건별 = pd.DataFrame(행)


def 판정값(d):
    """S = 판정 사건의 (방향 × 비정상 변화) 평균, 창이 겹치지 않으므로 분산을 더함"""
    S = (d["방향"] * d["비정상변화"]).mean()
    SE = np.sqrt((d["SE"] ** 2).sum()) / len(d)
    t = S / SE
    return S, SE, t, 2 * stats.t.sf(abs(t), 자유도)


판정행 = []
for 이름 in ["스프레드", "유입", "3개월금리"]:
    d = 사건별[(사건별["대상"] == 이름) & 사건별["사건"].isin(판정사건)]
    S, SE, t, p = 판정값(d)
    판정행.append({"대상": 이름, "S": S, "SE": SE, "t": t, "p": p, "최소검출효과": 2.8 * SE})
판정 = pd.DataFrame(판정행).set_index("대상")

# 판정 (설계 "판정_스프레드", "판정_유입")
sp, fl = 판정.loc["스프레드"], 판정.loc["유입"]
스프레드결과 = ("좁아짐" if sp["S"] < 0 else "넓어짐") if sp["p"] < 유의 else "유의하지 않음"
유입결과 = ("늘어남" if fl["S"] > 0 else "줄어듦") if fl["p"] < 유의 else "유의하지 않음"
판정["판정"] = ["성립" if 스프레드결과 == "좁아짐" else "성립 안 함", f"기록: {유입결과}", "강건성용 (4단계)"]

# 해석표 (설계 "해석표" - 미리 고정한 네 칸)
해석 = {("유의하지 않음", "유의하지 않음"): "논문이 이 법을 뺀 판단과 맞음",
        ("유의하지 않음", "좁아짐"): "돈보다 금리가 먼저 반응 -> 시장이 앞으로의 단기국채 수요를 미리 반영했을 가능성",
        ("늘어남", "좁아짐"): "법안도 다른 사건처럼 유입 충격으로 작동 -> 논문의 판단과 다름",
        ("늘어남", "유의하지 않음"): "유입이 금리로 이어지지 않음 -> 1부 결과와 어긋남"}.get(
    (유입결과, 스프레드결과), "미리 정한 네 칸 밖의 결과 -> 그대로 기록하고 해석은 따로")

pd.set_option("display.width", 220)
f = lambda v: f"{v:+.3f}"
print("\n[2층] 사건별 비정상 변화 (창 [0,+1], 평균 조정; 스프레드·3개월 bp, 유입 $B)")
print(사건별.pivot(index="사건", columns="대상", values="비정상변화")[["스프레드", "유입", "3개월금리"]].to_string(float_format=f))
print("\n사건별 t값")
print(사건별.pivot(index="사건", columns="대상", values="t")[["스프레드", "유입", "3개월금리"]].to_string(float_format=f))
print(f"\n[판정] 판정 사건 {판정사건} (D 는 FOMC 와 겹쳐 뺌), S = 법이 한 걸음 나아갈 때의 평균 반응")
print(판정.to_string(float_format=f))
print(f"\n[해석표] 유입 {유입결과} / 스프레드 {스프레드결과} -> {해석}")


#%% [셀 4] 저장 ===========================================================
재현경로.to_csv(데이터폴더 / "국채사건_3단계_재현경로.csv", index=False, encoding=CSV인코딩)
G경로.to_csv(데이터폴더 / "국채사건_3단계_GENIUS경로.csv", index=False, encoding=CSV인코딩)
사건별.to_csv(데이터폴더 / "국채사건_3단계_사건별.csv", index=False, encoding=CSV인코딩)
판정.assign(해석=[해석, "", ""]).to_csv(데이터폴더 / "국채사건_3단계_판정.csv", encoding=CSV인코딩)
print("저장: 국채사건_3단계_재현경로.csv, _GENIUS경로.csv, _사건별.csv, _판정.csv, 그림 2개")
