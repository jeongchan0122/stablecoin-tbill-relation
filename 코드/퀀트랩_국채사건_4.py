# -*- coding: utf-8 -*-
"""
퀀트랩_국채사건_4.py
=====================================================================
[연구] 국채 사건 연구: GENIUS Act 에 단기국채 스프레드와 스테이블코인 유입이 어떻게 반응했는가
[단계] 4단계: 강건성 - 미리 정해 둔 4가지로 바꿔 봐도 3단계 결과(부호·크기)가 비슷한가

[이 코드가 하는 일]
  설계 "강건성목록" 4가지를 그대로 돌립니다 (결과를 보고 고르지 않도록 설계에서 미리 고정).
    ① D 포함      : 판정 사건 A·B·C·E 에 D 를 더함 (FOMC 와 겹친 사건)
    ② 창 [0,0]    : 0일 하루만
    ③ 창 [-1,+1]  : 0일 전날부터 다음 날까지
    ④ 3개월 금리  : 스프레드 대신 3개월 금리 변화 (6개월 금리가 같이 움직인 몫을 빼지 않은 값)
  설계 "강건성_판정": 통과 기준 없음. 부호와 크기가 주 결과와 같은지만 기록 -> 판정은 바뀌지 않음.
  각 경우의 창 안에 FOMC 성명일이 있는지도 함께 적어 둡니다 (설계 "D_FOMC확인" 과 같은 확인).

[근거]
  묶은 사건의 창 = 첫 0일 + 앞 ~ 마지막 0일 + 뒤 (퀀트랩_사건연구_4 와 같은 규칙,
                  마지막 0일 = 사건창.csv 창끝의 하루 전 거래일)
  평균 조정·t값: MacKinlay (1997) / S·해석: 설계 "판정값" (3단계와 같은 계산)

[입력] 국채사건_설계.csv, 국채사건_자료_평일.csv (1단계), 국채사건_3단계_판정.csv (3단계),
       사건연구_사건창.csv, 사건연구_추정기간.csv, 확장_통화정책충격_USMPD.xlsx
[출력] 국채사건_4단계_강건성.csv, 국채사건_4단계_사건별.csv, 그림/국채사건_그림3_강건성.png
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

설계 = pd.read_csv(데이터폴더 / "국채사건_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
자료 = pd.read_csv(데이터폴더 / "국채사건_자료_평일.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
주결과 = pd.read_csv(데이터폴더 / "국채사건_3단계_판정.csv", encoding=CSV인코딩, index_col="대상")
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")

거래일 = 자료.index
판정사건 = 설계["판정사건"].split(",")                          # A,B,C,E
유의 = float(설계["유의수준"])

# 추정 기간 하루 변화의 평균·표준편차 (3단계와 같음 - 설계 "추정기간", "평균조정")
주 = 추정기간.loc["주(250일)"]
하루변화 = 자료[["스프레드_bp", "시총_B", "3개월_bp"]].diff().loc[주["시작"]:주["끝"]].dropna()
μ, σ = 하루변화.mean(), 하루변화.std(ddof=1)
자유도 = len(하루변화) - 1

# FOMC 성명일 (출처: USMPD Statements 시트, 퀀트랩_국채사건_1 [셀 4] 와 같은 파일·같은 읽기)
U = pd.read_excel(데이터폴더 / "확장_통화정책충격_USMPD.xlsx", sheet_name="Statements")
FOMC = pd.to_datetime(U["Date"], errors="coerce").dt.normalize().dropna().drop_duplicates()


#%% [셀 1] 계산 함수 ======================================================
def 창(e, 앞, 뒤):
    """묶음 규칙: 첫 0일 + 앞 ~ 마지막 0일 + 뒤 (출처: 퀀트랩_사건연구_4 [셀 1])"""
    k0 = 거래일.get_loc(pd.Timestamp(e["창시작"]))
    k1 = 거래일.get_loc(pd.Timestamp(e["창끝"])) - 1               # 마지막 0일 (주 창이 [0,+1] 이므로)
    return k0 + 앞, k1 + 뒤


def 사건별계산(사건들, 앞, 뒤, 열, 이름):
    """사건마다 비정상 변화 = (창 끝 − 창 시작 전날) − μ × 창 길이, SE = σ × √창 길이 (3단계 [셀 3] 과 같음)"""
    행 = []
    for _, e in 사건창[사건창["이름"].isin(사건들)].iterrows():
        i0, i1 = 창(e, 앞, 뒤)
        L = i1 - i0 + 1
        변화 = 자료[열].iloc[i1] - 자료[열].iloc[i0 - 1]
        비정상 = 변화 - μ[열] * L
        SE = σ[열] * np.sqrt(L)
        겹침 = FOMC[(FOMC >= 거래일[i0]) & (FOMC <= 거래일[i1])]
        행.append({"사건": e["이름"], "방향": int(e["방향"]), "대상": 이름,
                   "창시작": 거래일[i0].date(), "창끝": 거래일[i1].date(), "창길이": L,
                   "비정상변화": 비정상, "SE": SE, "t": 비정상 / SE,
                   "창안_FOMC": ",".join(str(d.date()) for d in 겹침) or "없음"})
    d = pd.DataFrame(행)
    # 창이 서로 겹치면 분산을 그냥 더할 수 없음 -> 겹치지 않는지 확인
    순서 = d.sort_values("창시작")
    assert (pd.to_datetime(순서["창시작"]).iloc[1:].to_numpy() > pd.to_datetime(순서["창끝"]).iloc[:-1].to_numpy()).all(), \
        "사건 창이 서로 겹칩니다"
    return d


def 판정값(d):
    """S = (방향 × 비정상 변화) 평균, SE = √(SE² 합) ÷ 사건 수 (3단계 [셀 3] 과 같음)"""
    S = (d["방향"] * d["비정상변화"]).mean()
    SE = np.sqrt((d["SE"] ** 2).sum()) / len(d)
    t = S / SE
    return {"S": S, "SE": SE, "t": t, "p": 2 * stats.t.sf(abs(t), 자유도), "최소검출효과": 2.8 * SE}


#%% [셀 2] 주 결과 + 강건성 4가지 ===========================================
# 출처: 설계 "강건성목록" (순서 그대로). 대상 = (자료 열, 표시 이름, 주 결과에서 비교할 대상)
스프 = ("스프레드_bp", "스프레드", "스프레드")
유입 = ("시총_B", "유입", "유입")
경우들 = {
    "주 결과 (A,B,C,E / [0,+1])": dict(사건들=판정사건, 앞=0, 뒤=1, 대상들=[스프, 유입]),
    "① D 포함 (A~E)":            dict(사건들=list(사건창["이름"]), 앞=0, 뒤=1, 대상들=[스프, 유입]),
    "② 창 [0,0]":                 dict(사건들=판정사건, 앞=0, 뒤=0, 대상들=[스프, 유입]),
    "③ 창 [-1,+1]":               dict(사건들=판정사건, 앞=-1, 뒤=1, 대상들=[스프, 유입]),
    "④ 3개월 금리":               dict(사건들=판정사건, 앞=0, 뒤=1, 대상들=[("3개월_bp", "3개월금리", "스프레드")]),
}

결과행, 사건행 = [], []
for 경우, c in 경우들.items():
    for 열, 이름, 비교 in c["대상들"]:
        d = 사건별계산(c["사건들"], c["앞"], c["뒤"], 열, 이름).assign(경우=경우)
        사건행.append(d)
        r = 판정값(d)
        주S = 주결과.loc[비교, "S"]
        결과행.append({"경우": 경우, "대상": 이름, **r,
                       "주결과와_같은부호": np.sign(r["S"]) == np.sign(주S),
                       "주결과대비_크기": r["S"] / 주S,
                       "5%유의": r["p"] < 유의,
                       "창안_FOMC": ",".join(sorted({x for x in d["창안_FOMC"] if x != "없음"})) or "없음"})
강건성 = pd.DataFrame(결과행)
사건별 = pd.concat(사건행, ignore_index=True)

# 주 결과가 3단계와 똑같이 나오는지 확인 (계산 방식이 같다는 점검)
for 이름 in ["스프레드", "유입"]:
    s = 강건성[(강건성["경우"].str.startswith("주")) & (강건성["대상"] == 이름)]["S"].iloc[0]
    assert np.isclose(s, 주결과.loc[이름, "S"]), f"{이름}: 3단계 S 와 다름"

pd.set_option("display.width", 240)
f = lambda v: f"{v:+.3f}"
print("[4단계] 강건성 (스프레드·3개월 bp, 유입 $B / S = 법이 한 걸음 나아갈 때의 평균 반응)")
print("        ④ 의 '같은 부호'·'크기'는 주 결과 스프레드 S 와 비교")
for 이름 in ["스프레드", "3개월금리", "유입"]:
    t = 강건성[강건성["대상"] == 이름].drop(columns="대상").set_index("경우")
    print(f"\n<{이름}>")
    print(t.to_string(float_format=f))

print("\n사건별 창 날짜 (경우별)")
print(사건별[사건별["대상"] != "유입"][["경우", "사건", "창시작", "창끝", "창길이", "창안_FOMC"]]
      .to_string(index=False))


#%% [셀 3] 그림: 경우별 S 와 95% 범위 ======================================
# 95% 범위 = S ± t(0.975, 자유도) × SE
k = stats.t.ppf(1 - 유의 / 2, 자유도)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, 이름들, 단위 in [(ax[0], ["스프레드", "3개월금리"], "bp"), (ax[1], ["유입"], "$B")]:
    t = 강건성[강건성["대상"].isin(이름들)].reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    색 = ["black" if 경우.startswith("주") else "steelblue" for 경우 in t["경우"]]
    for S, SE, yy, c in zip(t["S"], t["SE"], y, 색):
        a.errorbar(S, yy, xerr=k * SE, fmt="o", color=c, capsize=3, lw=1)
    a.set_yticks(y)
    a.set_yticklabels(t["경우"], fontsize=8)
    a.axvline(0, color="gray", lw=0.8)
    a.set_xlabel(f"S ({단위}), 가로 막대 = 95% 범위")
ax[0].set_title("스프레드 (④ 는 3개월 금리)", loc="left", fontsize=11)
ax[1].set_title("스테이블코인 유입", loc="left", fontsize=11)
fig.suptitle("국채 사건 연구 강건성: 검은 점 = 주 결과", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "국채사건_그림3_강건성.png", dpi=150)
plt.show()


#%% [셀 4] 저장 ===========================================================
강건성.to_csv(데이터폴더 / "국채사건_4단계_강건성.csv", index=False, encoding=CSV인코딩)
사건별.to_csv(데이터폴더 / "국채사건_4단계_사건별.csv", index=False, encoding=CSV인코딩)
print("저장: 국채사건_4단계_강건성.csv, 국채사건_4단계_사건별.csv, 그림/국채사건_그림3_강건성.png")
