# -*- coding: utf-8 -*-
"""
퀀트랩_사건연구_4.py
=====================================================================
[연구] 사건 연구: 미국 스테이블코인법(GENIUS Act)에 대한 주식시장 반응
[단계] 4단계: 강건성 (설계 "강건성목록" 그대로) + CRCL 참고

[이 코드가 하는 일]
  ① 3단계 주 판정을 미리 정한 네 가지로 바꿔 다시 계산
       시장 모형(시장만) / 추정 기간 120일 / 창 [0,0] / 창 [−1,+1]
     통과 기준은 없음. 부호와 크기가 주 결과와 같은지만 기록 (설계 "강건성_판정")
  ② CRCL: 시장 조정 수익률(R_CRCL − R_시장), 사건 C·D·E 만, 참고 보고 (설계 "참고종목방법")
     추정 기간이 없어 평소 잡음의 크기를 알 수 없으므로 t값은 내지 않음

[창을 바꿀 때 묶은 사건(B, E) 처리] (출처: 설계 "묶음규칙"을 그대로 적용)
  묶은 창 = 첫 0일 + 앞 ~ 마지막 0일 + 뒤
  주 창 [0,+1] 이면 1단계 사건창과 같음. [0,0] 이면 B = 5/20~5/21, E = 7/17~7/18
  (마지막 0일 = 1단계 창끝의 하루 전 거래일)

[입력] 사건연구_설계.csv, 사건연구_수익률_일별.csv, 사건연구_사건창.csv, 사건연구_추정기간.csv,
       사건연구_3단계_판정.csv
[출력] 사건연구_4단계_강건성.csv, 사건연구_4단계_CRCL참고.csv, 그림/사건연구_그림2_강건성.png
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

설계 = pd.read_csv(데이터폴더 / "사건연구_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
수익률 = pd.read_csv(데이터폴더 / "사건연구_수익률_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")
주판정 = pd.read_csv(데이터폴더 / "사건연구_3단계_판정.csv", encoding=CSV인코딩).set_index("그룹")

그룹들 = ["발행사", "거래소", "은행", "카드"]
비트 = 설계["비트코인요인"]
거래일 = 수익률.index


def 창날짜(e, 앞, 뒤):
    """묶음규칙 그대로: 첫 0일 + 앞 ~ 마지막 0일 + 뒤 (마지막 0일 = 1단계 창끝의 하루 전 거래일)"""
    k0 = 거래일.get_loc(pd.Timestamp(e["창시작"]))
    k1 = 거래일.get_loc(pd.Timestamp(e["창끝"])) - 1
    return 거래일[k0 + 앞], 거래일[k1 + 뒤]


#%% [셀 1] 사건 연구 함수 (3단계와 같은 계산, 창만 바꿀 수 있게) ==========
def 사건연구(g, 기간이름="주(250일)", X열=("시장", 비트), 앞=0, 뒤=1):
    기간 = 추정기간.loc[기간이름]
    추정 = 수익률.loc[기간["시작"]:기간["끝"]]
    X = np.column_stack([np.ones(len(추정))] + [추정[c].to_numpy() for c in X열])
    y = 추정[g].to_numpy()
    계수, *_ = np.linalg.lstsq(X, y, rcond=None)
    잔차 = y - X @ 계수
    자유도 = len(y) - X.shape[1]
    σ = np.sqrt(잔차 @ 잔차 / 자유도)
    CAR, SE, 방향 = [], [], []
    for _, e in 사건창.iterrows():
        시작, 끝 = 창날짜(e, 앞, 뒤)
        assert 시작 > pd.Timestamp(기간["끝"]), "창이 추정 기간과 겹칩니다"
        구간 = 수익률.loc[시작:끝]
        예측 = np.column_stack([np.ones(len(구간))] + [구간[c].to_numpy() for c in X열]) @ 계수
        CAR.append((구간[g].to_numpy() - 예측).sum() * 100)
        SE.append(σ * np.sqrt(len(구간)) * 100)
        방향.append(int(e["방향"]))
    S = np.mean(np.array(방향) * np.array(CAR))                        # 설계 "주판정값"
    SE_S = np.sqrt(np.sum(np.square(SE))) / len(CAR)
    t = S / SE_S
    return {"S_%": S, "SE_%": SE_S, "t": t, "p": 2 * stats.t.sf(abs(t), 자유도)}


# 주 결과가 3단계와 같은지 먼저 확인 (코드 점검용)
for g in 그룹들:
    assert np.isclose(사건연구(g)["S_%"], 주판정.loc[g, "S_%"]), f"{g}: 3단계와 다름"


#%% [셀 2] 강건성 =========================================================
# 출처: 설계 "강건성목록" - 시장 모형 / 추정 기간 120일 / 창 [0,0] / 창 [−1,+1]
점검들 = {"주 결과": {},
          "① 시장 모형": {"X열": ("시장",)},
          "② 추정 기간 120일": {"기간이름": "강건성(120일)"},
          "③ 창 [0,0]": {"앞": 0, "뒤": 0},
          "④ 창 [-1,+1]": {"앞": -1, "뒤": 1}}
행 = []
for g in 그룹들:
    주S = 주판정.loc[g, "S_%"]
    for 이름, 설정 in 점검들.items():
        r = 사건연구(g, **설정)
        행.append({"그룹": g, "점검": 이름, **r,
                   "부호같음": bool(np.sign(r["S_%"]) == np.sign(주S)),
                   "크기비_주=1": r["S_%"] / 주S})
강건 = pd.DataFrame(행)

pd.set_option("display.width", 220)
f = lambda v: f"{v:.3f}"
print("[4단계] 강건성 - 주판정값 S (사건당 평균 반응, %) / t / p")
for g in 그룹들:
    print(f"\n< {g} >")
    print(강건[강건["그룹"] == g].drop(columns="그룹").to_string(index=False, float_format=f))


#%% [셀 3] CRCL 참고 (시장 조정 수익률, 사건 C·D·E) ======================
# 출처: 설계 "참고종목방법" - 추정 기간이 없어 모형 대신 시장만 뺌, 판정에 안 씀
참고 = 설계["참고종목"]
행 = []
for _, e in 사건창.iterrows():
    시작, 끝 = 창날짜(e, 0, 1)
    구간 = 수익률.loc[시작:끝, [참고, "시장"]]
    if 구간[참고].isna().any():                     # 상장 전 (A, B)
        continue
    행.append({"사건": e["이름"], "창시작": 시작.date(), "창끝": 끝.date(),
               "시장조정CAR_%": (구간[참고] - 구간["시장"]).sum() * 100})
CRCL표 = pd.DataFrame(행)
print(f"\n[참고] {참고} 시장 조정 CAR (상장 2025-06-05 이후 사건만, 판정·t값 없음)")
print(CRCL표.to_string(index=False, float_format=f))


#%% [셀 4] 그림 + 저장 ====================================================
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False)
for ax, g in zip(axes, 그룹들):
    d = 강건[강건["그룹"] == g].reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    ax.errorbar(d["S_%"], y, xerr=1.96 * d["SE_%"], fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["점검"] if g == 그룹들[0] else [""] * len(d), fontsize=9)
    ax.set_title(g, fontsize=11)
    ax.set_xlabel("S (%), 95% 구간")
fig.suptitle("강건성: 사건당 평균 반응 S", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(그림폴더 / "사건연구_그림2_강건성.png", dpi=150)
plt.show()

강건.to_csv(데이터폴더 / "사건연구_4단계_강건성.csv", index=False, encoding=CSV인코딩)
CRCL표.to_csv(데이터폴더 / "사건연구_4단계_CRCL참고.csv", index=False, encoding=CSV인코딩)
print("저장: 사건연구_4단계_강건성.csv, 사건연구_4단계_CRCL참고.csv, 그림/사건연구_그림2_강건성.png")
