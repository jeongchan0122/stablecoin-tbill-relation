# -*- coding: utf-8 -*-
"""
퀀트랩_사건연구_3.py
=====================================================================
[연구] 사건 연구: 미국 스테이블코인법(GENIUS Act)에 대한 주식시장 반응
[단계] 3단계: 본 분석 - 주 판정

[이 코드가 하는 일]
  ① 그룹 포트폴리오마다 주 모형(시장 + IBIT)을 추정 기간 250일에 맞춤 (2단계와 같음)
  ② 사건 창 A~E 의 비정상 수익률(AR) = 실제 − 모형 예측, 창 안에서 더해 CAR
  ③ 주판정값 S = (−CAR_A + CAR_B + CAR_C + CAR_D + CAR_E) ÷ 5, t값, p값
  ④ 설계의 그룹별 기준으로 판정 + 기록만 하는 것(사건 A 부호, 사건별 CAR)

[근거]
  비정상 수익률·CAR·t값: MacKinlay (1997)  /  판정 기준: 설계 "판정_" 줄들 (퀀트랩_사건연구_0)
  p값: t분포, 자유도 = 추정 기간 일수 − 계수 3개

[입력] 사건연구_설계.csv, 사건연구_수익률_일별.csv, 사건연구_사건창.csv, 사건연구_추정기간.csv
[출력] 사건연구_3단계_사건별CAR.csv, 사건연구_3단계_판정.csv, 그림/사건연구_그림1_사건별CAR.png
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
plt.rcParams["font.family"] = "Malgun Gothic"        # 한글 글꼴 (윈도우)
plt.rcParams["axes.unicode_minus"] = False

설계 = pd.read_csv(데이터폴더 / "사건연구_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
수익률 = pd.read_csv(데이터폴더 / "사건연구_수익률_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")

그룹들 = ["발행사", "거래소", "은행", "카드"]
비트 = 설계["비트코인요인"]
유의 = float(설계["유의수준"])
# 설계의 예상 방향 (출처: 설계 "판정_" 줄들): +1, −1, 0(미정)
예상 = {"발행사": +1, "거래소": +1, "은행": -1, "카드": 0}


#%% [셀 1] 사건 연구 함수 ==================================================
def 사건연구(g, 기간이름="주(250일)", X열=("시장", 비트), 창=None):
    """그룹 g 의 사건별 CAR 과 주판정값 S.
    창: None 이면 1단계 사건창 그대로, (앞, 뒤) 를 주면 각 사건 0일 기준으로 바꿈 (4단계 강건성용)"""
    기간 = 추정기간.loc[기간이름]
    추정 = 수익률.loc[기간["시작"]:기간["끝"]]
    X = np.column_stack([np.ones(len(추정))] + [추정[c].to_numpy() for c in X열])
    y = 추정[g].to_numpy()
    계수, *_ = np.linalg.lstsq(X, y, rcond=None)                       # 주 모형 (설계 "주모형")
    잔차 = y - X @ 계수
    자유도 = len(y) - X.shape[1]
    σ = np.sqrt(잔차 @ 잔차 / 자유도)                                   # 평소 하루 잡음

    거래일 = 수익률.index
    행 = []
    for _, e in 사건창.iterrows():
        시작, 끝 = pd.Timestamp(e["창시작"]), pd.Timestamp(e["창끝"])
        if 창 is not None:                                              # 강건성: 0일 기준으로 창 다시 정함
            k = 거래일.get_loc(시작)
            시작, 끝 = 거래일[k + 창[0]], 거래일[k + 창[1]]
        구간 = 수익률.loc[시작:끝]
        예측 = np.column_stack([np.ones(len(구간))] + [구간[c].to_numpy() for c in X열]) @ 계수
        AR = 구간[g].to_numpy() - 예측                                  # 비정상 수익률 (MacKinlay 1997)
        CAR = AR.sum()
        SE = σ * np.sqrt(len(구간))
        행.append({"그룹": g, "사건": e["이름"], "방향": int(e["방향"]), "창시작": 시작.date(), "창끝": 끝.date(),
                   "창길이": len(구간), "CAR_%": CAR * 100, "SE_%": SE * 100, "t": CAR / SE,
                   "p": 2 * stats.t.sf(abs(CAR / SE), 자유도)})
    사건별 = pd.DataFrame(행)

    # 주판정값 S = 부호 조정 평균 (설계 "주판정값"), 창이 겹치지 않으므로 분산을 더함
    S = (사건별["방향"] * 사건별["CAR_%"]).mean()
    SE_S = np.sqrt((사건별["SE_%"] ** 2).sum()) / len(사건별)
    t = S / SE_S
    return 사건별, {"그룹": g, "S_%": S, "SE_%": SE_S, "t": t, "p": 2 * stats.t.sf(abs(t), 자유도),
                    "최소검출효과_%": 2.8 * SE_S}


#%% [셀 2] 주 판정 ========================================================
사건표, 판정행 = [], []
for g in 그룹들:
    사건별, 요약 = 사건연구(g)
    사건표.append(사건별)
    # 판정 (출처: 설계 "판정_발행사/거래소/은행/카드")
    if 예상[g] == 0:
        요약["판정"] = (f"유의 ({'+' if 요약['S_%'] > 0 else '−'} 방향 기록)" if 요약["p"] < 유의 else "유의하지 않음")
    else:
        맞는방향 = np.sign(요약["S_%"]) == 예상[g]
        요약["판정"] = "성립" if (맞는방향 and 요약["p"] < 유의) else "성립 안 함"
    # 기록만: 사건 A(부결)의 CAR 이 B~E 평균과 반대 부호인가 (설계 "기록만")
    A = 사건별.loc[사건별["사건"] == "A", "CAR_%"].iloc[0]
    BE = 사건별.loc[사건별["사건"] != "A", "CAR_%"].mean()
    요약["A_CAR_%"], 요약["B~E평균_%"] = A, BE
    요약["A반대부호"] = bool(np.sign(A) == -np.sign(BE))
    판정행.append(요약)
사건표 = pd.concat(사건표, ignore_index=True)
판정 = pd.DataFrame(판정행)

pd.set_option("display.width", 220)
f = lambda v: f"{v:.3f}"
print("[3단계] 사건별 CAR (주 모형, 창 [0,+1], 묶은 사건은 1단계 창)")
print(사건표.pivot(index="사건", columns="그룹", values="CAR_%")[그룹들].to_string(float_format=f))
print("\n사건별 t값")
print(사건표.pivot(index="사건", columns="그룹", values="t")[그룹들].to_string(float_format=f))
print("\n[주 판정] S = 부호 조정 평균 CAR (법이 한 걸음 나아갈 때의 평균 반응)")
print(판정[["그룹", "S_%", "SE_%", "t", "p", "최소검출효과_%", "판정"]].to_string(index=False, float_format=f))
print("\n[기록만] 사건 A(부결) 부호 점검")
print(판정[["그룹", "A_CAR_%", "B~E평균_%", "A반대부호"]].to_string(index=False, float_format=f))
print("\n주의: 그룹이 4개라 하나쯤은 우연히 5% 에서 유의할 수 있음 (설계 '기록만')")


#%% [셀 3] 그림 + 저장 ====================================================
fig, ax = plt.subplots(figsize=(9, 4.2))
사건이름 = list(사건창["이름"])
폭 = 0.2
for i, g in enumerate(그룹들):
    d = 사건표[사건표["그룹"] == g].set_index("사건").loc[사건이름]
    x = np.arange(len(사건이름)) + (i - 1.5) * 폭
    ax.bar(x, d["CAR_%"], 폭, yerr=1.96 * d["SE_%"], capsize=2, label=g)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(np.arange(len(사건이름)))
ax.set_xticklabels([f"{e}\n{s[5:]}" + (" (부결)" if d == -1 else "")
                    for e, s, d in zip(사건창["이름"], 사건창["창시작"], 사건창["방향"])])
ax.set_ylabel("누적 비정상 수익률 CAR (%)")
ax.set_title("GENIUS Act 사건별 CAR (주 모형: 시장 + IBIT, 막대 = 95% 구간)", loc="left", fontsize=11)
ax.legend(ncol=4, fontsize=9)
fig.tight_layout()
fig.savefig(그림폴더 / "사건연구_그림1_사건별CAR.png", dpi=150)
plt.show()

사건표.to_csv(데이터폴더 / "사건연구_3단계_사건별CAR.csv", index=False, encoding=CSV인코딩)
판정.to_csv(데이터폴더 / "사건연구_3단계_판정.csv", index=False, encoding=CSV인코딩)
print("저장: 사건연구_3단계_사건별CAR.csv, 사건연구_3단계_판정.csv, 그림/사건연구_그림1_사건별CAR.png")
