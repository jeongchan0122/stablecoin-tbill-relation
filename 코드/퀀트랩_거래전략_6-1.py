# -*- coding: utf-8 -*-
"""
퀀트랩_거래전략_6-1.py
=====================================================================
[연구] 거래전략 연구 (설계: 퀀트랩_거래전략_0.py -> 데이터/거래전략_설계.csv)
[단계] 6-1: 신호 시작 시점 사후 탐색  ※ 2~7단계 결과를 본 뒤 추가한 분석 (설계 "사후_시작시점")
       주 분석(2024년 시작)과 판정 A~E 는 바뀌지 않음. 6단계 강건성 목록(사전에 정함)과도 구분함

[왜 하나]
  1단계에서 신호 시작을 연 단위 기준(계수 > 0, F ≥ 10)으로 정해 2023년을 통째로 뺐음
  -> 2023년 중 관계가 생긴 시점이 있었다면, 그때부터 썼을 때 결론이 달라지는지 확인

[무엇을 하나]
  ① 시작 시점 찾기 (금리 안 씀): 2023Q1 ~ 2024Q1 분기별 1단계 (5일 유입 ← GIV 5일 합)
     원래 기준(계수 > 0, F ≥ 10)을 처음 넘는 분기의 첫날 = 새 시작. 시점을 여러 개 놓고 고르지 않음
  ② 새 시작 기준 검정력 (퀀트랩_거래전략_2 와 같은 계산, 계수는 안 봄)
  ③ 새 시작 기준 판정 B·C·D (퀀트랩_거래전략_3·4·5 와 같은 계산)
     표본 외 예측은 원래처럼 2025-01 부터 (사용자 확정 10/22 (가)), 추정에만 앞당긴 자료를 더 씀
  모든 결과는 원래(2024년 시작)와 나란히 출력. 원래 줄은 1~5단계 결과와 같아야 함

[출처 표기 규칙]  (퀀트랩_거래전략_0.py 와 같음)
  출처: 퀀트랩_거래전략_○ [셀 ○] / 사용자 확정 (날짜) / 설계 "항목이름"

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/  (읽기만 함)
  거래전략_설계.csv, 확장_코인체인_패널_일별.csv, 확장_합산시총_일별.csv, 거래전략_신호_평일.csv,
  거래전략_예측자료_평일.csv, 4단계_TB입찰_원자료.csv, 4단계_그림7_결과.csv, 확장_시장자료_원자료_일별.csv
[출력]  (기존 파일을 덮어쓰지 않음)
  데이터/거래전략_6-1_분기1단계.csv
  데이터/거래전략_6-1_시작시점.csv

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install pandas numpy scipy statsmodels
=====================================================================
"""

#%% [셀 0] 패키지 + 폴더 + 설계 읽기 =======================================
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# 폴더 (출처: 사용자 지시 - 바탕화면/스테이블코인_퀀트랩, 한국어 파일명)
데이터폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "데이터"
CSV인코딩 = "utf-8-sig"

# 설계 읽기 (출처: 퀀트랩_거래전략_0.py)
설계 = pd.read_csv(데이터폴더 / "거래전략_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
assert "사후_시작시점" in 설계.index, '설계에 "사후_시작시점" 줄이 없습니다. 퀀트랩_거래전략_0.py 를 고쳐 다시 실행하세요.'
표본끝 = pd.Timestamp(설계["표본끝"])                                    # 2026-03-31
첫창끝 = pd.Timestamp(설계["첫창끝_주"])                                 # 2024-12-31 (표본 외 예측 시작은 원래와 같음)
패널시작 = pd.Timestamp(설계["패널시작"])                                # 2021-01-01
주h = int(설계["주h"])                                                   # 13
NW더하기 = int(re.search(r"\+\s*(\d+)", 설계["NW시차규칙"]).group(1))     # 4
활성기준 = float(설계["활성기준_달러"])                                   # 1,000,000
최소N = int(설계["최소N"])                                               # 5
윈저배수 = float(설계["윈저배수"])                                        # 11.8608
신호지연 = int(설계["신호지연_달력일"])                                   # 1
z = norm.ppf(1 - float(설계["유의수준"]) / 2)                             # 1.96
MDE배수 = float(설계["MDE배수"])                                         # 2.8
문헌 = [float(v) for v in re.findall(r"\d+\.\d+", 설계["거래비용_문헌값"])]   # [0.74, 0.79]
원래비움끝 = 패널시작 + pd.Timedelta(days=int(설계["처음비움_달력일"]) - 1)    # 2023-12-31
점검분기 = pd.period_range("2023Q1", "2024Q1", freq="Q")                   # 출처: 사용자 확정 10/22 (2023년 + 원래 시작 분기)
print("데이터 폴더:", 데이터폴더)
print(f"원래 신호 비움 끝 {원래비움끝.date()}, 점검 분기 {점검분기[0]} ~ {점검분기[-1]}")


#%% [셀 1] 실시간 일별 GIV (퀀트랩_거래전략_1 [셀 1]~[셀 4] 와 같은 계산, 비움만 나중에) =====
print("실시간 GIV 계산 중 (수십 초)...")
패널 = pd.read_csv(데이터폴더 / "확장_코인체인_패널_일별.csv", parse_dates=["date"], encoding=CSV인코딩)
시총 = 패널[패널["date"] >= 패널시작].pivot_table(index="date", columns="pair", values="mcap", aggfunc="sum")
시총 = 시총.reindex(pd.date_range(시총.index.min(), 시총.index.max(), freq="D"))
전날 = 시총.shift(1)
g = (시총 - 전날) / 전날.where(전날 > 0) * 100
활성 = (시총 > 활성기준) & (전날 > 활성기준)
실시간패널 = 활성.astype(int).cummax().astype(bool)
하루값 = [줄[~np.isnan(줄)] for 줄 in g.where(활성).values]
누적 = np.cumsum([len(v) for v in 하루값])
이음 = np.concatenate(하루값)
MAD = np.full(len(g), np.nan)
for i, n in enumerate(누적):
    if n:
        x = 이음[:n]
        MAD[i] = np.median(np.abs(x - np.median(x)))
경계 = pd.Series(윈저배수 * MAD, index=g.index)
gw = g.clip(lower=-경계, upper=경계, axis=0)
s = 전날.where(활성)
s = s.div(s.sum(axis=1), axis=0)
gg = gw.where(활성)
N = ((전날 > 활성기준) & 실시간패널).sum(axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    ε = gg.sub((s * gg).sum(axis=1), axis=0)
    GIV일 = (s.sub(1 / N, axis=0) * ε).sum(axis=1, min_count=1)
GIV일[N < 최소N] = np.nan
평일 = pd.bdate_range(GIV일.index.min(), GIV일.index.max())


def 신호만들기(비움끝):
    """비움끝 이후 GIV 만 써서 5일 합(같은 날, 1단계용)과 거래 신호(하루 늦춤)를 만듦"""
    v = GIV일.copy()
    v[v.index <= 비움끝] = np.nan
    G5 = v.rolling(5).sum()
    return G5.reindex(평일), G5.shift(신호지연).reindex(평일)


# 확인: 원래 비움끝이면 퀀트랩_거래전략_1 이 저장한 신호와 같아야 함
원신호 = pd.read_csv(데이터폴더 / "거래전략_신호_평일.csv", parse_dates=["date"], index_col="date",
                     encoding=CSV인코딩)["거래신호_GIV5"]
_, 확인신호 = 신호만들기(원래비움끝)
print(f"확인: 거래전략_1 신호와 최대 차이 {(확인신호 - 원신호.reindex(평일)).abs().max():.2e} (0 이면 같은 계산)")


#%% [셀 2] ① 시작 시점 찾기: 분기별 1단계 (금리 안 씀) ==============================
# 식: 5일 유입 = 상수 + 계수 × GIV 5일 합, NW 시차 1 (퀀트랩_거래전략_1 [셀 7] 과 같음)
# 점검용 GIV 는 2022년부터 (퀀트랩_거래전략_1 의 365일 비움 버전과 같음)
MC = pd.read_csv(데이터폴더 / "확장_합산시총_일별.csv", parse_dates=["date"], index_col="date",
                 encoding=CSV인코딩)["Total"].reindex(평일) / 1e9
유입5 = (MC - MC.shift(5)).rename("유입")


def 일단계(d):
    r = sm.OLS(d["유입"], sm.add_constant(d["GIV"])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return r.params["GIV"], r.tvalues["GIV"] ** 2, int(r.nobs)


G5_점검, _ = 신호만들기(패널시작 + pd.Timedelta(days=364))
x = pd.concat([유입5, G5_점검.rename("GIV")], axis=1).dropna()
분기행, 새시작 = [], None
print("[분기별 1단계] 기준: 계수 > 0 이고 F ≥ 10 (출처: 설계 \"처음비움_달력일\" 의 기준을 분기에 적용)")
for 분기 in 점검분기:
    b, F, n = 일단계(x[(x.index >= 분기.start_time) & (x.index <= 분기.end_time)])
    통과 = (b > 0) and (F >= 10)
    분기행.append({"분기": str(분기), "계수": b, "F": F, "관측치": n, "통과": 통과})
    print(f"  {분기}: 계수 {b:+.2f}, F {F:6.1f}, 관측치 {n}  -> {'통과' if 통과 else '미통과'}")
    if 통과 and 새시작 is None:
        새시작 = 분기.start_time
pd.DataFrame(분기행).to_csv(데이터폴더 / "거래전략_6-1_분기1단계.csv", index=False, encoding=CSV인코딩)
print(f"처음 통과한 분기의 첫날: {새시작.date() if 새시작 is not None else '없음'}")
print("저장: 거래전략_6-1_분기1단계.csv")


#%% [셀 3] 예측 자료 재료 (퀀트랩_거래전략_2 [셀 1]~[셀 3], 5 [셀 1] 과 같음) ===========
자료 = pd.read_csv(데이터폴더 / "거래전략_예측자료_평일.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
평2 = 자료.index
스프 = 자료["스프레드_bp"]
FOMC = 자료["FOMC성명일"]
입찰 = pd.read_csv(데이터폴더 / "4단계_TB입찰_원자료.csv", encoding=CSV인코딩, dtype=str)
if "cash_management_bill_cmb" in 입찰.columns:
    입찰 = 입찰[입찰["cash_management_bill_cmb"] != "Yes"]
발표열 = [c for c in 입찰.columns if "announc" in c.lower() and c.lower().endswith("_date")][0]
입찰 = pd.DataFrame({"발표일": pd.to_datetime(입찰[발표열], errors="coerce"),
                     "발행일": pd.to_datetime(입찰["issue_date"], errors="coerce"),
                     "금액": pd.to_numeric(입찰["offering_amt"], errors="coerce") / 1e9}).dropna()
날 = pd.Series(평2, index=평2)
끝날 = 날.shift(-주h)
FOMC창 = (sum(FOMC.shift(-k) for k in range(1, 주h + 1)) > 0).astype(float)
입찰창 = pd.Series([입찰.loc[(입찰["발표일"] <= t) & (입찰["발행일"] > t) & (입찰["발행일"] <= e), "금액"].sum()
                   if pd.notna(e) else np.nan for t, e in zip(평2, 끝날)], index=평2)
그림7 = pd.read_csv(데이터폴더 / "4단계_그림7_결과.csv", encoding=CSV인코딩)
IV13 = 그림7[(그림7["사양"] == "B 주 분석 (ETF 없음)") & (그림7["방법"] == "IV") & (그림7["h"] == 주h)]["계수"].iloc[0]
시장 = pd.read_csv(데이터폴더 / "확장_시장자료_원자료_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
금리 = pd.DataFrame({1 / 12: 시장["1개월"], 0.25: 시장["3개월"], 0.5: 시장["6개월"]}).reindex(평2) / 100
통제 = ["지난5일", "FOMC", "입찰"]


def 예측자료(비움끝):
    _, 신호 = 신호만들기(비움끝)
    d = pd.DataFrame({"y": 스프.shift(-주h) - 스프, "신호": 신호.reindex(평2), "지난5일": 스프 - 스프.shift(5),
                      "FOMC": FOMC창, "입찰": 입찰창, "끝날": 끝날})
    return d[끝날 <= 표본끝].dropna()


def 맞추고예측(학습, 열, 오늘):
    X = np.column_stack([np.ones(len(학습))] + [학습[c].values for c in 열])
    b = np.linalg.lstsq(X, 학습["y"].values, rcond=None)[0]
    return float(np.r_[1.0, 오늘[열].values] @ b)


def 보간금리(날짜, 만기):
    줄 = 금리.loc[날짜].dropna()
    return np.nan if 줄.empty else float(np.interp(만기, 줄.index.values, 줄.values))


def 보유수익률(산날, 판날, n):
    """로그 보유수익률 (설계 "수익률계산", Campbell, Lo & MacKinlay 1997 10장)"""
    τ = (판날 - 산날).days / 365
    return n * 금리.at[산날, n] - (n - τ) * 보간금리(판날, n - τ), τ


#%% [셀 4] ②·③ 검정력과 판정 B·C·D 를 시작 시점마다 계산 ===========================
def 전부(비움끝):
    G5, _ = 신호만들기(비움끝)
    d = 예측자료(비움끝)
    sd = d["신호"].std()
    out = {"신호첫날": d.index.min().date(), "관측치": len(d)}
    # 1단계 계수 (퀀트랩_거래전략_1 [셀 7]: 표본끝까지)
    d1 = pd.concat([유입5, G5.rename("GIV")], axis=1).dropna()
    b1, F1, _ = 일단계(d1[d1.index <= 표본끝])
    # 표본 내 회귀 (퀀트랩_거래전략_2 [셀 5], 3 [셀 3])
    r = sm.OLS(d["y"], sm.add_constant(d[["신호"] + 통제])).fit(cov_type="HAC", cov_kwds={"maxlags": 주h + NW더하기})
    se = r.bse["신호"]
    δ = IV13 * 100 * b1                                      # 설계 "검정력_가정효과"
    k = abs(δ) / se
    out.update({"1단계계수": b1, "1단계F": F1, "SE": se, "가정효과_1SD": δ * sd, "MDE_1SD": MDE배수 * se * sd,
                "검정력": norm.cdf(k - z) + norm.cdf(-k - z),
                "B_계수_1SD": r.params["신호"] * sd, "B_p": 2 * norm.sf(abs(r.tvalues["신호"]))})
    # 표본 외 (퀀트랩_거래전략_4 [셀 3]·[셀 4]): 2025-01 부터, 끝날 ≤ t 인 관측치만
    행 = []
    for t in d.index[d.index > 첫창끝]:
        학습 = d[d["끝날"] <= t]
        행.append({"date": t, "판날": d.at[t, "끝날"], "실제": d.at[t, "y"], "학습": len(학습),
                  "큰": 맞추고예측(학습, ["신호"] + 통제, d.loc[t]), "기준": 맞추고예측(학습, 통제, d.loc[t])})
    p = pd.DataFrame(행).set_index("date")
    e큰, e기 = p["실제"] - p["큰"], p["실제"] - p["기준"]
    f = e기 ** 2 - (e큰 ** 2 - (p["기준"] - p["큰"]) ** 2)
    cw = sm.OLS(f.values, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": 주h + NW더하기}).tvalues[0]
    out.update({"C_예측수": len(p), "C_첫학습": int(p["학습"].iloc[0]), "C_R2": 1 - (e큰 ** 2).sum() / (e기 ** 2).sum(),
                "C_CW_t": cw, "C_CW_p": norm.sf(cw)})
    # 백테스트 (퀀트랩_거래전략_5 [셀 3]·[셀 4])
    초과, k3, k6 = [], [], []
    for t, 줄 in p.iterrows():
        if 금리.loc[t, [0.25, 0.5]].isna().any() or 금리.loc[줄["판날"]].isna().all():   # 설계 "금리빈날"
            초과.append(np.nan); k3.append(np.nan); k6.append(np.nan)
            continue
        R3, τ = 보유수익률(t, 줄["판날"], 0.25)
        R6, _ = 보유수익률(t, 줄["판날"], 0.5)
        초과.append((R3 - R6) * 1e4); k3.append((0.25 + 0.25 - τ) / 2); k6.append((0.5 + 0.5 - τ) / 2)
    p["초과"], p["k3"], p["k6"] = 초과, k3, k6
    p["갈아탐"] = (p["큰"] < 0) & p["초과"].notna()          # 설계 "전략규칙"
    갈 = p[p["갈아탐"]]
    out.update({"D_갈아탄날": len(갈), "D_결정일": len(p), "D_s별": 갈["초과"].sum() / (갈["k3"] + 갈["k6"]).sum(),
                "D_갈아탄날평균": 갈["초과"].mean(), "D_매일평균": p["초과"].mean()})
    out["D_신호타이밍"] = out["D_갈아탄날평균"] - out["D_매일평균"]
    return out


print("계산 중 (표본 외 예측을 날마다 다시 추정해서 1분쯤)...")
후보 = {"2024년 시작 (주 분석)": 원래비움끝}
if 새시작 is not None and 새시작 <= 원래비움끝:
    후보[f"{새시작.date()} 시작 (사후)"] = 새시작 - pd.Timedelta(days=1)
결과 = pd.DataFrame({이름: 전부(끝) for 이름, 끝 in 후보.items()}).T


#%% [셀 5] 출력 + 저장 ==========================================================
for 이름, r in 결과.iterrows():
    print(f"\n[{이름}] 신호 {r['신호첫날']} ~, 관측치 {r['관측치']}")
    print(f"  1단계: 계수 {r['1단계계수']:.3f}, F {r['1단계F']:.1f}")
    print(f"  검정력: SE {r['SE']:.3f}, 가정 효과 {r['가정효과_1SD']:+.2f} / MDE {r['MDE_1SD']:.2f} bp/1SD "
          f"-> {r['검정력']:.1%}")
    print(f"  B 표본 내: 계수 {r['B_계수_1SD']:+.2f} bp/1SD, p {r['B_p']:.3f} "
          f"-> {'성립' if (r['B_계수_1SD'] < 0 and r['B_p'] < 0.05) else '성립 안 함'}")
    print(f"  C 표본 외: 예측 {r['C_예측수']}개 (첫 학습 {r['C_첫학습']}), R² {r['C_R2']:+.2%}, CW t {r['C_CW_t']:+.2f}, "
          f"한쪽 p {r['C_CW_p']:.3f} -> {'성립' if (r['C_R2'] > 0 and r['C_CW_p'] < 0.05) else '성립 안 함'}")
    print(f"  D 백테스트: 갈아탄 날 {r['D_갈아탄날']}/{r['D_결정일']}, s* {r['D_s별']:.3f} bp "
          f"-> {'형식상 성립' if r['D_s별'] > max(문헌) else '성립 안 함'}, "
          f"신호 타이밍 {r['D_신호타이밍']:+.3f} bp (갈아탄 날 {r['D_갈아탄날평균']:+.3f} vs 매일 {r['D_매일평균']:+.3f})")
print("\n(주 분석 줄은 2~5단계 결과와 같아야 함: 검정력 70.8%, B −1.05·p 0.049, C R² −2.07%·CW t +0.70, D s* 0.951)")
결과.rename_axis("시작").to_csv(데이터폴더 / "거래전략_6-1_시작시점.csv", encoding=CSV인코딩)
print("저장: 거래전략_6-1_시작시점.csv")
