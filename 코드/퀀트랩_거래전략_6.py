# -*- coding: utf-8 -*-
"""
퀀트랩_거래전략_6.py
=====================================================================
[연구] 거래전략 연구 (설계: 퀀트랩_거래전략_0.py -> 데이터/거래전략_설계.csv)
[단계] 6단계: 강건성 - 설계 "강건성목록" 에 미리 적어 둔 것만

[무엇을 하나]  (판정 B·C 자체는 바뀌지 않음 - 그 위에 해석을 더하는 단계)
  ① 위약 (설계 "위약_정의", "위약_통과")
       신호를 h + 5 평일 뒤 값으로 바꿔 3단계 회귀를 다시 함
       -> 신호의 5일 창 전체가 결과 기간 뒤로 넘어감. 결과 기간 뒤의 정보는 그 전 변화를 일으킬 수 없으므로
          위약 계수가 양측 5% 에서 유의하지 않으면 통과
       (같은 생각: 퀀트랩_BIS_4 [셀 9] 사전추세 점검, Stock & Watson 2018)
  ② FOMC 창 제외 (설계 "FOMC창_정의")  : (t, t+h] 에 FOMC 성명일이 있는 관측치를 빼고 다시 (FOMC 통제는 뺌)
  ③ 부채한도 기간 제외 (설계 "부채한도기간"): 결과 기간 [t, t+h] 가 2025-01-21 ~ 2025-07-04 와 겹치는 관측치를 빼고 다시
  ④ h = 10·20·25: 3단계·4단계 표에서 옮겨 정리만 함
  ②·③ 은 판정 B(표본 내)와 C(표본 외, 추정·평가 모두에서 뺌) 둘 다. ① 은 B 에만 (설계 "강건성_C적용")
  판정 기준 (설계 "강건성_판정"): 통과·실패는 ① 위약만. ②~④ 는 부호와 크기만 기록

[출처 표기 규칙]  (퀀트랩_거래전략_0.py 와 같음)
  출처: 논문 ○○절 / 각주 ○ / 표 ○      출처: 퀀트랩_BIS_○ / 퀀트랩_거래전략_○
  출처: 사용자 확정 (날짜)               출처: 설계 "항목이름" -> 데이터/거래전략_설계.csv 의 해당 줄

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  거래전략_설계.csv, 거래전략_예측자료_평일.csv, 4단계_TB입찰_원자료.csv
  거래전략_3단계_표본내.csv, 거래전략_4단계_표본외.csv   (기준 결과 확인 + ④ 정리용)
[출력]
  데이터/거래전략_6단계_강건성.csv
  그림/거래전략_그림7_강건성.png

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
for k in ["위약_정의", "위약_통과", "FOMC창_정의", "부채한도기간", "강건성_판정", "강건성_C적용"]:
    assert k in 설계.index, f'설계에 "{k}" 줄이 없습니다. 퀀트랩_거래전략_0.py 를 고쳐 다시 실행하세요.'
표본끝 = pd.Timestamp(설계["표본끝"])                                    # 2026-03-31
첫창끝 = pd.Timestamp(설계["첫창끝_주"])                                 # 2024-12-31
주h = int(설계["주h"])                                                   # 13
NW더하기 = int(re.search(r"\+\s*(\d+)", 설계["NW시차규칙"]).group(1))     # 4
유의수준 = float(설계["유의수준"])                                        # 0.05
z = norm.ppf(1 - 유의수준 / 2)                                            # 1.96
위약이동 = 주h + int(re.search(r"h\s*\+\s*(\d+)", 설계["위약_정의"]).group(1))   # h + 5 = 18 평일
한도시작, 한도끝 = [pd.Timestamp(x) for x in 설계["부채한도기간"].split(",")]   # 2025-01-21, 2025-07-04
print("데이터 폴더:", 데이터폴더)
print(f"설계: 주h {주h}, 위약 = 신호를 {위약이동} 평일 뒤로, 부채한도 {한도시작.date()} ~ {한도끝.date()}")


#%% [셀 1] 자료 읽기 (퀀트랩_거래전략_4 [셀 1] 과 같음) ============================
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


#%% [셀 2] 예측 자료, 표본 내 회귀, 표본 외 예측 (퀀트랩_거래전략_3·4 와 같은 정의) ======
날열 = pd.Series(평일, index=평일)
기본통제 = ["지난5일", "FOMC", "입찰"]          # 출처: 설계 "통제변수_정의"


def 예측자료(h, 결과, 신호열=None):
    """신호열: 위약일 때 바꿔 넣을 신호 (없으면 원래 신호)"""
    x = 금리[결과]
    끝날 = 날열.shift(-h)
    d = pd.DataFrame({"y": x.shift(-h) - x, "신호": 신호 if 신호열 is None else 신호열,
                      "지난5일": x - x.shift(5)}, index=평일)
    d["FOMC"] = (sum(FOMC.shift(-k) for k in range(1, h + 1)) > 0).astype(float)
    d["입찰"] = [입찰.loc[(입찰["발표일"] <= t) & (입찰["발행일"] > t) & (입찰["발행일"] <= e), "금액"].sum()
                 if pd.notna(e) else np.nan for t, e in zip(평일, 끝날)]
    d["끝날"] = 끝날
    d = d[끝날 <= 표본끝]
    return d.dropna()


def 표본내(d, h, 통제):
    """퀀트랩_거래전략_3 [셀 2] 와 같은 회귀 (NW 시차 h + 4)"""
    X = sm.add_constant(d[["신호"] + 통제])
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + NW더하기})
    b, se = r.params["신호"], r.bse["신호"]
    return {"계수_bp": b, "SE_bp": se, "p": 2 * norm.sf(abs(b / se)), "관측치": int(r.nobs)}


def 맞추고예측(학습, 열, 오늘):
    X = np.column_stack([np.ones(len(학습))] + [학습[c].values for c in 열])
    b = np.linalg.lstsq(X, 학습["y"].values, rcond=None)[0]
    return float(np.r_[1.0, 오늘[열].values] @ b)


def 표본외(d, h, 통제):
    """퀀트랩_거래전략_4 [셀 3]·[셀 4] 와 같은 확장창 예측 + 표본 외 R² + Clark-West (기준 = 신호 뺀 모형)"""
    행 = []
    for t in d.index[d.index > 첫창끝]:
        학습 = d[d["끝날"] <= t]
        행.append({"실제": d.at[t, "y"], "큰": 맞추고예측(학습, ["신호"] + 통제, d.loc[t]),
                  "기준": 맞추고예측(학습, 통제, d.loc[t])})
    p = pd.DataFrame(행)
    e큰, e기 = p["실제"] - p["큰"], p["실제"] - p["기준"]
    f = e기 ** 2 - (e큰 ** 2 - (p["기준"] - p["큰"]) ** 2)
    r = sm.OLS(f.values, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": h + NW더하기})
    return {"R2": 1 - (e큰 ** 2).sum() / (e기 ** 2).sum(), "CW_t": r.tvalues[0], "CW_p": norm.sf(r.tvalues[0]),
            "예측수": len(p)}


#%% [셀 3] 기준 결과 다시 확인 (3단계·4단계와 같은 숫자가 나와야 함) ====================
d13 = 예측자료(주h, "스프레드")
신호SD = d13["신호"].std()                      # 1SD 환산 (퀀트랩_거래전략_3 [셀 3] 과 같음)
기준B = 표본내(d13, 주h, 기본통제)
기준C = 표본외(d13, 주h, 기본통제)

옛3 = pd.read_csv(데이터폴더 / "거래전략_3단계_표본내.csv", encoding=CSV인코딩)
옛4 = pd.read_csv(데이터폴더 / "거래전략_4단계_표본외.csv", encoding=CSV인코딩)
옛B = 옛3[(옛3["결과변수"] == "주(스프레드)") & (옛3["h"] == 주h)].iloc[0]
옛C = 옛4[(옛4["결과변수"] == "주(스프레드)") & (옛4["h"] == 주h)].iloc[0]
assert np.isclose(기준B["계수_bp"], 옛B["계수_bp"]) and np.isclose(기준C["R2"], 옛C["R2_신호뺀모형"]), \
    "3·4단계와 기준 결과가 다릅니다 (자료나 정의가 바뀌었는지 확인)"
print(f"확인: 기준 B 계수 {기준B['계수_bp'] * 신호SD:+.2f} bp / 1SD (p {기준B['p']:.3f}), "
      f"기준 C 표본 외 R² {기준C['R2']:+.2%} -> 3·4단계와 같음")


#%% [셀 4] ① 위약: 신호를 결과 기간 뒤로 옮김 (판정 B 에만) =========================
# 출처: 설계 "위약_정의" - h + 5 평일 뒤의 신호 (평일 표에서 칸 이동 = 평일 수 이동)
#   신호의 5 달력일 창 = 그 날 직전 5일. 5 평일 = 언제나 달력일 7일 -> 창 전체가 t+h 보다 뒤
위약신호 = 신호.shift(-위약이동)
d위약 = 예측자료(주h, "스프레드", 위약신호)
d위약 = d위약.loc[d위약.index.intersection(d13.index)]       # 기준과 같은 날짜만 (신호 첫날 이전 t 는 뺌)
위약 = 표본내(d위약, 주h, 기본통제)
위약통과 = 위약["p"] >= 유의수준                 # 출처: 설계 "위약_통과"
print(f"[① 위약] 신호를 {위약이동} 평일 뒤 값으로 바꿈, 관측치 {위약['관측치']}")
print(f"  위약 계수 {위약['계수_bp'] * 신호SD:+.2f} bp / 1SD, p {위약['p']:.3f}  "
      f"(원래 {기준B['계수_bp'] * 신호SD:+.2f}, p {기준B['p']:.3f})")
print(f"  -> {'통과 (결과 기간 뒤 정보는 예측력 없음)' if 위약통과 else '통과 못 함 (우연·공통추세 또는 역인과 - 구분 불가)'}")


#%% [셀 5] ②·③ 기간 제외: 판정 B (표본 내) =========================================
def FOMC제외(d):
    """설계 "FOMC창_정의": (t, t+h] 에 성명일이 있는 관측치(FOMC = 1) 제외"""
    return d[d["FOMC"] == 0]


def 한도제외(d):
    """설계 "부채한도기간": [t, t+h] 가 부채한도 기간과 겹치면 제외 (겹침 = t ≤ 끝 이고 t+h ≥ 시작)"""
    return d[~((d.index <= 한도끝) & (d["끝날"] >= 한도시작))]


설정들 = {"② FOMC 창 제외": (FOMC제외, ["지난5일", "입찰"]),          # FOMC 통제는 늘 0 이라 뺌
          "③ 부채한도 기간 제외": (한도제외, 기본통제)}
결과행 = [{"점검": "기준 (3·4단계)", "B_계수_1SD_bp": 기준B["계수_bp"] * 신호SD, "B_p": 기준B["p"],
          "B_관측치": 기준B["관측치"], "C_R2": 기준C["R2"], "C_CW_t": 기준C["CW_t"], "C_CW_p": 기준C["CW_p"],
          "C_예측수": 기준C["예측수"]},
         {"점검": "① 위약 (B 만)", "B_계수_1SD_bp": 위약["계수_bp"] * 신호SD, "B_p": 위약["p"],
          "B_관측치": 위약["관측치"], "위약통과": 위약통과}]
for 이름, (거르기, 통제) in 설정들.items():
    d = 거르기(d13)
    b = 표본내(d, 주h, 통제)
    결과행.append({"점검": 이름, "B_계수_1SD_bp": b["계수_bp"] * 신호SD, "B_p": b["p"], "B_관측치": b["관측치"]})
    print(f"[{이름}] B: 계수 {b['계수_bp'] * 신호SD:+.2f} bp / 1SD, p {b['p']:.3f}, 관측치 {b['관측치']} "
          f"(빠진 관측치 {len(d13) - len(d)}) / 부호 {'유지' if np.sign(b['계수_bp']) == np.sign(기준B['계수_bp']) else '바뀜'}")


#%% [셀 6] ②·③ 기간 제외: 판정 C (표본 외, 추정·평가 모두에서 뺌) ======================
# 출처: 설계 "강건성_C적용"
for 행, (이름, (거르기, 통제)) in zip(결과행[2:], 설정들.items()):
    c = 표본외(거르기(d13), 주h, 통제)
    행.update({"C_R2": c["R2"], "C_CW_t": c["CW_t"], "C_CW_p": c["CW_p"], "C_예측수": c["예측수"]})
    print(f"[{이름}] C: 표본 외 R² {c['R2']:+.2%}, Clark-West t {c['CW_t']:+.2f} (한쪽 p {c['CW_p']:.3f}), 예측 {c['예측수']}개")


#%% [셀 7] ④ h = 10·20·25 (3·4단계 표에서 옮김) + 저장 ===========================
for h in [10, 20, 25]:                               # 출처: 설계 "강건성목록"
    a = 옛3[(옛3["결과변수"] == "주(스프레드)") & (옛3["h"] == h)].iloc[0]
    c = 옛4[(옛4["결과변수"] == "주(스프레드)") & (옛4["h"] == h)].iloc[0]
    결과행.append({"점검": f"④ h = {h}", "B_계수_1SD_bp": a["계수_1SD_bp"], "B_p": a["p"], "B_관측치": a["관측치"],
                  "C_R2": c["R2_신호뺀모형"], "C_CW_t": c["CW_t_신호뺀모형"], "C_CW_p": c["CW_p_신호뺀모형"],
                  "C_예측수": c["예측수"]})
강건 = pd.DataFrame(결과행)
강건.to_csv(데이터폴더 / "거래전략_6단계_강건성.csv", index=False, encoding=CSV인코딩)

pd.set_option("display.width", 200)
print("[강건성 요약] B = 표본 내 (신호 1SD 당 bp), C = 표본 외 (기준 = 신호 뺀 모형)")
print(강건[["점검", "B_계수_1SD_bp", "B_p", "B_관측치", "C_R2", "C_CW_t", "C_CW_p"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("저장: 거래전략_6단계_강건성.csv")


#%% [셀 8] 그림 ==============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
보일 = 강건.iloc[::-1]
se = 보일["B_계수_1SD_bp"].abs() / norm.isf(보일["B_p"] / 2)              # p 에서 거꾸로 구한 표준오차
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.errorbar(보일["B_계수_1SD_bp"], range(len(보일)), xerr=z * se, fmt="o", color="#2171b5", ecolor="#9ecae1", capsize=3)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(기준B["계수_bp"] * 신호SD, color="#e6550d", ls="--", lw=0.8, label="기준 (3단계)")
ax.set_yticks(range(len(보일)))
ax.set_yticklabels(보일["점검"])
ax.set_xlabel("표본 내 계수 (신호 1SD 당 bp), 95% 구간")
ax.set_title("강건성: 판정 B (주결과변수 스프레드)", loc="left", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(그림폴더 / "거래전략_그림7_강건성.png", dpi=150)
plt.close(fig)
print("저장: 그림/거래전략_그림7_강건성.png")
