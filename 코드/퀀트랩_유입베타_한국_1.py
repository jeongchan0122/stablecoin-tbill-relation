# -*- coding: utf-8 -*-
"""
퀀트랩_유입베타_한국_1.py
=====================================================================
[연구] 유입 베타 연구 - 한국 확장: 같은 질문을 코스피·코스닥 주식으로 다시 봄
[단계] 한국 1단계: 자료 정리 + 한국 0일·사건 창 + 유입 베타 + 5개 묶음 + 검정력

[이 코드가 하는 일]
  사건 창의 수익률은 아직 보지 않습니다 (판정은 퀀트랩_유입베타_한국_2).
  ① DataGuide 종목·지수 파일을 읽어 날짜 × 종목 표로 정리
     - 거래정지일, 하루 |수익률| > 30% 인 날은 수익률에서 뺌 (설계 "한국_거래정지", "한국_가격제한")
  ② 요인을 한국 거래일에 맞춤
     - IBIT: 한국 거래일 t 에, 직전 한국 거래일 ~ t 전날 사이에 끝난 미국 거래일들의 IBIT 수익률 (설계 "한국_IBIT시점")
     - 유입: DefiLlama 날짜 t 값의 하루 변화 (설계 "한국_유입시점")
  ③ 3부 원사건의 시각을 한국 시간으로 바꿔 한국 0일 -> 창 [0,+1] -> 겹치면 묶기 (설계 "한국_0일규칙")
  ④ 한국 추정 기간: 한국 기준 사건 A 0일의 11 거래일 전에 끝나는 250일 (설계 "한국_추정기간")
  ⑤ 종목 거르기 (KS·KQ, 스팩 제외, 200일 이상) -> 종목별 유입 베타 -> 5개 묶음
  ⑥ 검정력: 차이 묶음(묶음5 − 묶음1)의 최소 검출 효과 (추정 기간 자료만)

[이 코드에서 정한 계산 방식 - 설계표에 없던 것]
  묶음의 비정상 수익률 = 구성 종목마다 '자기 시장 지수 + IBIT' 모형(3부 모형)으로 구한 비정상 수익률의 그날 평균
    - 미국은 묶음 수익률 하나에 시장 하나로 회귀했지만, 한국은 종목마다 시장(코스피/코스닥)이 달라서
      종목 단위로 비정상 수익률을 구한 뒤 평균함 (여러 종목의 비정상 수익률 합산: MacKinlay 1997)
    - 차이 묶음의 비정상 수익률 = 묶음5 평균 − 묶음1 평균, 표준편차는 추정 기간 이 값의 표준편차

[입력] 데이터 폴더: 유입베타_설계.csv, 사건연구_설계.csv(원사건), 사건연구_사건창.csv(미국 사건 이름),
       유입베타_DataGuide_한국_일별.xlsx, 유입베타_DataGuide_지수_일별.xlsx,
       사건연구_수익률_일별.csv(IBIT), 확장_합산시총_일별.csv(유입)
[출력] 유입베타_한국_사건창.csv, 유입베타_한국_추정기간.csv, 유입베타_한국_수익률_일별.csv, 유입베타_한국_시가총액_일별.csv,
       유입베타_한국_요인_일별.csv, 유입베타_한국_베타.csv(종목별 베타·묶음·이름·시장·업종), 유입베타_한국_검정력.csv
       (수익률·시가총액·베타 파일은 DataGuide 자료에서 나온 값 - 공유·업로드 금지)
[Spyder] "#%%" 셀 단위로 Shift+Enter ([셀 1] 엑셀 읽기가 1~3분 걸릴 수 있음)
=====================================================================
"""

#%% [셀 0] 패키지 + 설계 읽기 =============================================
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

데이터폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "데이터"
CSV인코딩 = "utf-8-sig"

설계 = pd.read_csv(데이터폴더 / "유입베타_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
삼부설계 = pd.read_csv(데이터폴더 / "사건연구_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
미국사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩, dtype=str)
print("한국 설계 항목:", [k for k in 설계.index if k.startswith("한국")])


#%% [셀 1] DataGuide 종목·지수 읽기 ========================================
# 종목 파일 모양: 9번째 줄이 열 이름 (코드, 코드명, 유형, 아이템코드, 아이템명, 집계주기, 날짜...) - 11/17 확인
원 = pd.read_excel(데이터폴더 / "유입베타_DataGuide_한국_일별.xlsx", header=8)
설명열 = ["코드", "코드명", "유형", "아이템코드", "아이템명", "집계주기"]
날짜열 = [c for c in 원.columns if c not in 설명열]
날짜 = pd.DatetimeIndex(pd.to_datetime(pd.Series(날짜열).astype(str)))


def 아이템표(코드, 숫자=True):
    """아이템 하나를 날짜 × 종목 표로"""
    d = 원[원["아이템코드"] == 코드].set_index("코드")[날짜열].T
    d.index = 날짜
    return d.apply(pd.to_numeric, errors="coerce") if 숫자 else d


수익률 = 아이템표("S410007900") / 100            # % -> 소수 (설계 "한국_종목자료")
시총 = 아이템표("S410001250")                     # 백만원
정지 = 아이템표("S410002600", 숫자=False)
시장구분 = 원[원["아이템코드"] == "CP10000300"].set_index("코드")[날짜열[-1]]   # 현재(2025-12-30) 값
업종 = 원[원["아이템코드"] == "CP10001090"].set_index("코드")[날짜열[-1]]
이름 = 원.drop_duplicates("코드").set_index("코드")["코드명"]

# 거른 날 (출처: 설계 "한국_거래정지", "한국_가격제한")
제한 = float(설계["한국_가격제한"]) / 100
정지날 = (정지 == 설계["한국_거래정지"]) & 수익률.notna()
제한날 = (수익률.abs() > 제한) & ~정지날
수익률 = 수익률.mask(정지날 | 제한날)
print(f"종목 파일: {수익률.shape[1]:,}종목, {날짜.min().date()} ~ {날짜.max().date()} ({len(날짜)}일)")
print(f"  뺀 수익률: 거래정지 {int(정지날.sum().sum()):,}칸, |수익률|>{제한:.0%} {int(제한날.sum().sum()):,}칸")

# 지수 파일: 첫 칸이 날짜인 줄만 자료 (11/17 확인 코드와 같은 읽기)
지원 = pd.read_excel(데이터폴더 / "유입베타_DataGuide_지수_일별.xlsx", header=None)
첫칸 = pd.to_datetime(지원[0], errors="coerce", format="mixed")
이름줄 = 지원[지원[0].astype(str).str.strip() == "코드명"].iloc[0, 1:3].tolist()
지수 = 지원[첫칸.notna()].iloc[:, 1:3].apply(pd.to_numeric, errors="coerce") / 100
지수.index, 지수.columns = 첫칸[첫칸.notna()], 이름줄
지수 = 지수.reindex(날짜)
assert 지수.notna().all().all() and {"코스피", "코스닥"} <= set(지수.columns), "지수 자료가 종목 날짜와 맞지 않습니다"


#%% [셀 2] 요인을 한국 거래일에 맞춤 (IBIT·유입) ============================
거래일 = 날짜                                        # 한국 거래일 = DataGuide 날짜 (비영업일 제외로 받음)
# IBIT (출처: 설계 "한국_IBIT시점", 미국 수익률은 퀀트랩_사건연구_1 파일)
미국 = pd.read_csv(데이터폴더 / "사건연구_수익률_일별.csv", index_col="date", parse_dates=["date"], encoding=CSV인코딩)["IBIT"]
IBIT = []
for i, t in enumerate(거래일):
    앞 = 거래일[i - 1] if i > 0 else t - pd.Timedelta(days=1)
    구간 = 미국[(미국.index >= 앞) & (미국.index < t)]          # 직전 한국 거래일 ~ t 전날에 열린 미국 장들
    IBIT.append(np.prod(1 + 구간.to_numpy()) - 1 if len(구간) else 0.0)
요인 = pd.DataFrame({"코스피": 지수["코스피"], "코스닥": 지수["코스닥"], "IBIT": IBIT}, index=거래일)
요인.loc[거래일[0], "IBIT"] = np.nan                            # 첫날은 직전 한국 거래일이 없음

# 유입 (출처: 설계 "한국_유입시점" = 0 -> DefiLlama 날짜 t 값, 퀀트랩_BIS_4-1 합산 시가총액)
합산 = pd.read_csv(데이터폴더 / "확장_합산시총_일별.csv", parse_dates=["date"], index_col="date", encoding=CSV인코딩)["Total"] / 1e9
요인["시총_B"] = 합산.reindex(거래일 + pd.Timedelta(days=int(설계["한국_유입시점"]))).to_numpy()
요인["유입_B"] = 요인["시총_B"].diff()
요인.index.name = "date"


#%% [셀 3] 한국 0일 + 사건 창 ==============================================
# 출처: 설계 "한국_0일규칙", "한국_장시간" / 원사건은 퀀트랩_사건연구_0 "원사건" (시각 = 미국 동부)
개장, 마감 = [pd.Timedelta(h) for h in [s + ":00" for s in 설계["한국_장시간"].split(",")]]


def 다음거래일(날):
    return 거래일[거래일.searchsorted(날, side="right")]


def 한국영일(날짜, 시각):
    if not 시각:                                       # 시각 기록 없음 -> 다음 한국 거래일
        return 다음거래일(pd.Timestamp(날짜))
    동부 = pd.Timestamp(f"{날짜} {시각}").tz_localize(ZoneInfo("America/New_York"))
    한국 = 동부.tz_convert(ZoneInfo("Asia/Seoul")).tz_localize(None)
    하루, 시각한국 = 한국.normalize(), 한국 - 한국.normalize()
    # 그날이 거래일이고 장 마감 전이면 그날 (장중, 또는 개장 전이라 그날 장에서 처음 반영), 아니면 다음 거래일
    return 하루 if (하루 in 거래일 and 시각한국 < 마감) else 다음거래일(하루)


# 원사건 번호 -> 미국 사건 이름 (A~E) (출처: 사건연구_사건창.csv 의 원사건 열)
번호이름 = {n: r["이름"] for _, r in 미국사건창.iterrows() for n in r["원사건"].split("·")}
원사건 = pd.DataFrame([r.split(",") for r in 삼부설계["원사건"].split(";")], columns=["번호", "날짜", "시각", "결과"])
원사건["미국사건"] = 원사건["번호"].map(번호이름)
원사건["한국0일"] = [한국영일(d, t) for d, t in zip(원사건["날짜"], 원사건["시각"])]
원사건["창끝"] = [다음거래일(d) for d in 원사건["한국0일"]]           # 창 [0,+1]
원사건 = 원사건.sort_values("한국0일").reset_index(drop=True)

# 묶기 (3부 "묶음규칙": 창이 겹치면 하나로, 묶은 창 = 첫 0일 ~ 마지막 0일의 다음 거래일)
묶음, 지금 = [], None
for _, r in 원사건.iterrows():
    if 지금 is not None and r["한국0일"] <= 지금["창끝"]:
        지금["번호"].append(r["번호"]); 지금["결과"].append(r["결과"]); 지금["미국"].append(r["미국사건"])
        지금["창끝"] = max(지금["창끝"], r["창끝"])
    else:
        지금 = {"번호": [r["번호"]], "결과": [r["결과"]], "미국": [r["미국사건"]], "창시작": r["한국0일"], "창끝": r["창끝"]}
        묶음.append(지금)

# 이름: 미국 사건 이름을 따름. 미국 사건 하나가 한국에서 둘로 나뉘면 B1, B2 처럼 번호를 붙임
행 = []
for m in 묶음:
    미국들 = sorted(set(m["미국"]))
    행.append({"미국사건": "+".join(미국들), "원사건": "·".join(m["번호"]), "창시작": m["창시작"].date(),
               "창끝": m["창끝"].date(), "창길이": int(((거래일 >= m["창시작"]) & (거래일 <= m["창끝"])).sum()),
               "방향": -1 if "부결" in m["결과"] else +1})
사건창 = pd.DataFrame(행)
사건창["이름"] = 사건창["미국사건"]
for u, d in 사건창.groupby("미국사건"):
    if len(d) > 1:
        사건창.loc[d.index, "이름"] = [f"{u}{i + 1}" for i in range(len(d))]
판정사건 = 설계["판정사건"].split(",")                                   # A,B,C,E
사건창["판정에씀"] = 사건창["미국사건"].apply(lambda s: all(x in 판정사건 for x in s.split("+")))
print("원사건의 한국 0일:")
print(원사건[["번호", "날짜", "시각", "결과", "미국사건", "한국0일"]].to_string(index=False))
print("\n한국 사건 창 (3부 묶기 규칙 그대로):")
print(사건창[["이름", "미국사건", "원사건", "창시작", "창끝", "창길이", "방향", "판정에씀"]].to_string(index=False))
if 사건창["미국사건"].str.contains(r"\+").any():
    print("  [확인 필요] 미국에서 다른 사건이던 원사건이 한국에서 한 창으로 묶였습니다")


#%% [셀 4] 한국 추정 기간 ==================================================
# 출처: 설계 "한국_추정기간" - 한국 기준 사건 A 0일의 11 거래일 전에 끝나는 250일 (강건성 120일도 같은 끝날)
k = 거래일.get_loc(pd.Timestamp(사건창.loc[사건창["미국사건"] == "A", "창시작"].iloc[0]))
끝 = k - 11
기간행 = []
for 구분, 일수 in [("주(250일)", 250), ("강건성(120일)", 120)]:
    시작 = 끝 - 일수 + 1
    assert 시작 >= 1, f"{구분}: 자료가 모자랍니다 (첫날은 유입 변화가 없어 못 씀)"
    기간행.append({"구분": 구분, "시작": 거래일[시작].date(), "끝": 거래일[끝].date(), "일수": 일수})
추정기간 = pd.DataFrame(기간행).set_index("구분")
주 = 추정기간.loc["주(250일)"]
주시작, 주끝 = pd.Timestamp(주["시작"]), pd.Timestamp(주["끝"])
assert 주끝 < pd.Timestamp(사건창["창시작"].min()), "추정 기간과 사건 창이 겹칩니다"
필요 = (요인.index >= 주시작) & (요인.index <= pd.Timestamp(사건창["창끝"].max()))
assert 요인.loc[필요].notna().all().all(), "추정 기간·사건 창에 빈 요인 값이 있습니다"
print("한국 추정 기간:")
print(추정기간.to_string())
추정 = 요인.loc[주시작:주끝]
print(f"  유입 하루 변화 평균 {추정['유입_B'].mean():+.3f} $B, 표준편차 {추정['유입_B'].std():.3f} $B / "
      f"유입과 IBIT(한국 시점) 상관 {추정['유입_B'].corr(추정['IBIT']):+.3f}")


#%% [셀 5] 종목 거르기 + 유입 베타 + 5개 묶음 ===============================
시장들 = 설계["한국_시장"].split(",")                                   # KS, KQ
스팩 = 이름.astype(str).str.contains(설계["한국_제외이름"])
일수 = 수익률.loc[주시작:주끝].notna().sum()
단계 = [("전체", 수익률.shape[1])]
남음 = 시장구분.reindex(수익률.columns).isin(시장들)
단계.append(("코스피·코스닥", int(남음.sum())))
남음 &= ~스팩.reindex(수익률.columns, fill_value=False).astype(bool)
단계.append(("스팩 제외", int(남음.sum())))
남음 &= 일수 >= int(설계["한국_자료충분"])
단계.append((f"추정 기간 수익률 {설계['한국_자료충분']}일 이상", int(남음.sum())))
print("[종목 거르기]")
for 이름_, n in 단계:
    print(f"  {이름_}: {n:,}개")
종목 = 남음[남음].index
수익률, 시총 = 수익률[종목], 시총[종목]
시장지수 = {"KS": "코스피", "KQ": "코스닥"}                               # 설계 "한국_시장수익률"
내시장 = 시장구분[종목].map(시장지수)

# 종목별 유입 베타 (설계 "베타모형" - 시장은 자기 시장 지수)
행 = []
for c in 종목:
    y = 수익률.loc[주시작:주끝, c]
    X = 요인.loc[y.index, [내시장[c], "IBIT", "유입_B"]].assign(상수=1.0)[["상수", 내시장[c], "IBIT", "유입_B"]]
    있음 = y.notna().to_numpy()
    Xv, yv = X.to_numpy()[있음], y.to_numpy()[있음]
    계수, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    잔차 = yv - Xv @ 계수
    σ2 = 잔차 @ 잔차 / (len(yv) - 4)
    SE = np.sqrt(σ2 * np.linalg.inv(Xv.T @ Xv)[3, 3])
    행.append({"코드": c, "유입베타": 계수[3], "유입베타_t": 계수[3] / SE, "시장베타": 계수[1], "IBIT베타": 계수[2]})
베타 = pd.DataFrame(행).set_index("코드")
베타["묶음"] = pd.qcut(베타["유입베타"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
베타 = 베타.join(pd.DataFrame({"코드명": 이름[종목], "시장": 시장구분[종목], "업종": 업종[종목], "일수": 일수[종목]}))

print(f"\n유입베타 t 가 |2| 넘는 종목 비율: {(베타['유입베타_t'].abs() > 2).mean():.1%} (우연이면 약 5%, 미국은 4.4%)")
print("[묶음별 요약]")
print(베타.groupby("묶음").agg(종목수=("유입베타", "size"), 평균유입베타=("유입베타", "mean"),
                              평균IBIT베타=("IBIT베타", "mean"), 평균시장베타=("시장베타", "mean"),
                              코스닥비율=("시장", lambda s: (s == "KQ").mean())).to_string(float_format=lambda v: f"{v:.4f}"))
print("\n[묶음별 많은 업종]")
for g, d in 베타.groupby("묶음"):
    print(f"  묶음{g}: {d['업종'].value_counts().head(3).to_dict()}")


#%% [셀 6] 묶음 비정상 수익률 (추정 기간만) + 검정력 ========================
# 종목마다 3부 모형(자기 시장 + IBIT)을 추정 기간에 맞춰 추정 기간 안의 비정상 수익률 -> 묶음별 그날 평균
AR = pd.DataFrame(index=수익률.loc[주시작:주끝].index, columns=종목, dtype=float)
for c in 종목:
    y = 수익률.loc[AR.index, c]
    X = np.column_stack([np.ones(len(y)), 요인.loc[AR.index, 내시장[c]], 요인.loc[AR.index, "IBIT"]])
    있음 = y.notna().to_numpy()
    계수, *_ = np.linalg.lstsq(X[있음], y.to_numpy()[있음], rcond=None)
    AR[c] = y.to_numpy() - X @ 계수
묶음AR = pd.DataFrame({f"묶음{g}": AR[베타.index[베타["묶음"] == g]].mean(axis=1) for g in range(1, 6)})
묶음AR["차이"] = 묶음AR["묶음5"] - 묶음AR["묶음1"]

판정창 = 사건창[사건창["판정에씀"]]
창길이합, 사건수 = int(판정창["창길이"].sum()), len(판정창)
검정력 = pd.DataFrame({"잔차SD_일_%": 묶음AR.std(ddof=1) * 100})
검정력["S_표준오차_%"] = 검정력["잔차SD_일_%"] * np.sqrt(창길이합) / 사건수
검정력["최소검출효과_%"] = 2.8 * 검정력["S_표준오차_%"]
검정력.index.name = "묶음"
print(f"\n[검정력] 판정 사건 {list(판정창['이름'])}, 창 길이 합 {창길이합}일 - 추정 기간 자료만 사용")
print(검정력.to_string(float_format=lambda v: f"{v:.3f}"))


#%% [셀 7] 저장 ============================================================
사건창.to_csv(데이터폴더 / "유입베타_한국_사건창.csv", index=False, encoding=CSV인코딩)
추정기간.to_csv(데이터폴더 / "유입베타_한국_추정기간.csv", encoding=CSV인코딩)
수익률.rename_axis("date").to_csv(데이터폴더 / "유입베타_한국_수익률_일별.csv", encoding=CSV인코딩)
시총.rename_axis("date").to_csv(데이터폴더 / "유입베타_한국_시가총액_일별.csv", encoding=CSV인코딩)
요인.to_csv(데이터폴더 / "유입베타_한국_요인_일별.csv", encoding=CSV인코딩)
베타.to_csv(데이터폴더 / "유입베타_한국_베타.csv", encoding=CSV인코딩)
검정력.to_csv(데이터폴더 / "유입베타_한국_검정력.csv", encoding=CSV인코딩)
print("저장: 유입베타_한국_사건창·추정기간·수익률_일별·시가총액_일별·요인_일별·베타·검정력 .csv")
print("      (수익률·시가총액·베타 파일은 DataGuide 자료에서 나온 값 - 공유·업로드 금지)")
