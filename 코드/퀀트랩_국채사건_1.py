# -*- coding: utf-8 -*-
"""
퀀트랩_국채사건_1.py
=====================================================================
[연구] 국채 사건 연구: GENIUS Act 에 단기국채 스프레드와 스테이블코인 유입이 어떻게 반응했는가
[단계] 1단계: 자료 정리 + 사건 날짜·FOMC·추정 기간 확인

[이 코드가 하는 일]
  사건 창의 스프레드·유입 값은 아직 계산하지 않습니다 (검정력부터 - 설계 "검정력").
  ① 1부 확장 자료(퀀트랩_BIS_4-1)에서 3개월·6개월 금리와 합산 시가총액을 읽어 평일 자료로 정리
  ② 원 논문 그림 6 사건 4개의 0일을 국채 거래일 달력으로 정함 (설계 "재현_0일규칙")
  ③ GENIUS 사건 A~E 의 창 날짜가 모두 국채 거래일인지 확인 (주식 달력으로 만든 창이므로)
  ④ USMPD 로 FOMC 성명 발표일이 어느 사건 창과 겹치는지 다시 확인 (설계 "그림6_D표시", "판정사건")
  ⑤ 추정 기간(2024-04-24 ~ 2025-04-23)에 빈 값이 없는지 확인

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  국채사건_설계.csv                (퀀트랩_국채사건_0)
  확장_시장자료_원자료_일별.csv     (퀀트랩_BIS_4-1 [셀 2]: FRED 3개월·6개월 등, % 단위)
  확장_합산시총_일별.csv            (퀀트랩_BIS_4-1 [셀 1]: DefiLlama 7개 코인 합산, Total 열, 달러)
  확장_통화정책충격_USMPD.xlsx      (퀀트랩_BIS_4-1: SF 연준 USMPD, Statements 시트)
  사건연구_사건창.csv, 사건연구_추정기간.csv   (퀀트랩_사건연구_1)
[출력] 같은 폴더
  국채사건_자료_평일.csv     (평일 × 스프레드 bp, 3개월 금리 bp, 합산 시가총액 $B)
  국채사건_재현사건.csv      (논문 사건 4개의 0일)
  국채사건_FOMC확인.csv      (사건 창과 FOMC 성명일 대조)

[Spyder] "#%%" 셀 단위로 Shift+Enter
=====================================================================
"""

#%% [셀 0] 패키지 + 설계 읽기 =============================================
from pathlib import Path

import numpy as np
import pandas as pd

데이터폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "데이터"
CSV인코딩 = "utf-8-sig"

설계 = pd.read_csv(데이터폴더 / "국채사건_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
print("설계 항목 수:", len(설계))


#%% [셀 1] 평일 자료 만들기 ================================================
# 금리: 퀀트랩_BIS_4-1 [셀 2] 가 FRED 에서 받은 % 값 (열 이름 "3개월" = DGS3MO, "6개월" = DGS6MO)
시장 = pd.read_csv(데이터폴더 / "확장_시장자료_원자료_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)
# 국채 거래일 = 3개월·6개월 금리가 둘 다 있는 날 (FRED 는 채권시장 휴일에 값이 없음) - 설계 "날짜단위"
금리 = 시장[["3개월", "6개월"]].dropna()
자료 = pd.DataFrame(index=금리.index)
자료["스프레드_bp"] = (금리["3개월"] - 금리["6개월"]) * 100    # 설계 "스프레드자료" (% -> bp)
자료["3개월_bp"] = 금리["3개월"] * 100                          # 강건성 "3개월 금리 변화" 용

# 유입: DefiLlama 합산 시가총액 (출처: 퀀트랩_BIS_4-1 [셀 1], Total = 7개 코인 합, 달러 -> $B)
합산 = pd.read_csv(데이터폴더 / "확장_합산시총_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)["Total"] / 1e9
# 평일 d 에 붙일 값: 설계 "유입시점" 규칙대로 (DefiLlama 날짜는 그날 00:00 UTC 기록)
밀기 = int(설계["유입시점"])
자료["시총_B"] = 합산.reindex(자료.index + pd.Timedelta(days=밀기)).to_numpy()

자료 = 자료.dropna()
자료.index.name = "date"
print(f"평일 자료: {자료.index.min().date()} ~ {자료.index.max().date()}, {len(자료)}일 "
      f"(유입은 DefiLlama 날짜 +{밀기}일 값)")


#%% [셀 2] 논문 사건 4개의 0일 ============================================
# 출처: 설계 "재현_사건" (원 논문 그림 6 주석), "재현_0일규칙" (쉬는 날이면 다음 평일)
거래일 = 자료.index


def 이날또는다음(날):
    날 = pd.Timestamp(날)
    return 날 if 날 in 거래일 else 거래일[거래일.searchsorted(날)]


재현 = pd.DataFrame([r.split(",") for r in 설계["재현_사건"].split(";")], columns=["사건", "논문날짜", "방향"])
재현["0일"] = [이날또는다음(d).date() for d in 재현["논문날짜"]]
재현["바뀜"] = 재현["0일"].astype(str) != 재현["논문날짜"]
앞, 뒤 = map(int, 설계["재현_기간"].split(","))
for _, r in 재현.iterrows():                                     # ±15 평일이 자료 안에 있는지
    k = 거래일.get_loc(pd.Timestamp(r["0일"]))
    assert k + 앞 - 1 >= 0 and k + 뒤 < len(거래일), f"{r['사건']}: 사건 전후 자료가 모자람"
print("논문 사건의 0일 (국채 거래일 기준):")
print(재현.to_string(index=False))


#%% [셀 3] GENIUS 사건 창이 국채 거래일인지 확인 ==========================
# 출처: 설계 "사건자료" - 사건연구_사건창.csv (주식 거래일로 만든 창)
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)
문제 = []
for _, e in 사건창.iterrows():
    시작, 끝 = pd.Timestamp(e["창시작"]), pd.Timestamp(e["창끝"])
    if 시작 not in 거래일 or 끝 not in 거래일:
        문제.append(e["이름"])
    k = 거래일.get_loc(시작)
    assert k >= 16, "사건 전 자료가 모자람"
    # 창 안의 국채 거래일 수가 주식 창 길이와 같은지 (채권시장만 쉬는 날이 끼면 달라짐)
    n = int(((거래일 >= 시작) & (거래일 <= 끝)).sum())
    if n != int(e["창길이"]):
        문제.append(f"{e['이름']}(길이 {e['창길이']} -> {n})")
print("GENIUS 사건 창:", "모두 국채 거래일과 일치" if not 문제 else f"[확인 필요] {문제}")
print(사건창.to_string(index=False))


#%% [셀 4] FOMC 성명일과 사건 창 대조 ======================================
# 출처: 설계 "그림6_D표시", "D_FOMC확인" / USMPD Statements 시트 (퀀트랩_BIS_4-1 과 같은 파일)
U = pd.read_excel(데이터폴더 / "확장_통화정책충격_USMPD.xlsx", sheet_name="Statements")
FOMC = pd.to_datetime(U["Date"], errors="coerce").dt.normalize().dropna().drop_duplicates().sort_values()
행 = []
for _, e in 사건창.iterrows():
    시작, 끝 = pd.Timestamp(e["창시작"]), pd.Timestamp(e["창끝"])
    겹침 = FOMC[(FOMC >= 시작) & (FOMC <= 끝)]
    가까움 = FOMC[(FOMC >= 시작 - pd.Timedelta(days=7)) & (FOMC <= 끝 + pd.Timedelta(days=7))]
    행.append({"사건": e["이름"], "창시작": 시작.date(), "창끝": 끝.date(),
               "창안_FOMC": ",".join(str(d.date()) for d in 겹침) or "없음",
               "앞뒤7일_FOMC": ",".join(str(d.date()) for d in 가까움) or "없음"})
FOMC확인 = pd.DataFrame(행)
print("사건 창과 FOMC 성명일:")
print(FOMC확인.to_string(index=False))
판정사건 = 설계["판정사건"].split(",")
for _, r in FOMC확인.iterrows():
    if r["창안_FOMC"] != "없음" and r["사건"] in 판정사건:
        print(f"  [주의] 판정에 쓰는 사건 {r['사건']} 의 창에 FOMC 가 있습니다 -> 설계 확인 필요")
assert FOMC확인.set_index("사건").loc["D", "창안_FOMC"] != "없음", "설계와 달리 D 창에 FOMC 가 없습니다"


#%% [셀 5] 추정 기간 확인 + 저장 ===========================================
# 출처: 설계 "추정기간" (사건연구_추정기간.csv 의 주 250일)
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")
주 = 추정기간.loc["주(250일)"]
추정 = 자료.loc[주["시작"]:주["끝"]]
print(f"추정 기간 {주['시작']} ~ {주['끝']}: 국채 거래일 {len(추정)}일 (주식 거래일 {주['일수']}일), "
      f"빈 값 {int(추정.isna().sum().sum())}개")
# 주식과 채권의 휴일이 달라 일수가 조금 다를 수 있음 (예: 채권시장만 쉬는 콜럼버스 데이)

자료.to_csv(데이터폴더 / "국채사건_자료_평일.csv", encoding=CSV인코딩)
재현.to_csv(데이터폴더 / "국채사건_재현사건.csv", index=False, encoding=CSV인코딩)
FOMC확인.to_csv(데이터폴더 / "국채사건_FOMC확인.csv", index=False, encoding=CSV인코딩)
print("저장: 국채사건_자료_평일.csv, 국채사건_재현사건.csv, 국채사건_FOMC확인.csv")
