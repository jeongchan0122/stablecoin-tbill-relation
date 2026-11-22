# -*- coding: utf-8 -*-
"""
퀀트랩_유입베타_1.py
=====================================================================
[연구] 유입 베타 연구: 사건 전 '스테이블코인 유입과 같이 움직이던 정도'로 고른 주식이
       GENIUS Act 소식에 더 크게 반응했는가
[단계] 1단계: 자료 정리 + 종목 거르기

[이 코드가 하는 일]
  유입 베타·묶음·사건 반응은 아직 계산하지 않습니다 (2단계부터).
  ① CRSP 전 종목 파일을 읽고 보통주·거래소 조건으로 거름 (조건 값 분포를 먼저 출력)
  ② 1부 합산 시가총액에 유입시점 규칙(d+1)을 붙여 거래일별 '유입'(하루 변화, $B)을 만듦
  ③ 3부 파일에서 시장(vwretd)·IBIT 수익률을 가져와 붙임
  ④ 추정 기간 수익률 200일 이상 / 추정 기간 마지막 날 주가 5달러 이상 조건으로 종목을 거름
  ⑤ 저장: 종목 수익률·시가총액 (날짜 × PERMNO), 요인 (시장·IBIT·유입), 종목 목록

[입력] 바탕화면/스테이블코인_퀀트랩/데이터/
  유입베타_설계.csv                 (퀀트랩_유입베타_0)
  유입베타_CRSP_전종목_일별.csv      (WRDS CRSP Stock Version 2, Daily Stock File, 전 종목)
  확장_합산시총_일별.csv             (퀀트랩_BIS_4-1 [셀 1]: DefiLlama 7개 코인 합산, Total 열, 달러)
  사건연구_수익률_일별.csv, 사건연구_추정기간.csv, 사건연구_사건창.csv   (퀀트랩_사건연구_1)
[출력] 같은 폴더
  유입베타_수익률_일별.csv     (날짜 × PERMNO, 거른 종목만, 배당 포함 수익률)
  유입베타_시가총액_일별.csv   (날짜 × PERMNO, DlyCap - 강건성 '시가총액 가중' 용)
  유입베타_요인_일별.csv       (날짜 × 시장·IBIT·시총_B·유입_B)
  유입베타_종목목록.csv        (거른 종목의 티커·회사명·업종·추정 기간 수익률 일수·마지막 날 주가)
  ※ 위 수익률·시가총액 파일은 CRSP 에서 나온 자료라 공유·업로드 금지

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행 (CRSP 파일이 커서 [셀 1]이 1~2분 걸림)
[패키지]  pip install pandas numpy
=====================================================================
"""

#%% [셀 0] 패키지 + 폴더 + 설계 읽기 ======================================
from pathlib import Path

import numpy as np
import pandas as pd

데이터폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "데이터"
CSV인코딩 = "utf-8-sig"

# 설계 (출처: 퀀트랩_유입베타_0 - 값은 여기서 읽기만 함)
설계 = pd.read_csv(데이터폴더 / "유입베타_설계.csv", encoding=CSV인코딩, dtype=str).set_index("항목")["값"]
print("설계 항목 수:", len(설계))

# 추정 기간·사건 창 (출처: 설계 "추정기간", "사건자료" - 3부 파일 그대로)
추정기간 = pd.read_csv(데이터폴더 / "사건연구_추정기간.csv", encoding=CSV인코딩).set_index("구분")
사건창 = pd.read_csv(데이터폴더 / "사건연구_사건창.csv", encoding=CSV인코딩)
주시작, 주끝 = pd.Timestamp(추정기간.loc["주(250일)", "시작"]), pd.Timestamp(추정기간.loc["주(250일)", "끝"])
print(f"추정 기간 {주시작.date()} ~ {주끝.date()} / 마지막 사건 창 끝 {사건창['창끝'].max()}")


#%% [셀 1] CRSP 전 종목 읽기 + 보통주·거래소 조건 ==========================
# 출처: 설계 "종목자료" (DlyRet = 배당 포함 수익률), "보통주조건", "거래소조건"
열 = ["PERMNO", "DlyCalDt", "DlyRet", "DlyPrc", "DlyCap", "Ticker", "IssuerNm", "SICCD",
     "SecurityType", "SecuritySubType", "ShareType", "USIncFlg", "PrimaryExch"]
원본 = pd.read_csv(데이터폴더 / "유입베타_CRSP_전종목_일별.csv", usecols=열, parse_dates=["DlyCalDt"],
                   dtype={"PERMNO": int, "SICCD": str}, low_memory=False)
print(f"CRSP 원본: {len(원본):,}행, 종목 {원본['PERMNO'].nunique():,}개, "
      f"{원본['DlyCalDt'].min().date()} ~ {원본['DlyCalDt'].max().date()}")

# 조건 값이 설계와 같은 글자로 들어 있는지 먼저 봄 (CRSP Version 2 값 확인용)
for c in ["SecurityType", "SecuritySubType", "ShareType", "USIncFlg", "PrimaryExch"]:
    print(f"  {c}: {원본[c].value_counts().head(8).to_dict()}")

보통 = dict(kv.split("=") for kv in 설계["보통주조건"].replace(" ", "").split(","))   # SecurityType=EQTY ...
거래소 = 설계["거래소조건"].split("=")[1].replace(" ", "").split(",")                # N, A, Q
조건 = np.logical_and.reduce([원본[k] == v for k, v in 보통.items()]) & 원본["PrimaryExch"].isin(거래소)
종목행 = 원본[조건].copy()
print(f"\n보통주·거래소 조건 통과: {len(종목행):,}행, 종목 {종목행['PERMNO'].nunique():,}개")

# 같은 날 같은 PERMNO 가 두 줄인 경우: 읽은 열 값이 모두 같으면 한 줄만 남김
#   (11/19 확인: 69종목 243줄, 모든 열 값이 같은 중복 - WRDS 내보내기에서 생긴 것)
#   값이 하나라도 다른 중복이 있으면 다른 증권이 섞인 것이므로 멈춤
완전중복 = 종목행.duplicated(keep="first")
종목행 = 종목행[~완전중복]
assert not 종목행.duplicated(["DlyCalDt", "PERMNO"]).any(), "값이 다른데 같은 날 같은 종목인 줄이 있습니다 - 확인 필요"
print(f"완전히 같은 중복 줄 {int(완전중복.sum())}개 제거 -> {len(종목행):,}행")


#%% [셀 2] 유입 자료 (거래일별 하루 변화, $B) ==============================
# 출처: 설계 "유입자료", "유입시점" - 퀀트랩_국채사건_1 [셀 1] 과 같은 방법
거래일 = pd.DatetimeIndex(np.sort(원본["DlyCalDt"].unique()))       # CRSP 에 있는 날 = 미국 주식 거래일
합산 = pd.read_csv(데이터폴더 / "확장_합산시총_일별.csv", parse_dates=["date"], index_col="date",
                   encoding=CSV인코딩)["Total"] / 1e9                # 달러 -> $B (출처: 퀀트랩_BIS_4-1)
밀기 = int(설계["유입시점"])
요인 = pd.DataFrame(index=거래일)
요인.index.name = "date"
요인["시총_B"] = 합산.reindex(거래일 + pd.Timedelta(days=밀기)).to_numpy()   # 거래일 d 에 DefiLlama d+1 값
요인["유입_B"] = 요인["시총_B"].diff()                                 # 전 거래일 대비 (주말 몫은 월요일에 들어감)


#%% [셀 3] 시장·IBIT 붙이기 + 빈 날 확인 ===================================
# 출처: 설계 "시장·비트코인자료" - 3부의 시장(vwretd)·IBIT 열
삼부 = pd.read_csv(데이터폴더 / "사건연구_수익률_일별.csv", index_col="date", parse_dates=["date"],
                   encoding=CSV인코딩)
요인["시장"] = 삼부["시장"].reindex(요인.index)
요인["IBIT"] = 삼부["IBIT"].reindex(요인.index)

# 추정 기간 + 사건 창에서 요인이 하나도 비면 안 됨
필요 = (요인.index >= 주시작) & (요인.index <= pd.Timestamp(사건창["창끝"].max()))
빈칸 = 요인.loc[필요, ["시장", "IBIT", "유입_B"]].isna().sum()
assert (빈칸 == 0).all(), f"추정 기간·사건 창에 빈 요인 값: {빈칸[빈칸 > 0].to_dict()}"

# 3부와 거래일 달력이 같은지 (추정 기간 250일이 그대로 나와야 함)
주일수 = int(((거래일 >= 주시작) & (거래일 <= 주끝)).sum())
assert 주일수 == int(추정기간.loc["주(250일)", "일수"]), f"추정 기간 거래일 수가 3부와 다름: {주일수}"
추정 = 요인.loc[주시작:주끝]
print(f"요인 자료: 추정 기간 {주일수}일 빈 값 없음")
print(f"  유입 하루 변화: 평균 {추정['유입_B'].mean():+.3f} $B, 표준편차 {추정['유입_B'].std():.3f} $B "
      f"(국채 연구 2단계와 비슷해야 함 - 달력이 조금 달라 완전히 같지는 않음)")
print(f"  유입과 IBIT 의 상관 (추정 기간): {추정['유입_B'].corr(추정['IBIT']):+.3f}")


#%% [셀 4] 종목 거르기: 자료 충분 + 주가 하한 ==============================
# 출처: 설계 "자료충분" (추정 기간 수익률 200일 이상), "주가하한" (추정 기간 마지막 날 |주가| 5달러 이상)
수익률 = 종목행.pivot(index="DlyCalDt", columns="PERMNO", values="DlyRet").reindex(거래일)
시총 = 종목행.pivot(index="DlyCalDt", columns="PERMNO", values="DlyCap").reindex(거래일)
주가 = 종목행.pivot(index="DlyCalDt", columns="PERMNO", values="DlyPrc").reindex(거래일)
수익률.index.name = 시총.index.name = "date"

일수 = 수익률.loc[주시작:주끝].notna().sum()
마지막주가 = 주가.loc[주끝].abs()          # CRSP 는 종가가 없으면 호가 평균을 음수로 적던 관례가 있어 절댓값
하한일수, 하한주가 = int(설계["자료충분"]), float(설계["주가하한"])

단계 = [("보통주·거래소 조건 통과", 수익률.columns.size),
        (f"추정 기간 수익률 {하한일수}일 이상", int((일수 >= 하한일수).sum()))]
남음 = (일수 >= 하한일수) & (마지막주가 >= 하한주가)
단계.append((f"추정 기간 마지막 날({주끝.date()}) 주가 {하한주가:.0f}달러 이상", int(남음.sum())))
print("[종목 거르기]")
for 이름, n in 단계:
    print(f"  {이름}: {n:,}개")
print(f"  (마지막 날 주가가 비어 빠진 종목: {int(((일수 >= 하한일수) & 마지막주가.isna()).sum())}개)")

남은종목 = 남음[남음].index
수익률, 시총 = 수익률[남은종목], 시총[남은종목]

# 사건 기간에 상장 폐지 등으로 수익률이 끊긴 종목 수 (설계 "묶음가중" - 그날부터 평균에서 빠짐)
사건구간 = 수익률.loc[pd.Timestamp(사건창["창시작"].min()):pd.Timestamp(사건창["창끝"].max())]
print(f"  사건 기간(첫 창 시작 ~ 마지막 창 끝)에 수익률이 하루라도 빈 종목: {int(사건구간.isna().any().sum())}개")


#%% [셀 5] 종목 목록 + 저장 ================================================
마지막행 = 종목행[종목행["DlyCalDt"] <= 주끝].sort_values("DlyCalDt").groupby("PERMNO").last()
목록 = 마지막행.loc[남은종목, ["Ticker", "IssuerNm", "SICCD", "PrimaryExch"]].copy()
목록["추정기간_수익률일수"] = 일수[남은종목]
목록["마지막날_주가"] = 마지막주가[남은종목]
목록.index.name = "PERMNO"
print("\n거래소별 종목 수:", 목록["PrimaryExch"].value_counts().to_dict())

수익률.to_csv(데이터폴더 / "유입베타_수익률_일별.csv", encoding=CSV인코딩)
시총.to_csv(데이터폴더 / "유입베타_시가총액_일별.csv", encoding=CSV인코딩)
요인.to_csv(데이터폴더 / "유입베타_요인_일별.csv", encoding=CSV인코딩)
목록.to_csv(데이터폴더 / "유입베타_종목목록.csv", encoding=CSV인코딩)
print("저장: 유입베타_수익률_일별.csv, 유입베타_시가총액_일별.csv, 유입베타_요인_일별.csv, 유입베타_종목목록.csv")
print("      (수익률·시가총액 파일은 CRSP 자료 - 공유·업로드 금지)")
