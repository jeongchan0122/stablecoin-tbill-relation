# -*- coding: utf-8 -*-
"""
퀀트랩_BIS_3.py
=====================================================================
[논문] Ahmed & Aldasoro (2026), "Stablecoins and safe asset prices",
       BIS Working Paper No 1270 (2026년 6월 개정판)
[단계] 복제 3단계: 도구변수(GIV)의 배제 조건 점검
       논문 2.3절 "Exclusion" 문단 + 부록 C.3(표 6) + 부록 C.4(표 7)의 네 가지 근거를 재현
         ① 구성상 직교      (2.3절 "By construction, the granular IV is ... after the
                               size-weighted common factor has been removed")
         ② 주성분 분석      (2.3절 "the first principal component explains only about 16%
                               of cross-sectional variance")
         ③ 2원 분산 분해    (2.3절 끝 + 부록 C.4 표 7)
         ④ 교란변수 상관    (2.3절 + 부록 C.3 표 6) - 무료 자료만 (출처: 사용자 지시)

[출처 표기 규칙]
  # 출처: 논문 ○○절 / 각주 ○ / 표 ○ 주석   -> 논문 근거
  # 출처: 사용자 지시                       -> 연구자가 정한 조건
  # 출처: 퀀트랩_BIS_2.py                   -> 2단계에서 논문 수치로 확정한 사항
  # 출처: 저자 답신                         -> 논문 저자(Aldasoro, Ahmed 공동)가 2026년 10월
                                              이메일로 확인해 준 구현 방식
  # [논문에 명시 없음]                      -> 논문에 방법이 안 나와 있는 부분
                                              (이유와 영향을 함께 적음)

[수정 기록]
  2026-10-12 저자 답신 반영 (출처: 저자 답신, 사용자 지시)
    [셀 1] 윈저화 경계의 중심: 중앙값 -> 0  (MAD 계산은 그대로 풀링 중앙값 기준)
    [셀 2] GIV 식의 1/N 에서 N: "t−1·t 모두 활성" -> "t−1 활성"

[공통 주의] (출처: 논문 부록 C.1, 표 2 (3)열 / 사용자 지시)
  본 연구는 DeFiLlama-only 패널이다.
  [논문에 명시 없음] 논문은 ②·③·④를 기준 패널(Token Terminal 12개 페어 대체 포함)과
  DeFiLlama-only 패널 중 어느 것으로 계산했는지 밝히지 않는다.
  -> 논문 숫자와 소수점까지 같을 필요는 없고, "같은 결론"이 나오는지가 비교의 기준

[입력] (출처: 사용자 지시 - 앞 단계 CSV를 불러와 씀)
  바탕화면/스테이블코인_퀀트랩/데이터/코인체인_패널_일별.csv   (퀀트랩_BIS_1.py, 종료일 2026-03-31)
  바탕화면/스테이블코인_퀀트랩/데이터/GIV_선택조합_평일.csv    (퀀트랩_BIS_2.py [셀 6])

[Spyder] "#%%" 셀 단위로 Shift+Enter, [셀 0]부터 순서대로 실행
[패키지]  pip install pandas numpy statsmodels requests yfinance
=====================================================================
"""

#%% [셀 0] 패키지 + 설정 =================================================
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

# ---------------------------------------------------------------
# 폴더 (출처: 사용자 지시 - 바탕화면/스테이블코인_퀀트랩, 한국어 파일명)
# ---------------------------------------------------------------
결과폴더 = Path.home() / "Desktop" / "스테이블코인_퀀트랩"
데이터폴더 = 결과폴더 / "데이터"
CSV인코딩 = "utf-8-sig"

# ---------------------------------------------------------------
# FRED API 키 (출처: 사용자 지시 - FRED API 사용)
# 키는 코드에 적지 않고 프로젝트 폴더의 FRED_키.txt 에서 읽음 (퀀트랩_BIS_1.py 와 같음)
# ---------------------------------------------------------------
_키파일 = Path.home() / "Desktop" / "스테이블코인_퀀트랩" / "FRED_키.txt"   # GitHub에 올리지 않는 파일
FRED_API_KEY = os.environ.get("FRED_API_KEY") or (
    _키파일.read_text(encoding="utf-8").strip() if _키파일.exists() else "여기에_FRED_키_붙여넣기")
FRED주소 = "https://api.stlouisfed.org/fred/series/observations"

# ---------------------------------------------------------------
# 표본 기간 (출처: 논문 표 1 주석 "Sample: January 4, 2021 to March 31, 2026")
# ---------------------------------------------------------------
표본시작 = pd.Timestamp("2021-01-04")
표본끝 = pd.Timestamp("2026-03-31")

# 평일 달력 (출처: 퀀트랩_BIS_2.py [셀 0] - 표 1·표 5 의 T = 1,367 로 역산)
평일 = pd.bdate_range(표본시작, 표본끝)

# 각주 18의 기준 (출처: 각주 18)
활성기준 = 1e6                 # "exceeds $1 million on both t − 1 and t"
최소N = 5                      # "require N_t ≥ 5 for GIV_t to be defined"
윈저배수 = 8 * 1.4826          # "robust-winsorized at ±8 × 1.4826 × MAD"

print("데이터 폴더:", 데이터폴더)
print("평일 수:", len(평일), " (표 1·표 5: 1,367)")


#%% [셀 1] 코인-체인 패널 → 성장률·활성·윈저화 (②·③의 재료) ===============
# 2단계와 똑같은 방식으로 다시 계산합니다 (출처: 퀀트랩_BIS_2.py [셀 2]·[셀 3]·[셀 6])
# 2단계와 같은 방식으로 성장률, 활성 여부, 윈저화를 다시 계산합니다. 분석일 수를 표 7의 1,913일과 비교해 출력
패널 = pd.read_csv(데이터폴더 / "코인체인_패널_일별.csv", parse_dates=["date"], encoding=CSV인코딩)
패널 = 패널[패널["date"] <= 표본끝]

# 날짜 × 페어 표, 달력일 (출처: 퀀트랩_BIS_2.py [셀 2])
# 1단계가 저장한 코인×체인 패널을 읽어서, 행이 날짜(달력일)이고 열이 페어인 표로 바꿉니다. 2단계 [셀 2]와 같습니다.
시총표 = 패널.pivot_table(index="date", columns="pair", values="mcap", aggfunc="sum")
시총표 = 시총표.reindex(pd.date_range(시총표.index.min(), 표본끝, freq="D"))

# 성장률(%p)과 활성 여부 (출처: 2.3절, 표 2 주석, 활성의 정의는 각주 18 / 퀀트랩_BIS_2.py [셀 2])
# [논문에 명시 없음] 성장률 공식 -> 2단계와 같은 단순 성장률 (퀀트랩_BIS_2.py 주석 참고)
전날 = 시총표.shift(1)
g = (시총표 - 전날) / 전날.where(전날 > 0) * 100
활성 = (시총표 > 활성기준) & (전날 > 활성기준)

# 패널 = 한 번이라도 활성이었던 226개 페어 (출처: 퀀트랩_BIS_2.py [셀 2], 2.2절 226개·각주 17 117개와 일치)
패널페어 = 활성.columns[활성.any()]
시총표, g, 활성, 전날 = 시총표[패널페어], g[패널페어], 활성[패널페어], 전날[패널페어]

# 윈저화 (출처: 저자 답신, 퀀트랩_BIS_2.py [셀 6]과 같음)
#   MAD 범위 = 전체 : 저자 답신 - 모든 날짜·모든 코인체인 페어의 일별 성장률을 한데 모아 MAD 계산
#   MAD 기준 = 중앙값: 저자 답신 - 모아 놓은 전체의 중앙값으로부터의 절대편차 중앙값
#   경계 중심 = 0    : 저자 답신 - 윈저화 경계는 중앙값이 아니라 0 을 중심으로 ±8 × 1.4826 × MAD
#   MAD 표본 = 활성  : 저자 답신 - t−1 과 t 모두 시총이 100만 달러를 넘는 페어만 MAD 계산용 성장률 표본에 넣음
#   배수 8 × 1.4826  : 각주 18
# 출처: 표 7 주석 "we within-day demean the winsorized growth-rate panel" -> ③은 윈저화된 성장률 사용
# ③분산 분해는 표 7 주석에 "winsorized growth-rate panel"이라고 명시돼 있어서 윈저화된 gw를 씁니다. 
# ②주성분 분석은 논문에 명시가 없어서 [셀 3]에서 윈저화 적용·미적용을 둘 다 계산
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    x = g.where(활성).values
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
# 경계는 0 중심 (출처: 저자 답신). med 는 MAD 계산에만 쓰임
gw = g.clip(lower=-윈저배수 * mad, upper=윈저배수 * mad)

# 코인·체인 이름 ("USDT / Tron" -> "USDT", "Tron")  (출처: 퀀트랩_BIS_1.py [셀 3]의 pair 표기)
# "USDT / Tron"을 "USDT"와 "Tron"으로 나눕니다. ③분산 분해에서 코인 더미와 체인 더미를 만들 때 필요합니다.
페어코인 = pd.Series({p: p.split(" / ", 1)[0] for p in 패널페어})
페어체인 = pd.Series({p: p.split(" / ", 1)[1] for p in 패널페어})

# ---------------------------------------------------------------
# ②·③의 날짜 표본 = 2021-01-04 ~ 2026-03-31 달력일 중 활성 페어가 5개 이상인 날
# 출처: 표 7 "Unit-day observations 166,019 / Trading days 1,913"
#       각주 18 "require N_t ≥ 5 for GIV_t to be defined"
#   퀀트랩_BIS_2.py 에서 달력일 기준 N_t ≥ 5 인 날이 1,915일(2021-01-01부터)이었고,
#   표본 시작 전 1월 2·3일을 빼면 1,913일 -> 표 7의 1,913일과 같은지 아래에서 출력해 확인
#   (표 7은 이 날들을 "trading days"라고 부르지만 숫자는 달력일 기준과 맞음)
# ---------------------------------------------------------------
N = 활성.sum(axis=1)
분석일 = N.index[(N.index >= 표본시작) & (N.index <= 표본끝) & (N >= 최소N)]
print(f"페어 수: {len(패널페어)}  (2.2절: 226)")
print(f"분석일 수: {len(분석일)}  (표 7: 1,913)")
print(f"활성 페어-일: {int(활성.loc[분석일].values.sum()):,}  (표 7: 166,019)")
print(f"윈저화 경계: ±{윈저배수 * mad:.2f} (%p)  (0 중심, 출처: 저자 답신)")


#%% [셀 2] ① 구성상 직교 확인 ===============================================
# 출처: 2.3절 "By construction, the granular IV is a share-weighted cross-section of
#       unit-level idiosyncratic residuals after the size-weighted common factor has been removed"
# 출처: 1장·2.3절 "orthogonal to any common factor in stablecoin growth"
# 논문은 이것을 계산이 아니라 식 (2)의 구조로 설명합니다.
# 이 셀은 논문 결과를 재현한다기보다, 우리 GIV가 그 구조를 실제로 갖추고 있는지 점검하는 셀
# 여기서는 실제 데이터로 그 구조가 성립하는지 두 가지로 확인합니다.
#   (가) 가중치 합: Σ_i (s_i,t−1 − 1/N_t)  -> [참고용] 판정에 쓰지 않음
#        예전 N 정의(양일 활성)에서는 늘 0 이었지만, 저자 N 정의(t−1 활성)에서는 t 에 빠진 페어가 있는 날 0 이 아님
#        공통 충격이 걸러지는 이유는 이 합이 0 이어서가 아니라 ε = g − ḡ 에서 ḡ 를 빼기 때문
#        (Σs = 1 이므로 모든 g 에 c 를 더하면 ḡ 도 c 만큼 커져 ε 는 그대로) -> ① 판정은 (나)로 함
#   (나) 실제로 모든 활성 페어의 성장률에 날마다 다른 공통 충격을 더해도 GIV_t 가 그대로인지
# 식 (2) (출처: 2.3절 / 퀀트랩_BIS_2.py [셀 3] GIV_일별 함수와 동일)
# N 의 정의 (출처: 저자 답신 - GIV 의 동일가중 조정(1/N)에 쓰는 활성 페어 수 N 은 t−1 의 활성 여부만으로 셈)
#   -> 성장률 표본(t−1·t 모두 활성, 각주 18)과 N 을 세는 기준(t−1 활성)이 다름
# [논문·저자 답신에 명시 없음] 점유율 s 의 분모 -> 퀀트랩_BIS_2.py [셀 3]과 같게
#   "t−1·t 모두 활성인 페어의 전날 시총 합" (BIS_2 와 같은 GIV 를 만들기 위함)
# 가중치 합 Σ(s − 1/N) 에 대해 (위 두 정의에서 나오는 수식상 결과):
#   s 의 합 = 1, 1/N 은 양일 활성 페어 수(n)만큼 더해지므로 Σ(s − 1/N) = 1 − n/N
#   -> t−1 에는 활성이었지만 t 에 $1M 이하가 된 페어가 없는 날: n = N 이라 합 = 0
#   -> 그런 페어가 있는 날: n < N 이라 합 > 0 (오류가 아니라 저자 정의에서 나오는 값)
# 출력되는 최대 절댓값이 1e-15 정도(컴퓨터 계산 오차 수준)면 모든 날 n = N 이라는 뜻
def GIV_일별(g_, 활성_, 전날_):
    s = 전날_.where(활성_)
    s = s.div(s.sum(axis=1), axis=0)
    gg = g_.where(활성_)
    N_ = (전날_ > 활성기준).sum(axis=1)   # 출처: 저자 답신 - N 은 t−1 활성만으로 셈 (퀀트랩_BIS_2.py 와 같음)
    g_bar = (s * gg).sum(axis=1)
    eps = gg.sub(g_bar, axis=0)
    giv = (s.sub(1 / N_, axis=0) * eps).sum(axis=1)
    giv[N_ < 최소N] = np.nan   # [저자 답신에 명시 없음] N ≥ 5 를 t−1 활성 N 에 적용 (퀀트랩_BIS_2.py [셀 3]과 같음)
    return giv, s.sub(1 / N_, axis=0)
    


giv_원래, 가중치 = GIV_일별(gw, 활성, 전날)
가중치합 = 가중치.sum(axis=1).loc[분석일]
print(f"(가) 가중치 합의 최대 절댓값: {가중치합.abs().max():.2e}  (0 이면 구조 성립)")

# 공통 충격: 날마다 다른 값 (평균 0, 표준편차 10%p) - 크기는 확인용일 뿐 결과와 무관
# (나) 공통 충격을 실제로 더해 보기
# 날마다 모든 활성 페어의 성장률에 같은 값을 더합니다. 금리 변화나 위험선호 변화처럼 시장 전체에 똑같이 닥친 충격을 흉내 낸 것입니다.
# 그다음 GIV를 다시 계산해 원래 GIV와 비교합니다. 차이도 1e-15 정도면 정상입니다.
# 난수의 크기(10%p)는 확인용일 뿐이고, 어떤 크기를 넣어도 결과는 0이어야 합니다. 이 난수 설정은 논문에 있는 게 아니라 점검용 장치
공통충격 = pd.Series(np.random.default_rng(0).normal(0, 10, len(gw)), index=gw.index)
giv_충격, _ = GIV_일별(gw.add(공통충격, axis=0), 활성, 전날)
차이 = (giv_충격 - giv_원래).loc[분석일].abs().max()
print(f"(나) 공통 충격을 더한 뒤 GIV 변화의 최대 절댓값: {차이:.2e}  (0 이면 공통 충격이 걸러짐)")

# 한계 (논문 2.3절의 논리 그대로): 이 성질은 "모든 페어에 같은 크기"로 닥친 공통 충격만 걸러냄.
# 페어마다 크기가 다른 공통 요인이 얼마나 되는지는 ②·③이, 외부 변수와의 관계는 ④가 다룸.
# 이 점검이 보여주지 못하는 것
# 걸러지는 건 모든 페어에 똑같은 크기로 닥친 충격뿐입니다. 
# 예를 들어 금리가 오를 때 USDT/Tron이 유독 크게 반응하는 식으로, 공통 요인이 페어마다 다르게 작용하면 GIV에 섞일 수 있습니다. 
# 그래서 논문도 ②주성분 분석(그런 공통 요인이 얼마나 큰가), ③분산 분해(변동이 어디서 오는가), ④교란변수 상관(외부 변수와 관련 있는가)을 추가로 제시

# [저자 답신 반영 전 결과] 둘 다 사실상 0이 나왔다. 날마다 표준편차 10%짜리 공통충격을 넣었는데도 GIV는 움직이지 않음
# [저자 답신 반영 후 결과] (나) 1.17e-15 -> 공통 충격은 그대로 걸러짐
#   (가) 3.44e-01 -> 저자 답신의 N 정의(t−1 활성)에서는 가중치 합이 0 이 아닐 수 있음.
#   최대값 날은 2024-04-18 (DefiLlama 체인 자료가 31개 페어에서 빈 날). 자료가 빈 날 100일 / 1,913일,
#   빠진 페어의 전날 시총 비중 최대 0.9%

# 진단: t−1 활성 페어 중 t에 성장률 표본에서 빠진 페어가 있는 날 확인
# 출처: 사용자 지시 - 자료 점검용 (논문에 없는 계산, 결과에 영향 없음)
N_t1 = (전날 > 활성기준).sum(axis=1)                    # t−1 활성 페어 수 (저자 답신의 N)
n_양일 = 활성.sum(axis=1)                               # t−1·t 모두 활성인 페어 수 (각주 18)
값없음 = ((전날 > 활성기준) & 시총표.isna()).sum(axis=1)  # t−1 활성인데 t 자료가 비어 있는 페어 수
진단 = pd.DataFrame({"N(t−1 활성)": N_t1, "n(양일 활성)": n_양일, "t 자료 없음": 값없음,
                     "빠진 비율": 1 - n_양일 / N_t1}).loc[분석일]
다른날 = 진단[진단["N(t−1 활성)"] != 진단["n(양일 활성)"]]
print(f"두 수가 다른 날: {len(다른날)}일 / {len(분석일)}일")
print(다른날.sort_values("빠진 비율", ascending=False).head(10))
#%% [셀 3] ② 주성분 분석 =================================================
# 출처: 2.3절 "A principal-components decomposition of the underlying stablecoin-chain
#       growth-rate panel confirms that the dominant common factor is modest: the first
#       principal component explains only about 16% of cross-sectional variance"
# 목표값: 첫 번째 주성분의 설명 비율 ≈ 16%
# 주성분 분석이 뭘 보려는 건가
# 226개 페어의 성장률이 매일 오르내리는데, 이 움직임 중 모든 페어가 한 방향으로 같이 움직이는 부분이 얼마나 되는지를 봅니다.
# 첫 번째 주성분: 페어들을 가장 많이 함께 움직이게 하는 하나의 공통 요인입니다. 가중치는 페어마다 다를 수 있습니다.
# 설명 비율이 크면: 페어들이 대부분 같이 움직여서 페어 고유의 움직임이 적다는 뜻입니다. 그러면 GIV가 쓸 재료가 부족하고, 공통 요인이 GIV에 섞일 위험도 커집니다.
# 논문의 16%: 공통 요인이 작고 나머지 84%가 페어별로 따로 움직인다는 뜻입니다. GIV가 쓸 고유 변동이 충분하다는 근거입니다.
# [셀 2]가 "모든 페어에 같은 크기로 닥친 충격"을 봤다면, 이 셀은 "페어마다 다른 크기로 작용하는 공통 요인"이 얼마나 큰지를 봅니다.
#
# [논문에 명시 없음] 아래 세 가지 -> 모든 경우(2 × 2 × 2 = 8가지) 계산 (출처: 사용자 지시 -
#   논문에 없는 설정은 가능한 경우를 모두 계산)
#   (가) 윈저화: 적용 / 미적용
#        - 2.3절은 "underlying growth-rate panel"이라고만 하고, 표 7(③)은 "winsorized"라고 명시
#   (나) 불균형 패널 처리: 페어마다 존재 기간이 달라 빈칸이 있음. 주성분 분석은 빈칸을 못 다룸
#        - "0 채움"  : 페어별 평균을 뺀 뒤 빈칸(비활성)을 0 으로 채움 (평균값으로 채우는 것과 같음)
#        - "균형패널": 분석일 전체에서 매일 활성인 페어만 사용, 1,913일 동안 하루도 빠짐없이 활성인 페어만 씁니다.
#   (다) 행렬: 공분산 행렬 / 상관 행렬(페어별 표준화)
#              공분산: 원래 크기 그대로 씁니다. 성장률이 크게 출렁이는 작은 페어의 영향이 커집니다.
#              상관: 페어마다 표준편차로 나눠서 모든 페어를 같은 비중으로 봅니다.
# 공통: 관측치 = 분석일(표 7과 같은 날짜) × 그날 활성인 페어 (출처: 각주 18 - GIV 에 들어가는 단위)
#
# 계산 순서
# 분석일(1,913일) × 226개 페어 표를 만듭니다. 비활성인 날은 빈칸 (빈칸 처리 방식에 따라 페어를 고른다-(나)가 처리방식)
# 페어별 평균을 빼고 상관방식이면 표준편차로 나눈다. 그리고 빈칸을 0으로 채운다(균형패널은 빈칸 x)
# 그 후 공분산 행렬의 고유값을 구한다. 가장 큰 고유값 ÷ 고유값 합 × 100이 첫 번째 주성분의 설명 비율
def 첫주성분비율(행렬, 방식_결측, 방식_행렬):
    X = 행렬.copy()
    if 방식_결측 == "균형패널":
        X = X.loc[:, X.notna().all(axis=0)]
    else:
        X = X.loc[:, X.notna().sum(axis=0) >= 2]      # 평균·표준편차를 구하려면 최소 2일 필요 (수학적 조건)
    X = X - X.mean(axis=0)
    if 방식_행렬 == "상관":
        sd = X.std(axis=0)
        X = X.loc[:, sd > 0] / sd[sd > 0]
    X = X.fillna(0.0)
    공분산 = np.cov(X.values, rowvar=False)
    고유값 = np.linalg.eigvalsh(공분산)
    return 고유값[-1] / 고유값.sum() * 100, X.shape[1]


주성분결과 = []
for 윈저 in ["적용", "미적용"]:
    원자료 = (gw if 윈저 == "적용" else g).where(활성).loc[분석일]
    for 방식_결측 in ["0 채움", "균형패널"]:
        for 방식_행렬 in ["공분산", "상관"]:
            비율, 페어수 = 첫주성분비율(원자료, 방식_결측, 방식_행렬)
            주성분결과.append({"윈저화": 윈저, "결측처리": 방식_결측, "행렬": 방식_행렬,
                           "사용페어수": 페어수, "첫주성분_설명비율(%)": round(비율, 1)})

주성분결과 = pd.DataFrame(주성분결과)
주성분결과.to_csv(데이터폴더 / "배제조건_주성분분석.csv", index=False, encoding=CSV인코딩)
print("논문 2.3절: 첫 번째 주성분 ≈ 16%\n")
print(주성분결과.to_string(index=False))

# ② 주성분 분석 정리 [저자 답신 반영 후 결과 - 반영 전과 같음]
# 설정	                     결과 	                      판단
# 공분산 33%, 43~48%	  오류 페어 하나가 지배	공통 요인의 척도로 쓸 수 없음
# 상관 행렬(4개)	           2.0~20.4%	     페어 수 대비 독립 기준값 수준이거나 조금 큼
# 윈저화 적용,0 채움,공분산	 3.3%	                  공통 요인 작음
# 결론: 16%라는 숫자는 재현하지 못했습니다. 하지만 오류 페어의 영향을 받지 않는 설정에서는 모두 "지배적인 공통 요인이 작다"는 논문 결론과 같은 방향입니다. 
# 보고서에는 이 과정, 특히 한 페어가 지배한 3개 중 2개가 논문 표 4의 대체 페어라는 점을 기록
#   (출처: 표 4 - USDC / Polygon 은 분류 L "DL level inconsistent with external benchmarks",
#    TUSD / Ethereum 은 분류 C "DL series contains negative and erratic values". USDT / BSC 는 표 4에 없음) 
# 이는 우리 데이터가 논문과 같은 출처라는 또 하나의 방증이기도 합니다.

#%% [셀 4] ③ 2원 분산 분해 (표 7) =========================================
# 출처: 부록 C.4 표 7 주석 (계산 방법이 명시되어 있음)
#   "For each trading day t in the working sample ..., we within-day demean the winsorized
#    growth-rate panel g_i,t and fit ĝ_i,t = α_s(i),t + β_c(i),t + u_i,t by OLS with stablecoin
#    and chain dummies. Sums of squares are accumulated across days and normalised by total
#    within-day variance."
# 목표값 (표 7): 코인 쪽 6.3%, 체인 쪽 61.7%, 공분산 −3.6%, 페어 고유 35.5%
#
# 계산 순서 (표 7 주석 그대로)
#   1) 분석일마다 그날 활성 페어의 윈저화된 성장률을 모음
#   2) 그날 평균을 뺌 (within-day demean)
#   3) 코인 더미 + 체인 더미로 OLS -> α(코인 효과), β(체인 효과), u(잔차)
#   4) 제곱합을 날짜에 걸쳐 누적 -> 전체 제곱합으로 나눔
#
# α 와 β 를 따로 떼어내는 부분:
#   코인 더미와 체인 더미를 함께 넣으면 α 에 상수 k 를 더하고 β 에서 k 를 빼도 적합값이 같아서
#   α·β 가 하나로 정해지지 않음. 표 7은 Var(α), Var(β) 즉 "분산"을 보고하므로 상수 k 는 분산에
#   영향이 없음 -> α 와 β 를 그날 단위들 위에서 평균 0 으로 맞춘 뒤 제곱합을 구함
#   (이렇게 하면 표 7 주석대로 네 줄의 합이 100% 가 됨: Var(y) = Var(α)+Var(β)+2Cov(α,β)+Var(u))
# [논문에 명시 없음] 어떤 날 코인·체인 연결 구조가 끊겨 있으면(공유 체인이 없는 코인 등)
#   α·β 가 상수 이상으로 정해지지 않을 수 있음 -> 최소 노름 해(np.linalg.lstsq)를 사용
#   이런 날이 몇 개인지 아래에서 출력 (0 이면 영향 없음)
제곱합 = {"코인": 0.0, "체인": 0.0, "공분산": 0.0, "잔차": 0.0, "전체": 0.0}
불확정일 = 0
관측치 = 0
for d in 분석일:
    열 = 활성.columns[활성.loc[d].values]
    y = gw.loc[d, 열].values.astype(float)
    ok = np.isfinite(y)
    열, y = 열[ok], y[ok]
    y = y - y.mean()                                      # within-day demean (표 7 주석)
    Ds = pd.get_dummies(페어코인[열]).values.astype(float)   # 코인 더미 (표 7 주석 "stablecoin dummies")
    Dc = pd.get_dummies(페어체인[열]).values.astype(float)   # 체인 더미 (표 7 주석 "chain dummies")
    X = np.hstack([Ds, Dc])
    b, _, 계수, _ = np.linalg.lstsq(X, y, rcond=None)
    if 계수 < X.shape[1] - 1:      # 정상적으로는 상수 1개만큼만 부족
        불확정일 += 1
    a = Ds @ b[:Ds.shape[1]]
    c = Dc @ b[Ds.shape[1]:]
    u = y - a - c
    a, c = a - a.mean(), c - c.mean()
    제곱합["코인"] += a @ a
    제곱합["체인"] += c @ c
    제곱합["공분산"] += 2 * (a @ c)
    제곱합["잔차"] += u @ u
    제곱합["전체"] += y @ y
    관측치 += len(y)

분산분해 = pd.DataFrame({
    "구성요소": ["코인 쪽 Var(α)", "체인 쪽 Var(β)", "공분산 2Cov(α,β)", "페어 고유 Var(u)"],
    "본 연구(%)": [round(제곱합[k] / 제곱합["전체"] * 100, 1) for k in ["코인", "체인", "공분산", "잔차"]],
    "논문 표 7(%)": [6.3, 61.7, -3.6, 35.5],
})
분산분해.to_csv(데이터폴더 / "배제조건_분산분해.csv", index=False, encoding=CSV인코딩)
print(분산분해.to_string(index=False))
print(f"\n합계: {분산분해['본 연구(%)'].sum():.1f}%  (표 7 주석: 네 줄의 합 100%)")
print(f"관측치: {관측치:,} (표 7: 166,019) / 날짜: {len(분석일)} (표 7: 1,913)")
print(f"α·β 연결 구조가 끊긴 날: {불확정일}일  (0 이면 영향 없음)")


# [저자 답신 반영 전 결과] 거의 똑같이 재현됨 (6.5 / 61.3 / −3.6 / 35.8)
# [저자 답신 반영 후 결과] 변화 없음 (6.5 / 61.3 / −3.6 / 35.8)


#%% [셀 5] ④-가 교란변수 수집 (무료 자료만) ===============================
# 출처: 2.3절 + 표 6 - 12개 교란변수
#   VIX, Bloomberg US FCI, Goldman FCI, log S&P 500, log gold, log Brent, log broad dollar NEER,
#   log BGCI, log Bitcoin, log Ether, MMF flow, ETF flow
# 출처: 사용자 지시 - 유료 자료는 빼고 무료로 구할 수 있는 것만 (대체 자료는 다음 단계에서 검토)
#
# 논문이 밝힌 출처
#   - 2.1절 "we collect data on the VIX index, the (log) S&P 500, the (log) gold price, the (log)
#     Brent oil price, the (log) US dollar nominal effective exchange rate (NEER) ... from FRED
#     and Bloomberg"
#   - 2.1절 "We source daily prices for Bitcoin and Ether ... from Yahoo Finance"
#   - 2.1절 MMF 유입: iMoneyNet, ETF 유입: ETF.com  -> 유료
#   - 2.3절 Bloomberg US FCI, Goldman FCI, Bloomberg Galaxy Crypto Index -> Bloomberg (유료)
#
# 이번에 제외하는 6개 (유료 또는 논문이 밝힌 무료 출처에 없음)
#   Bloomberg US FCI, Goldman FCI, BGCI, MMF 유입, ETF 유입  : 유료
#   금 가격 : 논문 출처는 "FRED and Bloomberg"인데, FRED의 일별 금 가격(LBMA) 시리즈는
#             중단되어 2021~2026 자료가 없음 -> 대체 자료 검토 때 다룸
#
# [논문에 명시 없음] FRED 시리즈 코드. 논문은 "FRED and Bloomberg"라고만 씀.
#   아래 코드를 쓰고, [셀 6]에서 논문 표 5의 요약 통계(T, 평균, 표준편차, 최소, 최대)와 비교해 검증
FRED_교란변수 = {
    "VIX": "VIXCLS",          # CBOE Volatility Index: VIX (일별 종가)
    "S&P 500": "SP500",       # S&P 500 (일별 종가)
    "Brent": "DCOILBRENTEU",  # Crude Oil Prices: Brent - Europe (달러/배럴)
    "달러 NEER": "DTWEXBGS",  # Nominal Broad U.S. Dollar Index (표 6 "log broad dollar NEER"와 이름 대응)
}
# Yahoo Finance 종목 코드 (출처: 2.1절 Yahoo Finance)
# [논문에 명시 없음] 종목 코드와 가격 항목 -> Yahoo 의 BTC-USD, ETH-USD 일별 종가(Close) 사용
야후_교란변수 = {"Bitcoin": "BTC-USD", "Ether": "ETH-USD"}

# 5일 변화를 구하려면 표본 시작 전 5 거래일 값이 필요 -> 조금 앞에서부터 받음 (계산상 필요)
받기시작 = pd.Timestamp("2020-12-01")

if "붙여넣기" in FRED_API_KEY:
    raise ValueError("[셀 0]: 프로젝트 폴더에 FRED_키.txt 를 만들고 FRED API 키를 한 줄로 넣으세요 (또는 환경변수 FRED_API_KEY).")

원수준 = {}
for 이름, 코드 in FRED_교란변수.items():
    자료 = requests.get(FRED주소, timeout=120, params={
        "series_id": 코드, "api_key": FRED_API_KEY, "file_type": "json",
        "observation_start": 받기시작.strftime("%Y-%m-%d"),
        "observation_end": 표본끝.strftime("%Y-%m-%d"),
    }).json()
    # FRED는 휴장일 값을 "." 으로 줌 -> 빈 값 (FRED 자료 형식)
    원수준[이름] = pd.Series({pd.to_datetime(o["date"]): (np.nan if o["value"] == "." else float(o["value"]))
                          for o in 자료["observations"]})
    print(f"  {이름} ({코드}): {원수준[이름].notna().sum():,}개")

import yfinance as yf
for 이름, 코드 in 야후_교란변수.items():
    h = yf.Ticker(코드).history(start=받기시작.strftime("%Y-%m-%d"),
                               end=(표본끝 + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                               auto_adjust=False)
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()   # 시간대 표시 제거 (날짜만 사용)
    원수준[이름] = s
    print(f"  {이름} ({코드}): {s.notna().sum():,}개")

원수준 = pd.DataFrame(원수준)
원수준.index.name = "date"
원수준.to_csv(데이터폴더 / "교란변수_원자료_일별.csv", encoding=CSV인코딩)
print("저장: 교란변수_원자료_일별.csv")


#%% [셀 6] ④-나 평일 달력에 맞추기 + 표 5와 비교 ===========================
# 교란변수를 평일 달력에 맞추고, 고른 FRED 시리즈가 논문과 같은지 표 5로 검증하는 셀(GIV와 5일 유입이 모두 평일(월~금) 기준이라)
# 평일 달력 (출처: 퀀트랩_BIS_2.py [셀 0], 각주 13 "we merge the dataset at the trading-day frequency")
#
# 휴장일 처리
#   논문 표 5에서 VIX, log S&P 500, log Brent, log broad dollar NEER 의 T 가 모두 1,367
#   = 평일 전체 -> 논문은 휴장일(공휴일)에도 값이 있게 만들었다는 뜻 (출처: 표 5)
# 근거: 논문 표 5에서 VIX, log S&P 500, log Brent, log 달러 NEER의 T가 모두 1,367, 즉 평일 전체. 논문은 휴장일에도 값을 채웠다는 뜻
#   [논문에 명시 없음] 어떤 방법으로 채웠는지 -> 두 가지 모두 계산 (출처: 사용자 지시)
#     "앞값 채움": 직전 거래일 값을 씀 (T = 1,367 이 되는 방식)
#     "안 채움"  : 원자료 그대로, 휴장일은 빈 값
평일_넓게 = pd.bdate_range(받기시작, 표본끝)
수준 = {"앞값 채움": 원수준.reindex(원수준.index.union(평일_넓게)).sort_index().ffill().reindex(평일_넓게),
        "안 채움": 원수준.reindex(평일_넓게)}

# 표 5와 비교 (출처: 표 5 - VIX 는 수준, 나머지는 로그 수준)
표5 = {"VIX": (1367, 19.3, 5.25, 11.9, 52.3), "S&P 500": (1367, 8.49, 0.19, 8.18, 8.85),
       "Brent": (1367, 4.37, 0.16, 3.93, 4.85), "달러 NEER": (1367, 4.79, 0.04, 4.71, 4.87)}
비교행 = []
for 이름, 논문값 in 표5.items():
    s = 수준["앞값 채움"][이름].loc[평일]
    s = s if 이름 == "VIX" else np.log(s)
    비교행.append([이름, s.notna().sum(), s.mean(), s.std(), s.min(), s.max(), *논문값])
비교 = pd.DataFrame(비교행, columns=["변수", "T", "평균", "표준편차", "최소", "최대",
                                "표5_T", "표5_평균", "표5_표준편차", "표5_최소", "표5_최대"])
print("[표 5와 비교] 앞값 채움 기준 (VIX 는 수준, 나머지는 로그)")
print(비교.round(2).to_string(index=False))

# 이 결과가 말해주는 것
# 휴장일 앞값 채움 방식이 논문과 맞을 가능성이 높습니다.

#%% [셀 7] ④-다 GIV 와 교란변수의 상관 (표 6) ===============================
# 배제 조건 ④의 핵심 계산으로, GIV와 교란변수 6개의 상관을 구해 논문 표 6과 비교하는 셀
# GIV가 "금리와 무관한 이유로 생긴 유입"이라면, 위험선호(VIX, S&P 500), 원자재(브렌트유), 달러, 암호자산 시장(비트코인, 이더리움)의 움직임과 상관이 거의 없어야 합니다. 
# 논문은 12개 변수 모두 상관의 절댓값이 0.10 이하라는 것을 근거로 듭니다(2.3절, 표 6).
# 출처: 부록 C.3 표 6 주석 (계산 방법이 명시되어 있음)
#   - 빈도 맞춤: "the 5-day-sum granular IV GIV(5d)_t with the 5-day change of the confounder;
#     ... the daily granular IV GIV_t with the 1-day change of the confounder"
#   - 변화의 종류: "Changes are log-differences for price-type series (S&P 500, gold, Brent,
#     NEER, BGCI, Bitcoin, Ether, VIX) and arithmetic differences for index- and level-type
#     series (Bloomberg US FCI, Goldman FCI, MMF AUM)"
#     -> 이번 6개 변수는 모두 로그 차분 (VIX 포함)
#   - 유의성: "Newey–West HAC standard errors with one lag, with *, **, *** denoting
#     significance at the 10, 5 and 1% level"
#   - 표본: "January 2021 to March 2026" -> 평일 2021-01-04 ~ 2026-03-31
# 5일 변화 = 5 거래일 전 대비 (출처: 각주 13 "5-day flow measures therefore refer to five trading days")
#
# [논문에 명시 없음] "HAC-robust correlation"의 유의성 계산 방법
#   상관계수 ρ 자체는 표준 피어슨 상관. 유의성은 두 변수를 표준화한 뒤 한쪽을 다른 쪽에
#   회귀하면 기울기 = ρ 가 되므로, 그 기울기의 Newey-West(시차 1) 검정으로 계산.
#   어느 쪽을 종속변수로 두는지는 논문에 없음 -> 두 방향 모두 계산해 표시
GIV = pd.read_csv(데이터폴더 / "GIV_선택조합_평일.csv", parse_dates=["date"], index_col="date",
                  encoding=CSV인코딩)


def 별표(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def HAC상관(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    za = (d.iloc[:, 0] - d.iloc[:, 0].mean()) / d.iloc[:, 0].std(ddof=0)
    zb = (d.iloc[:, 1] - d.iloc[:, 1].mean()) / d.iloc[:, 1].std(ddof=0)
    ρ = float((za * zb).mean())
    p = []
    for y_, x_ in [(zb, za), (za, zb)]:       # (교란변수 ~ GIV), (GIV ~ 교란변수)
        r = sm.OLS(y_, sm.add_constant(x_)).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        p.append(float(r.pvalues.iloc[1]))
    return ρ, p[0], p[1], len(d)


# 논문 표 6 값 (출처: 표 6) - (5일, 일별)
표6 = {"VIX": ("-0.055*", "+0.006"), "S&P 500": ("+0.017", "-0.021"), "Brent": ("-0.033", "-0.011"),
       "달러 NEER": ("-0.044", "-0.038"), "Bitcoin": ("+0.006", "+0.003"), "Ether": ("-0.087**", "-0.024")}

상관행 = []
for 채움, 표 in 수준.items():
    로그 = np.log(표)                                     # 가격형 -> 로그 (표 6 주석)
    for 이름 in 표6:
        변화5 = (로그[이름] - 로그[이름].shift(5)).reindex(평일)
        변화1 = (로그[이름] - 로그[이름].shift(1)).reindex(평일)
        ρ5, p5a, p5b, n5 = HAC상관(GIV["GIV_5일합"], 변화5)
        ρ1, p1a, p1b, n1 = HAC상관(GIV["GIV_일별"], 변화1)
        상관행.append({"휴장일처리": 채움, "변수": 이름,
                    "5일_ρ": round(ρ5, 3), "5일_별(교란~GIV)": 별표(p5a), "5일_별(GIV~교란)": 별표(p5b), "5일_T": n5,
                    "5일_표6": 표6[이름][0],
                    "일별_ρ": round(ρ1, 3), "일별_별(교란~GIV)": 별표(p1a), "일별_별(GIV~교란)": 별표(p1b), "일별_T": n1,
                    "일별_표6": 표6[이름][1]})

상관 = pd.DataFrame(상관행)
상관.to_csv(데이터폴더 / "배제조건_교란변수상관.csv", index=False, encoding=CSV인코딩)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 20)
for 채움 in 수준:
    t = 상관[상관["휴장일처리"] == 채움].drop(columns="휴장일처리")
    print(f"\n[{채움}]")
    print(t.to_string(index=False))
    print(f"  최대 |ρ|: 5일 {t['5일_ρ'].abs().max():.3f}, 일별 {t['일별_ρ'].abs().max():.3f}"
          f"  (표 6 - 12개 변수 기준: 0.097, 0.056)")
    
# 결론 [저자 답신 반영 후 결과]
# 6개 변수 모두 |ρ| 0.065 이하로, 논문 기준(0.10 이하)을 만족합니다.
# 부호가 12개 모두 논문과 같습니다.
# 크기도 대부분 0.02 안에서 맞습니다 (S&P 500 5일 0.018 대 0.017).
# 달러 NEER 일별 −0.040 은 반영 후 10% 수준 유의(*)로 표시됨 (논문 −0.038, 유의 표시 없음) - p값이 10% 경계 근처

#%% [셀 8] 결과 요약 =====================================================
# 논문의 결론과 같은 방향인지 한눈에 보기 (판단 기준은 모두 논문 문장)
print("① 구성상 직교 (2.3절)")
print(f"   가중치 합 최대 {가중치합.abs().max():.1e}, 공통 충격 후 GIV 변화 최대 {차이:.1e}")
print("② 주성분 분석 (2.3절: 첫 주성분 약 16%)")
print(f"   8가지 설정 범위: {주성분결과['첫주성분_설명비율(%)'].min()}% ~ {주성분결과['첫주성분_설명비율(%)'].max()}%")
print("③ 분산 분해 (표 7: 코인 6.3 / 체인 61.7 / 공분산 −3.6 / 페어 고유 35.5)")
print("   " + " / ".join(f"{r['구성요소'].split()[0]} {r['본 연구(%)']}" for _, r in 분산분해.iterrows()))
print("④ 교란변수 상관 (2.3절: 12개 변수 모두 |ρ| ≤ 0.10) - 이번에는 무료 6개 변수만")
for 채움 in 수준:
    t = 상관[상관["휴장일처리"] == 채움]
    print(f"   [{채움}] 최대 |ρ| 5일 {t['5일_ρ'].abs().max():.3f}, 일별 {t['일별_ρ'].abs().max():.3f}")
    
#%% [셀 9] 배제 조건 ①~④ 최종 정리 =========================================
# [셀 2]~[셀 7]의 결과를 한 표로 모읍니다. 모든 숫자는 앞 셀에서 계산된 값을 그대로 씀.
# ②는 설정별로 "첫 주성분이 한 페어에 쏠려 있는지"를 함께 계산해 유효·무효를 나눕니다.

# ---------------------------------------------------------------
# 논문 부록 C.1 표 4 - 저자들이 기준 패널에서 Token Terminal 로 대체한 12개 페어
# 출처: 표 4 "Stablecoin-chain substitutions from Token Terminal in the baseline panel"
#   분류 L = DL 수준값이 외부 기준과 불일치, C = DL 자료 오류, E = TT 가 더 이른 날짜까지 있음
# ---------------------------------------------------------------
표4_페어 = {
    "USDC / Arbitrum": "L", "USDC / Polygon": "L", "USDC / Solana": "E", "USDC / Stellar": "E",
    "USDT / Solana": "E", "TUSD / Ethereum": "C", "TUSD / Tron": "E", "USDC / Linea": "E",
    "USDC / Tron": "E", "USDC / Sui": "E", "USDC / World Chain": "E", "USDC / Cardano": "C",
}

# ---------------------------------------------------------------
# ② 설정별 첫 주성분에서 가장 비중이 큰 페어
#   첫 주성분 벡터의 각 원소를 제곱하면 페어별 비중이 되고, 합이 1 (고유벡터의 길이 = 1)
#   [셀 3]의 첫주성분비율 함수와 같은 전처리를 그대로 씀
# [논문에 명시 없음] "한 페어가 지배한다"의 기준 -> 과반(비중 ≥ 0.5)으로 표시
#   이 기준은 해석용 표시일 뿐, 어떤 추정값 계산에도 쓰이지 않음. 비중 숫자는 그대로 함께 출력
# ---------------------------------------------------------------
def 최대비중페어(행렬, 방식_결측, 방식_행렬):
    X = 행렬.copy()
    if 방식_결측 == "균형패널":
        X = X.loc[:, X.notna().all(axis=0)]
    else:
        X = X.loc[:, X.notna().sum(axis=0) >= 2]
    X = X - X.mean(axis=0)
    if 방식_행렬 == "상관":
        sd = X.std(axis=0)
        X = X.loc[:, sd > 0] / sd[sd > 0]
    X = X.fillna(0.0)
    _, 벡터 = np.linalg.eigh(np.cov(X.values, rowvar=False))
    비중 = pd.Series(벡터[:, -1] ** 2, index=X.columns)
    return 비중.idxmax(), float(비중.max())


진단 = []
for _, r in 주성분결과.iterrows():
    원자료 = (gw if r["윈저화"] == "적용" else g).where(활성).loc[분석일]
    페어, 비중 = 최대비중페어(원자료, r["결측처리"], r["행렬"])
    진단.append({"최대비중페어": 페어, "그 페어 비중": round(비중, 3),
               "표4 해당(분류)": 표4_페어.get(페어, "-"),
               "판정": "무효(한 페어가 과반)" if 비중 >= 0.5 else "유효"})
주성분진단 = pd.concat([주성분결과.reset_index(drop=True), pd.DataFrame(진단)], axis=1)
주성분진단.to_csv(데이터폴더 / "배제조건_주성분분석_진단.csv", index=False, encoding=CSV인코딩)

유효 = 주성분진단[주성분진단["판정"] == "유효"]
무효 = 주성분진단[주성분진단["판정"] != "유효"]

# ---------------------------------------------------------------
# ③ 표 7과의 최대 차이
# ---------------------------------------------------------------
최대차이_표7 = (분산분해["본 연구(%)"] - 분산분해["논문 표 7(%)"]).abs().max()

# ---------------------------------------------------------------
# ④ 부호 일치 개수 (앞값 채움 기준) - 표 6 문자열의 첫 글자(+/−)와 우리 ρ 의 부호 비교
# ---------------------------------------------------------------
t4 = 상관[상관["휴장일처리"] == "앞값 채움"]
부호일치 = 0
for _, r in t4.iterrows():
    for 열ρ, 열표 in [("5일_ρ", "5일_표6"), ("일별_ρ", "일별_표6")]:
        논문부호 = -1 if r[열표].startswith("-") else 1
        부호일치 += int(np.sign(r[열ρ]) == 논문부호)
부호전체 = len(t4) * 2
최대ρ = max(상관["5일_ρ"].abs().max(), 상관["일별_ρ"].abs().max())

# ---------------------------------------------------------------
# 최종 표
# ---------------------------------------------------------------
최종 = pd.DataFrame([
    ["① 구성상 직교",
     f"공통 충격 후 GIV 변화 최대 {차이:.1e} (참고: 가중치 합 최대 {가중치합.abs().max():.2f} - 저자 N 정의에서 생기는 값, 판정에 안 씀)",
     "성립" if 차이 < 1e-10 else "확인 필요"],
    ["② 주성분 분석",
     f"유효 {len(유효)}개 설정: {유효['첫주성분_설명비율(%)'].min()}~{유효['첫주성분_설명비율(%)'].max()}% / "
     f"무효 {len(무효)}개: 한 페어가 지배 ({', '.join(sorted(set(무효['최대비중페어'])))})",
     "16% 숫자는 미재현, 결론 비교는 유효 설정으로"],
    ["③ 분산 분해",
     " / ".join(f"{v}" for v in 분산분해["본 연구(%)"]) + "  (코인/체인/공분산/페어 고유)",
     f"표 7과 최대 차이 {최대차이_표7:.1f}%p"],
    ["④ 교란변수 상관",
     f"무료 {len(t4)}개 변수 |ρ| 최대 {최대ρ:.3f}, 부호 {부호일치}/{부호전체} 일치",
     "논문 기준(|ρ| ≤ 0.10) " + ("충족" if 최대ρ <= 0.10 else "미충족") + ". 유료 5개와 금은 미포함"],
], columns=["근거", "결과", "논문과 비교"])
최종.to_csv(데이터폴더 / "배제조건_최종정리.csv", index=False, encoding=CSV인코딩)

pd.set_option("display.max_colwidth", 200)
print("[② 설정별 진단]")
print(주성분진단.to_string(index=False))
print("\n[배제 조건 ①~④ 최종 정리]")
for _, r in 최종.iterrows():
    print(f"{r['근거']}\n   결과: {r['결과']}\n   비교: {r['논문과 비교']}")
print("\n저장: 배제조건_주성분분석_진단.csv, 배제조건_최종정리.csv")

#%% [셀 10] ② 보조 점검: 논문 표 4의 12개 페어를 뺀 주성분 분석 =================
# 목적: 논문이 기준 패널에서 손본 페어를 빼면, 첫 주성분 비율이 2.3절의 "약 16%"에
#       가까워지는지 확인 (출처: 사용자 지시 - 추가 점검)
# 출처: 논문 부록 C.1 표 4 "Stablecoin-chain substitutions from Token Terminal in the
#       baseline panel" - 저자들이 DefiLlama 값을 Token Terminal 값으로 바꾼 12개 페어
# [논문에 명시 없음] 기준 패널에서 이 12개는 "삭제"가 아니라 "대체"됨.
#   Token Terminal 값이 없으므로 빼는 것으로 근사함 -> 참고용. [셀 3] 결과를 대체하지 않음
# 필요한 셀: [셀 0], [셀 1] (다운로드 없음). 화면 출력만, 파일 저장 없음
# 계산 방법: [셀 3] 첫주성분비율 + [셀 9] 최대비중페어와 똑같은 전처리

표4_12 = ["USDC / Arbitrum", "USDC / Polygon", "USDC / Solana", "USDC / Stellar",
          "USDT / Solana", "TUSD / Ethereum", "TUSD / Tron", "USDC / Linea",
          "USDC / Tron", "USDC / Sui", "USDC / World Chain", "USDC / Cardano"]

# 표 4 이름이 우리 패널의 페어 이름과 맞는지 확인 (안 맞는 이름이 있으면 그 페어는 안 빠짐)
없는페어 = [p for p in 표4_12 if p not in gw.columns]
print("우리 패널에 없는 표 4 페어:", 없는페어 if 없는페어 else "없음 (12개 모두 찾음)")
남길열 = [p for p in gw.columns if p not in 표4_12]


def 첫주성분_진단(행렬, 방식_결측, 방식_행렬):
    X = 행렬.copy()
    if 방식_결측 == "균형패널":
        X = X.loc[:, X.notna().all(axis=0)]
    else:
        X = X.loc[:, X.notna().sum(axis=0) >= 2]      # 평균·표준편차에 최소 2일 필요 ([셀 3]과 같음)
    X = X - X.mean(axis=0)
    if 방식_행렬 == "상관":
        sd = X.std(axis=0)
        X = X.loc[:, sd > 0] / sd[sd > 0]
    X = X.fillna(0.0)
    고유값, 벡터 = np.linalg.eigh(np.cov(X.values, rowvar=False))
    비중 = pd.Series(벡터[:, -1] ** 2, index=X.columns)  # 첫 주성분에서 페어별 비중 (합 = 1)
    return 고유값[-1] / 고유값.sum() * 100, X.shape[1], 비중.idxmax(), 비중.max()


행 = []
for 윈저 in ["적용", "미적용"]:
    for 표4처리 in ["포함", "제외"]:          # "포함" = [셀 3]과 같은 계산 (제대로 돌았는지 확인용)
        원자료 = (gw if 윈저 == "적용" else g).where(활성).loc[분석일]
        if 표4처리 == "제외":
            원자료 = 원자료[남길열]
        for 결측 in ["0 채움", "균형패널"]:
            for 행렬종류 in ["공분산", "상관"]:
                비율, n, 페어, 비중 = 첫주성분_진단(원자료, 결측, 행렬종류)
                행.append({"윈저화": 윈저, "결측처리": 결측, "행렬": 행렬종류, "표4페어": 표4처리,
                          "사용페어수": n, "첫주성분(%)": round(비율, 1),
                          "최대비중페어": 페어, "그 비중": round(비중, 3)})

표4점검 = pd.DataFrame(행).sort_values(["윈저화", "결측처리", "행렬", "표4페어"], ascending=[False, True, True, False])
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\n논문 2.3절: 첫 번째 주성분 약 16%")
print(표4점검.to_string(index=False))
