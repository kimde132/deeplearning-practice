# =====================================================================
#  실습 2 — 진동으로 압연 모터 전류를 맞혀 보기 (02 다변수 선형회귀)
# =====================================================================
#  실행: python 실습2_문제.py      (이 파일이 있는 폴더에서)
#  필요한 것: numpy, pandas  /  데이터는 옆의 데이터/ 폴더에 들어 있습니다.
#
#  [상황]
#    P제철 열간압연기(SPM01)에 진동센서 두 개(TOP·BOT)와 모터 전류계가 붙어 있습니다.
#    그런데 전류계가 자주 고장 납니다. 진동만 있을 때 전류를 추정할 수 있을까요?
#    맞힐 대상(정답) = CUR-MTR_RMS  (모터 전류의 실효값)
#
#  [쓰는 도구]  02 에서 배운 것 전부. 새 라이브러리 없습니다.
#    다변수 X / train·test 분할 / 열별 표준화(학습용 통계로만) / 경사하강 / R2
#    ※ 02_선형회귀_다변수_train_test.py 를 옆에 띄워 놓고 베껴 쓰세요. 그게 정상입니다.
#
#  [푸는 법]  TODO 를 위에서부터 하나씩 채우고, 그때그때 실행해서 숫자를 확인하세요.
#             한 번에 다 짜고 실행하면 어디서 틀렸는지 못 찾습니다.
# =====================================================================

# =====================================================================
#  [정답출력 참고]  각 항목의 [나와야 하는 출력] 은 이렇게 만든 값입니다
# =====================================================================
#  · 수업용코드 02 의 함수(예측/손실/기울기_밟아보기/학습/MSE/R2, h=0.0001)를 한 글자도
#    안 바꾸고 이 데이터에 돌린 값입니다. (numpy 2.5.3 / pandas 3.0.5 / 이 폴더의 venv)
#  · 숫자만 같으면 맞은 겁니다. 출력 모양(줄 배치, 단위 표기)은 달라도 됩니다.
#  · 실습4 머리말에 적힌 실습2 정답값(0.7982 / 0.6080, 가중치, 0.9984 / 0.9982, 0.7451 / 0.7754)과
#    전부 일치하므로 출제자의 정답지와 같은 숫자입니다.
#  · [G1] 은 문제에 epochs 와 표준화 방식이 정해져 있지 않습니다. 02 §5-1 방식
#    (Z_train[:n] 그대로 쓰고, 5/10/30/100 대는 epochs=2000, 399 대는 500) 기준으로 적었습니다.
#    다르게 하면 5 대 줄이 달라질 수 있습니다. 그 경우의 값도 블록 안에 괄호로 적어 뒀습니다.
#  · [G1] 의 print("\nF2") 는 원본 파일의 오타로 보입니다. G1 의 출력입니다. 손대지 않았습니다.
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")
d = pd.read_csv(os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig")
# ↑ encoding="utf-8-sig" 빠뜨리면 첫 열 이름이 깨져서 KeyError 납니다.


# =====================================================================
# A. 데이터부터 본다  (모델 얘기는 아직 이르다)
# =====================================================================
# [A1] 표의 모양과 열 이름, 결측 개수를 찍으세요.
#      힌트: d.shape / list(d.columns) / d.isna().sum().sum()
#
# [나와야 하는 출력]  (숫자만 맞으면 됩니다. 출력 모양까지 똑같을 필요는 없습니다)
#     모양: (570, 19)
#     열: ['MEAS_DT', 'BURST_ID', 'ROWS', 'DUR_S',
#          'VIB-TOP_RMS', 'VIB-TOP_KUR', 'VIB-TOP_CRF', 'VIB-TOP_PTP', 'VIB-TOP_STD',
#          'VIB-BOT_RMS', 'VIB-BOT_KUR', 'VIB-BOT_CRF', 'VIB-BOT_PTP', 'VIB-BOT_STD',
#          'CUR-MTR_RMS', 'CUR-MTR_KUR', 'CUR-MTR_CRF', 'CUR-MTR_PTP', 'CUR-MTR_STD']
#     결측 개수: 0      (열별로 세도 전부 0)
# TODO
print("\nA1")

print(d.shape)
print(list(d.columns))
print(d.isnull().sum().sum())

# [A2] 정답으로 쓸 CUR-MTR_RMS 의 요약통계를 보세요. (describe)
#      → 이 값이 대략 몇에서 몇 사이인지 말할 수 있어야 합니다.
#        나중에 "MSE 500" 이 큰 건지 작은 건지 판단하는 기준이 됩니다.
#
# [나와야 하는 출력]
#     count    570.000000
#     mean     111.292407
#     std       38.382288
#     min        6.325006
#     25%       82.244598
#     50%       93.381820
#     75%      145.054399
#     max      193.420110
#     (대략 6 ~ 193 A, 가운데 절반은 82 ~ 145 A. 표준편차 38 → 분산 ≈ 1473.
#      "MSE 500" 은 평균 오차가 √500 ≈ 22 A 쯤이라는 뜻입니다)
# TODO
print("\nA2")

print(d["CUR-MTR_RMS"].describe())

# [A3] 숫자 열들의 상관계수 중, CUR-MTR_RMS 와의 상관만 크기순으로 보세요.
#      힌트: d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values()
#
#      ★ 보고 나서 답하세요 (주석으로 적어 두기) ★
#        (1) 상관이 0.98, 0.99 로 말도 안 되게 높은 열이 몇 개 보입니다. 이름이 뭔가요?
#        (2) 그 열들을 입력으로 쓰면 안 되는 이유가 있습니다. 뭘까요?
#            힌트: 열 이름의 앞부분을 보세요. CUR-MTR-... 로 시작하죠.
#                 전류계가 고장 나서 전류를 추정하려는 건데, 그 입력은 어디서 옵니까?
#      내 답: (1)
#             (2)
#
# [나와야 하는 출력]  (작은 것부터 큰 순)
#     VIB-BOT_KUR   -0.469390
#     VIB-BOT_CRF   -0.448028
#     CUR-MTR_KUR   -0.249195
#     CUR-MTR_CRF   -0.089699
#     VIB-TOP_STD    0.082713
#     VIB-TOP_RMS    0.093488
#     VIB-TOP_PTP    0.142816
#     ROWS           0.229055
#     DUR_S          0.229055
#     VIB-TOP_KUR    0.230738
#     VIB-TOP_CRF    0.327160
#     VIB-BOT_PTP    0.684858
#     VIB-BOT_STD    0.712887
#     VIB-BOT_RMS    0.738609
#     CUR-MTR_PTP    0.982105
#     CUR-MTR_STD    0.998795
#     CUR-MTR_RMS    1.000000
# TODO
print("\nA3")

print(d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values())

# =====================================================================
# B. 입력 고르고 train / test 나누기
# =====================================================================
# [B1] 입력(특징) 4개를 아래 이름 그대로 쓰세요. 정답은 CUR-MTR_RMS.
특징이름 = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS"]
# X = ...   (d[특징이름].values.astype(float))
# y = ...   (d["CUR-MTR_RMS"].values.astype(float))
# X.shape, y.shape 를 찍어서 (570, 4) 와 (570,) 인지 확인하세요.
#
# [나와야 하는 출력]
#     X.shape: (570, 4) / y.shape: (570,)
# TODO
print("\nB1")

X = d[특징이름].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)

print(f"x.shape {X.shape} / y.shape : {y.shape}")

# [B2] 7:3 으로 나누세요. 반드시 '섞은 다음에' 자릅니다.
#      RandomState(42) 를 쓰면 정답지와 숫자가 똑같이 나옵니다.
#      힌트: 순서 = np.random.RandomState(42).permutation(len(X))
#            n_train = int(len(X) * 0.7)
#      학습용 몇 대 / 시험용 몇 대인지 찍으세요.
#
# [나와야 하는 출력]
#     학습용 399 대 / 시험용 171 대
#     (확인용: 순서[:10] = [507  70 131 400 541 362 188  29  81 250])
# TODO
print("\nB2")

order = np.random.RandomState(42).permutation(len(X))
n_train = int(len(X) * 0.7)
print(f"학습용 : {n_train}대 / 시험용 : {len(X) - n_train}대")
print(f"확인용 : order[:10] = {order[:10]}")

# [B3] 열별 표준화. ★ mu 와 sd 는 학습용에서만 구합니다 ★
#      시험용도 학습용의 mu, sd 로 변환하세요.
#      확인: 표준화 후 학습용 열별 평균은 0, 퍼짐은 1.
#            시험용 평균은 0 이 아닙니다. 그게 맞습니다 (이유를 말할 수 있어야 합니다).
#
# [나와야 하는 출력]
#     mu = [ 0.1092  0.375  -0.7263  0.0796]
#     sd = [ 0.063   0.2084  0.8804  0.0784]
#     표준화 후 학습용 열별 평균: [0. 0. 0. 0.]
#     표준화 후 학습용 열별 퍼짐: [1. 1. 1. 1.]
#     표준화 후 시험용 열별 평균: [-0.09  -0.089  0.128 -0.093]   ← 0 이 아님
# TODO
print("\nB3")

tr = order[:n_train]
te = order[n_train:]

X_train = X[tr]
X_test = X[te]

y_train = y[tr]
y_test = y[te]

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

print(f"mu = {mu.round(4)}")
print(f"sd = {sd.round(4)}")

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

print(f"표준화 후 학습용 열별 평균 : {Z_train.mean(axis=0).round(4)}")
print(f"표준화 후 학습용 열별 퍼짐 : {Z_train.std(axis=0).round(4)}")
print(f"표준화 후 시험용 열별 평균 : {Z_test.mean(axis=0).round(3)}")

# =====================================================================
# C. 학습 — 02 의 함수를 그대로 가져다 쓰세요
# =====================================================================
# [C1] 예측 / 손실 / 기울기_밟아보기 / 학습 / R2 / MSE 를 02 에서 복사해 오세요.
#      한 글자도 안 바꿔도 됩니다. 그게 이 실습의 포인트입니다.
#      (데이터가 바뀌어도 걷는 방법은 안 바뀝니다)
#
# [나와야 하는 출력]
#     출력 없음. 예측 / 손실 / 기울기_밟아보기 / 학습 / MSE / R2 여섯 개와 h = 0.0001 이 정의되면 됩니다.
# TODO
print("\nC1")


def 예측(Z, w, b):
    return Z @ w + b


def 손실(Z, y, w, b):  # 01 과 완전히 같음: (실제 − 예측)² 의 평균 = MSE
    return np.mean((y - 예측(Z, w, b)) ** 2)


h = 0.0001


def 기울기_밟아보기(Z, y, w, b):
    gw = np.zeros(len(w))
    for j in range(len(w)):  # j = 0,1,2,3 : j 번째 손잡이만 움직여 본다
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h  # j 번째만 살짝 키우고
        w_minus[j] -= h  # j 번째만 살짝 줄여서
        gw[j] = (손실(Z, y, w_plus, b) - 손실(Z, y, w_minus, b)) / (
            2 * h
        )  # (오른쪽 손실 − 왼쪽 손실) ÷ 거리
    gb = (손실(Z, y, w, b + h) - 손실(Z, y, w, b - h)) / (2 * h)  # b 도 한 번
    return gw, gb


def 학습(Z, y, lr=0.1, epochs=500):
    w = np.zeros(Z.shape[1])  # Z.shape[1] = 열 수 = 4. 가중치 4개를 0 에서 출발
    b = 0.0
    for _ in range(epochs):  # 500 바퀴 (에폭 500)
        gw, gb = 기울기_밟아보기(Z, y, w, b)
        w = w - lr * gw
        b = b - lr * gb
    return w, b


w, b = 학습(Z_train, y_train)
print("학습 완료 — 표준화 눈금의 가중치")
for 이름, wi in zip(특징이름, w):
    print(
        f"    {이름:6s} w = {wi:+.3f}"
    )  # {:+.3f} = 부호 붙여 소수 3자리 (00 미리보기 ①)
print(f"    절편 b = {b:.3f}")


def MSE(y, yhat):  # 손실과 같은 식. 채점용으로 이름만 따로
    return np.mean((y - yhat) ** 2)


def R2(y, yhat):  # 01 9번의 R². 1 에 가까울수록 좋음, 0 = 평균만 말하는 수준
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# [C2] 학습용으로 학습(lr=0.1, epochs=500)하고, 특징별 가중치와 절편 b 를 찍으세요.
#
# [나와야 하는 출력]
#     VIB-BOT_RMS  w = +24.692
#     VIB-BOT_PTP  w = +14.966
#     VIB-BOT_KUR  w = -5.908
#     VIB-TOP_RMS  w = -18.962
#     절편 b = 112.640
# TODO
print("\nC2")

w, b = 학습(Z_train, y_train)
for idx in range(Z_train.shape[1]):
    print(f"{특징이름[idx]} w = {w[idx].round(3)}")
print(f"절편 b = {b.round(3)}")

# [C3] 학습용 R2 / 시험용 R2 를 나란히 찍으세요. MSE 도 같이.
#
#      ★ 답하세요 ★
#        차이가 얼마입니까? 02 의 경보선(0.05 정상 / 0.10 넘으면 의심)에 비춰 보면
#        이 모델은 건강한가요, 과적합인가요?
#        ai4i 데이터(02)에서는 차이가 0.03 이었습니다. 왜 여기선 다를까요?
#      내 답:
#
# [나와야 하는 출력]
#     학습용(train)  R² 0.7982   MSE 307.993
#     시험용(test)   R² 0.6080   MSE 519.897
#     (차이 0.1902)
# TODO
print("\nC3")

print(
    f"학습용(train) R2 : {round(R2(y_train, 예측(Z_train, w, b)), 4)} MSE : {round(MSE(y_train, 예측(Z_train, w, b)), 4)}"
)
print(
    f"시험용(test) R2 : {round(R2(y_test, 예측(Z_test, w, b)), 4)} MSE : {round(MSE(y_test, 예측(Z_test, w, b)), 4)}"
)
print(
    f"차이 {round(R2(y_train, 예측(Z_train, w, b)) - R2(y_test, 예측(Z_test, w, b)), 4)}"
)

# =====================================================================
# D. 함정 1 — "점수가 너무 좋으면 의심하라"
# =====================================================================
# [D1] 특징에 "CUR-MTR_STD" 를 하나 추가해서(총 5개) 다시 학습하고 채점하세요.
#      B2 의 분할(순서)은 그대로 재사용합니다. 표준화는 다시 해야 합니다(열이 5개니까).
#
#      ★ 답하세요 ★
#        (1) R2 가 몇으로 나왔나요? 학습용·시험용 둘 다 적으세요.
#        (2) 이 모델을 현장에 넣으면 잘 될까요? 이유는?
#        (3) 이걸 부르는 이름이 있습니다 — '누수(leakage)'.
#            이 경우 정확히 무엇이 새어 들어온 겁니까?
#      내 답: (1)
#             (2)
#             (3)
#
# [나와야 하는 출력]
#     VIB-BOT_RMS  w = -4.617
#     VIB-BOT_PTP  w = +3.375
#     VIB-BOT_KUR  w = -0.319
#     VIB-TOP_RMS  w = +0.202
#     CUR-MTR_STD  w = +40.142
#     절편 b = 112.640
#     학습용 R² 0.9984   MSE 2.379
#     시험용 R² 0.9982   MSE 2.441
# TODO
print("\nD1")

특징이름 = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS", "CUR-MTR_STD"]

X = d[특징이름].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)

X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

w, b = 학습(Z_train, y_train)

for idx in range(Z_train.shape[1]):
    print(f"{특징이름[idx]} w = {w[idx].round(3)}")
print(f"절편 b = {b.round(3)}")
print(
    f"학습용(train) R2 : {round(R2(y_train, 예측(Z_train, w, b)), 4)} MSE : {round(MSE(y_train, 예측(Z_train, w, b)), 4)}"
)
print(
    f"시험용(test) R2 : {round(R2(y_test, 예측(Z_test, w, b)), 4)} MSE : {round(MSE(y_test, 예측(Z_test, w, b)), 4)}"
)

# =====================================================================
# E. 함정 2 — 상관 순위와 실제 쓸모는 다르다
# =====================================================================
# [E1] 02 §6-1 처럼 특징을 하나씩 빼고 다시 학습해, 학습용·시험용 R2 를 각각 찍으세요.
#      (4개 특징이니 4줄이 나옵니다. 힌트: Z_train[:, 남길])
#
#      ★ 먼저 예상하고 적으세요. 실행은 그다음에. ★
#        A3 에서 본 상관을 보면 VIB-BOT_RMS 가 0.74 로 1등,
#        VIB-TOP_RMS 는 0.09 로 사실상 무관해 보입니다.
#        그럼 VIB-TOP_RMS 를 빼도 점수가 안 변하겠죠?
#      내 예상:
#
# [나와야 하는 출력]  (전부 쓰면 학습용 0.7982 / 시험용 0.6080)
#     VIB-BOT_RMS 빼면 → 학습용 0.7748 / 시험용 0.6840
#     VIB-BOT_PTP 빼면 → 학습용 0.7899 / 시험용 0.5271
#     VIB-BOT_KUR 빼면 → 학습용 0.7856 / 시험용 0.5576
#     VIB-TOP_RMS 빼면 → 학습용 0.6741 / 시험용 0.3392
# TODO
print("\nE1")

특징이름 = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS"]

X = d[특징이름].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)

X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

for minus in range(4):
    remain = [j for j in range(4) if j != minus]
    w2, b2 = 학습(Z_train[:, remain], y_train)
    r_tr = R2(y_train, 예측(Z_train[:, remain], w2, b2))
    r_te = R2(y_test, 예측(Z_test[:, remain], w2, b2))
    print(f"{특징이름[minus]:4s} 빼면 -> 학습용 {r_tr:.4f} / 시험용 {r_te:.4f}")


# [E2] 실행 결과를 보고 답하세요.
#        (1) 예상이 맞았나요?
#        (2) VIB-BOT_RMS 를 뺐을 때 시험용 점수가 어떻게 됐습니까?
#            왜 그럴까요?  힌트: d[특징이름].corr() 를 찍어 보세요.
#                                 VIB-BOT_RMS 와 VIB-BOT_PTP 의 상관은?
#        (3) VIB-TOP_RMS 를 뺐을 때는요? 정답과 상관이 0.09 밖에 안 되는데 왜?
#        (4) 여기서 얻을 교훈을 한 줄로 적으세요.
#      내 답:
#
# [참고 출력]  d[특징이름].corr().round(2)
#                  VIB-BOT_RMS  VIB-BOT_PTP  VIB-BOT_KUR  VIB-TOP_RMS
#     VIB-BOT_RMS         1.00         0.95        -0.35         0.59
#     VIB-BOT_PTP         0.95         1.00        -0.19         0.67
#     VIB-BOT_KUR        -0.35        -0.19         1.00         0.07
#     VIB-TOP_RMS         0.59         0.67         0.07         1.00
# TODO
print("\nE2")

print(d[특징이름].corr().round(2))

# =====================================================================
# F. 함정 3 — 섞어서 자른 게 정말 옳았나
# =====================================================================
# [F1] 이 데이터는 MEAS_DT(측정시각) 순으로 정렬돼 있습니다. 1월부터 12월까지.
#      02 에서는 "섞고 잘라라, 안 섞으면 편향된다" 고 배웠죠.
#      이번엔 반대로 해 보세요 — 섞지 말고 앞 399행을 학습용, 뒤 171행을 시험용으로.
#      (즉 1~9월로 배워서 10~12월을 맞히기)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 섞었을 때(C3)보다 높나요, 낮나요?
#        (2) 결과가 예상과 다를 겁니다. 그래도 실무에서 예측 모델을 만들 때는
#            보통 이 '시간순 분할' 쪽을 씁니다. 왜 그럴까요?
#            힌트: 현장에 배포된 모델이 맞혀야 하는 데이터는 '언제' 것입니까?
#      내 답: (1)
#             (2)
#
# [나와야 하는 출력]
#     학습용 399 대 (2024-01-10 ~ 2024-09-02) / 시험용 171 대 (2024-09-03 ~ 2024-12-15)
#     VIB-BOT_RMS  w = +8.386
#     VIB-BOT_PTP  w = +31.453
#     VIB-BOT_KUR  w = -7.456
#     VIB-TOP_RMS  w = -24.941
#     절편 b = 116.417
#     학습용 R² 0.7451   MSE 388.384
#     시험용 R² 0.7754   MSE 256.457
# TODO
print("\nF1")

order = list(range(len(X)))
tr, te = (order[:n_train], order[n_train:])

X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

w, b = 학습(Z_train, y_train)

print(
    f"학습용 {len(tr)}대 ({d[:n_train]['MEAS_DT'][0][:10]} ~ {d[:n_train]['MEAS_DT'][n_train - 1][:10]}) / 시험용 {len(te)}대 ({d[n_train:]['MEAS_DT'][n_train][:10]} ~ {d[n_train:]['MEAS_DT'][len(X) - 1][:10]})"
)

for 이름, wi in zip(특징이름, w):
    print(f"    {이름:6s} w = {wi:+.3f}")
print(f"    절편 b = {b:.3f}")

print(
    f"학습용(train) R2 : {round(R2(y_train, 예측(Z_train, w, b)), 4)} MSE : {round(MSE(y_train, 예측(Z_train, w, b)), 4)}"
)
print(
    f"시험용(test) R2 : {round(R2(y_test, 예측(Z_test, w, b)), 4)} MSE : {round(MSE(y_test, 예측(Z_test, w, b)), 4)}"
)

# =====================================================================
# G. 데이터가 몇 대면 충분한가
# =====================================================================
# [G1] 학습용을 5 / 10 / 30 / 100 / 399 대로 바꿔 가며 학습하고,
#      학습용 R2 와 시험용 R2 를 표처럼 찍으세요. (적은 데이터는 epochs 를 늘리세요)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 음수로 나오는 구간이 있습니다. 음수는 무슨 뜻입니까?
#            힌트: R2 = 0 이 '무조건 평균만 대답하는 모델' 입니다.
#        (2) 데이터가 늘 때 학습용 점수와 시험용 점수는 각각 어느 방향으로 움직입니까?
#        (3) 이 설비에서 쓸 만한 모델을 만들려면 최소 몇 건쯤 필요해 보입니까?
#      내 답:
#
# [나와야 하는 출력]
#     (02 §5-1 방식: Z_train[:n], y_train[:n] 으로 학습하고, 시험용은 B2 의 171 대 그대로.
#      epochs 는 5/10/30/100 대 = 2000, 399 대 = 500 으로 걸었습니다.)
#     학습용 n   학습용 R²   시험용 R²
#          5      0.9981     0.6330
#         10      0.9627    -0.0958
#         30      0.9033    -0.8177
#        100      0.8472     0.2495
#        399      0.7982     0.6080
#     (399 대도 2000 걸음 걸으면 시험용 0.6070. n 대만으로 mu, sd 를 다시 구해 표준화하면
#      5 대 줄만 0.9999 / 0.4331 로 달라지고 나머지는 소수 둘째 자리까지 같습니다)
# TODO
print("\nG1")

train_num = [5, 10, 30, 100, 399]
epoch_num = [2000, 2000, 2000, 2000, 500]

print(f"학습용 n    학습용 R2   시험용 R2")

order = np.random.RandomState(42).permutation(len(X))
n_train = int(len(X) * 0.7)
tr, te = order[:n_train], order[n_train:]

X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

for long, ep in zip(train_num, epoch_num):
    Z_small, y_small = Z_train[:long], y_train[:long]
    w, b = 학습(Z_small, y_small, epochs=ep)
    print(
        f"{long}    {round(R2(y_small, 예측(Z_small, w, b)), 4)}    {round(R2(y_test, 예측(Z_test, w, b)), 4)}"
    )

# =====================================================================
# H. 마무리 — 보고서 3줄
# =====================================================================
# 팀장에게 보고한다고 치고, 아래 세 줄을 채우세요.
#
#   1) 전류계가 고장 났을 때 진동으로 전류를 추정할 수 있는가? (된다/안 된다/조건부)
#      근거 점수:
#
#   2) 이 모델을 쓸 때 반드시 붙여야 할 경고 문구 한 줄:
#
#   3) 점수를 더 올리려면 다음에 뭘 해 보겠는가? (한 가지만, 이유와 함께)
#
# =====================================================================
