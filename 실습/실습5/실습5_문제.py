# =====================================================================
#  실습 5 — 실습 2·3 을 파이토치로 다시 (05 파이토치)
# =====================================================================
#  실행: python 실습5_문제.py      (이 파일이 있는 폴더에서)
#  필요한 것: numpy, pandas, torch, scikit-learn  /  데이터는 옆의 데이터/ 폴더에 있습니다.
#
#  [상황]
#    실습 2 와 3 의 데이터와 질문을 그대로 다시 씁니다.
#      · 압연기(SPM01) 진동 4개로 모터 전류(CUR-MTR_RMS) 맞히기     ← 실습 2 (회귀)
#      · 배풍기 진동 5개로 1호기 / 2호기 맞히기                      ← 실습 3 (분류)
#
#    실습 2·3 에서는 기울기를 '밟아 보며' 손으로 구했습니다.
#    실습 4 에서는 사이킷런이 공식으로 단번에 풀었습니다.
#    오늘은 파이토치로 같은 일을 합니다. 05 의 한 문장이 진짜인지 직접 확인하는 날입니다.
#        "파이토치가 대신 해 주는 건 기울기 계산 하나뿐이다.
#         나머지(손실 정하기, 한 걸음 내려가기)는 손코드와 한 글자도 다르지 않다"
#
#    그래서 오늘의 채점 기준은 간단합니다.
#        · 실습 2 의 숫자 (0.7982 / 0.6080, 가중치 +24.692 …) 가 그대로 나오는가     → B
#        · 실습 3 의 숫자 (손실 0.5062, 정확도 0.744 …) 가 그대로 나오는가            → C
#    안 나오면 어딘가 손코드와 '다른 걸음' 을 걸은 겁니다. 그걸 찾는 게 오늘의 연습입니다.
#
#  [쓰는 도구]  05 에서 배운 것 전부.
#    텐서·float32 / requires_grad·backward / nn.Linear·MSELoss·SGD / 루프 네 줄 /
#    BCEWithLogitsLoss·torch.sigmoid·no_grad / TensorDataset·DataLoader·Adam
#    ※ 05_파이토치_같은일을_루프로.py 와 실습2_문제.py, 실습3_문제.py 를 옆에 띄워 놓으세요.
#      분할과 표준화는 실습 2·3 에서 가져오고, 학습만 05 방식으로 바꿉니다.
#
#  [힌트를 이렇게 줍니다]
#    · 어느 표(d / e)의 어느 열을 쓰는지, 변수 이름, 분할·표준화 코드는 전부 적어 뒀습니다.
#      이건 실습 2·3 에서 이미 한 것이라 오늘 연습할 내용이 아닙니다. 그대로 쓰세요.
#    · 오늘 연습할 것(파이토치 부품과 루프)은 '어떤 부품을 쓰는지' 와 '05 의 몇 번' 까지만 드립니다.
#      호출하는 줄은 05 를 보고 직접 쓰세요.
#    · 05 에 안 나오는 문법은 코드를 그대로 드립니다. (B1 의 0 으로 시작하기)
#    · 다른 실습 파일은 두 가지 때만 엽니다.
#        ① 함수 복사: 실습 2 의 손실()·h·기울기_밟아보기() (A3), 실습 3 의 네칸() (C2)
#        ② 숫자 비교: ★ 답하세요 ★ 에서 실습 2·3·4 의 결과와 견줄 때
#
#  [오늘의 시간 배분]  코드는 40분이면 다 칩니다.
#    나머지는 ★ 답하세요 ★ 에 쓰세요. 특히 B3, C3, D2 의 '일부러 틀려 보기' 가 오늘의 핵심입니다.
# =====================================================================

# =====================================================================
#  [정답출력 참고]  각 항목의 [나와야 하는 출력] 은 이렇게 만든 값입니다
# =====================================================================
#  · torch 2.14.0+cpu / scikit-learn 1.9.0 / numpy 2.5.3 / pandas 3.0.5 (이 프로젝트의 venv) 로
#    직접 돌려서 확인한 값입니다. 같은 venv 면 같은 숫자가 나옵니다.
#  · 파이토치는 float32 로 계산합니다. 손코드(float64)와 소수 넷째 자리가 가끔 1 다를 수 있습니다.
#    셋째 자리까지 같으면 맞은 겁니다.
#  · torch.manual_seed 를 '어디서' 부르느냐에 따라 숫자가 달라집니다.
#    각 항목에 적은 순서(씨앗 → 모델 → 옵티마이저 → 로더) 그대로 쓰면 정답지와 똑같이 나옵니다.
#  · [B1] [C1] 은 실습 2·3 과 똑같이 '가중치 0 에서 출발' 하도록 만든 값입니다.
#    파이토치 기본값(난수 출발)으로 두면 B2 의 500 줄처럼 소수 셋째 자리가 조금 다릅니다.
#  · [B3] 은 경고(UserWarning)가 500 번 뜹니다. 에러가 아니고, 그 경고가 이 항목의 주제입니다.
#  · [D1] 은 매 에폭 데이터를 새로 섞습니다. 씨앗이 다르면 숫자가 달라지는 게 정상입니다.
# =====================================================================

import os
import numpy as np
import pandas as pd
# import torch 와 사이킷런 도구는 각 항목에서 직접 쓰세요. 05 맨 위와 §4 에 있습니다.

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")
d = pd.read_csv(
    os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig"
)  # 실습 2 데이터
e = pd.read_csv(
    os.path.join(DATA, "T-SP-EXF01_진동특징.csv"), encoding="utf-8-sig"
)  # 실습 3 데이터
# ↑ encoding="utf-8-sig" 빠뜨리면 첫 열 이름이 깨져서 KeyError 납니다.

# =====================================================================
#  [어느 표를 언제 쓰나]  — 표가 두 개라 헷갈리기 쉽습니다. 먼저 읽으세요
# =====================================================================
#                 표   행 수   입력(X)                          정답(y)
#   A · B · D    d    570    진동 4개 (아래 특징이름)            CUR-MTR_RMS  (모터 전류, 숫자)    ← 회귀
#   C            e    558    진동 5개 (C 머리의 특징c)          2호기면 1, 1호기면 0               ← 분류
#
#   · A·B·D 에서 만드는 이름: X, y → X_t, y_t → X_train, X_test, y_train, y_test → Ztr, Zte, Ytr, Yte
#   · C 에서 만드는 이름:     Xc, yc → Zc_tr, Zc_te, Yc_tr, yc_te   (이름에 c 가 붙습니다)
#   · e 는 C 에 가서야 씁니다. A·B·D 에서는 d 만 쓰세요.
# =====================================================================

특징이름 = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS"]
# [0] 압연 데이터 d 에서 입력 X 와 정답 y 를 만드세요. (아직 numpy 입니다. 텐서는 A1 에서)
#       X : d 의 특징이름 4개 열     →  d[특징이름].values.astype(float)
#       y : d 의 "CUR-MTR_RMS" 열    →  d["CUR-MTR_RMS"].values.astype(float)
#     실습 2 §B1 과 똑같은 두 줄입니다. 모양을 찍어서 확인하세요.
#
# [나와야 하는 출력]  (숫자만 맞으면 됩니다. 출력 모양까지 똑같을 필요는 없습니다)
#     X.shape: (570, 4) / y.shape: (570,)
# TODO
print("[0]")
X = d[특징이름].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)

print(X.shape)
print(y.shape)

# =====================================================================
# A. 텐서와 자동 미분 — 파이토치가 대신 해 주는 단 하나        (05 §1, §2)
# =====================================================================
# [A1] [0] 의 X, y 를 텐서로 바꿔 X_t, y_t 에 담고, 모양과 dtype 을 찍으세요. (05 §1)
#        X_t = torch.tensor(X, dtype=torch.float32)   ← y_t 도 같은 모양으로
#      그리고 비교용으로 dtype 을 '안 붙이고' 만든 텐서 torch.tensor(X) 의 dtype 도 찍어 보세요.
#      마지막으로 X_t 의 앞 2행을 찍으세요.
#      (X_t, y_t 는 이 항목에서 확인만 하고 끝입니다. 학습에는 A2 에서 새로 만드는 텐서를 씁니다)
#
#      ★ 답하세요 ★
#        dtype 을 안 붙이면 무엇이 됩니까? 왜 그렇게 됩니까? (X 는 원래 무슨 타입이었죠?)
#        05 §1 은 "파이토치 부품들은 float32 를 기대한다" 고 했습니다.
#        이걸 안 지키면 언제 터질까요? 텐서를 만들 때? 아니면 나중에?
#      내 답:
#
# [나와야 하는 출력]
#     X_t: torch.Size([570, 4]) torch.float32 / y_t: torch.Size([570]) torch.float32
#     dtype 안 붙이면: torch.float64
#     X_t[:2] = tensor([[ 0.1629,  0.5425, -1.3589,  0.0687],
#                       [ 0.1322,  0.4254, -1.3252,  0.0731]])
# TODO
print("\nA1")
import torch

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

print(f"X_t : {X_t.shape} {X_t.dtype} / y_t : {y_t.shape} {y_t.dtype}")

X_t = torch.tensor(X)
print(f"dtype을 안 붙이면 : {X_t.dtype}")

X_t = torch.tensor(X, dtype=torch.float32)
print(X_t[:2])

# [A2] 실습 2 §B2, §B3 와 '똑같은' 분할과 표준화를 numpy 로 한 다음, 텐서로 바꾸세요.
#      ★ train_test_split 이나 StandardScaler 를 쓰면 안 됩니다 ★
#        오늘은 실습 2 와 숫자가 똑같이 나오는지 보는 게 목적이라, 분할이 한 줄이라도 다르면 안 됩니다.
#
#      ① 섞어서 7:3 으로 나누기 (실습 2 §B2)
#           순서 = np.random.RandomState(42).permutation(len(X))
#           n_train = int(len(X) * 0.7)                 # 399
#           tr = 순서[:n_train],  te = 순서[n_train:]
#           X_train, X_test = X[tr], X[te]  /  y_train, y_test = y[tr], y[te]
#      ② 열별 표준화 (실습 2 §B3) — mu, sd 는 학습용에서만
#           mu = X_train.mean(axis=0),  sd = X_train.std(axis=0)
#           Z_train = (X_train - mu) / sd,  Z_test = (X_test - mu) / sd     ← 시험용도 학습용의 mu, sd 로
#      ③ 텐서로 바꾸기 — 이름은 Ztr, Zte, Ytr, Yte
#           Ztr, Zte  ← Z_train, Z_test 를 float32 텐서로
#           Ytr, Yte  ← y_train, y_test 를 float32 텐서로 + .reshape(-1, 1) 로 (건수, 1) 세로 모양
#           (세로 모양이 왜 필요한지는 B3 에서 직접 봅니다)
#           힌트: 05 §4 의 to_t 도우미를 가져오면 네 줄이 짧아집니다.
#      ④ 확인: 네 텐서의 모양, 그리고 Ztr 의 열별 평균과 퍼짐을 찍으세요.
#           Ztr.mean(0) / Ztr.std(0, unbiased=False)   (unbiased=False = numpy 와 같은 방식. 05 §2)
#
# [나와야 하는 출력]
#     Ztr torch.Size([399, 4]) / Zte torch.Size([171, 4]) / Ytr torch.Size([399, 1]) / Yte torch.Size([171, 1])
#     표준화 후 학습용 열별 평균: [0. 0. 0. 0.]  / 퍼짐: [1. 1. 1. 1.]
#     (평균이 -0. 으로 찍혀도 맞습니다. 아주 작은 음수가 반올림된 것)
# TODO
print("\nA2")
order = np.random.RandomState(42).permutation(len(X))
n_train = int(len(X) * 0.7)
tr = order[:n_train]
te = order[n_train:]
X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

Ztr = torch.tensor(Z_train, dtype=torch.float32)
Zte = torch.tensor(Z_test, dtype=torch.float32)
Ytr = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
Yte = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print(f" Ztr : {Ztr.shape} / Zte : {Zte.shape} / Ytr : {Ytr.shape} / Yte : {Yte.shape}")
print(
    f"표준화 후 학습용 열별 평균 : {Ztr.mean(0).round(decimals=3)} / 퍼짐 : {Ztr.std(0, unbiased=False).round(decimals=3)}"
)

# [A3] 자동 미분. (05 §2)
#      ① 손잡이 만들기
#           w = torch.zeros(4, requires_grad=True)       # 가중치 4칸, 전부 0
#           b = torch.tensor(0.0, requires_grad=True)    # 절편, 0
#      ② 실습 2 의 손실을 '텐서로' 계산하고 backward() 하기
#           손실 = ((정답 - (Ztr @ w + b)) ** 2).mean()
#           ★ 여기서 '정답' 은 Ytr 이 아닙니다 ★
#             y_train 을 reshape 없이 텐서로만 바꾼 것(모양 (399,))을 쓰세요. 이름은 자유 (예: ytr1).
#             Ztr @ w 가 (399,) 모양이라 정답도 (399,) 여야 짝이 맞습니다. B3 에서 이 얘기를 다시 합니다.
#           손실 값과 w.grad, b.grad 를 찍으세요.
#      ③ 비교용으로 같은 기울기를 numpy 로 '밟아서' 구하기
#           실습 2 [C1] 의 손실(), h = 0.0001, 기울기_밟아보기() 셋을 복사해 오세요.
#           (기울기_밟아보기 는 손실 을 부르고, h 도 씁니다. 하나라도 빠지면 NameError)
#           기울기_밟아보기(Z_train, y_train, np.zeros(4), 0.0) 을 찍어 ② 와 나란히 비교합니다.
#           ※ 여기서는 numpy 인 Z_train, y_train 을 넣습니다. 텐서 Ztr 이 아닙니다.
#      ④ ② 의 손실 계산과 backward() 를 한 번 더 하고 (w, b 는 그대로 두고) w.grad, b.grad 를 다시 찍으세요.
#
#      ★ 답하세요 ★
#        (1) 파이토치 값과 밟아보기 값이 몇 자리까지 같습니까?
#            밟아보기는 손실을 10번(w 4개 × 2 + b × 2) 계산했습니다. backward 는 몇 번입니까?
#            특징이 1000개라면 둘의 차이는 어떻게 됩니까?
#        (2) backward 를 두 번 부르면 왜 두 배가 됩니까? 이게 B 의 루프 첫 줄과 무슨 관계입니까?
#      내 답: (1)
#             (2)
#
# [나와야 하는 출력]
#     w=0, b=0 에서 손실 14214.182
#     파이토치   w 방향 [-61.167 -54.472  38.66   -6.177]   b 방향 -225.279
#     밟아보기   w 방향 [-61.167 -54.472  38.66   -6.177]   b 방향 -225.279
#     한 번 더 backward 하면  w 방향 [-122.335 -108.944   77.321  -12.354]   b 방향 -450.559  <- 두 배
# TODO
print("\nA3")
w = torch.zeros(4, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

ytr1 = torch.tensor(y_train, dtype=torch.float32)
loss = ((ytr1 - (Ztr @ w + b)) ** 2).mean()
print(f"w=0, b=0 에서 손실 : {loss.round(decimals=3)}")
loss.backward()
print(f"파이토치 w 방향 {w.grad.round(decimals=3)} b 방향 {b.grad.round(decimals=3)}")
loss = ((ytr1 - (Ztr @ w + b)) ** 2).mean()
loss.backward()
print(
    f"한 번 더 backward 하면 w 방향 {w.grad.round(decimals=3)} b 방향 {b.grad.round(decimals=3)}"
)

# =====================================================================
# B. 표준 루프로 실습 2 를 다시 — 같은 숫자가 나오는가          (05 §3, §4)
# =====================================================================
# [B1] A2 의 Ztr(입력), Ytr(정답, 세로 모양) 로 실습 2 를 파이토치 표준 루프로 다시 학습하세요.
#        부품: model = torch.nn.Linear(4, 1)  /  loss_fn = torch.nn.MSELoss()  /
#              opt = torch.optim.SGD(model.parameters(), lr=0.1)
#        루프: 500 에폭. 실습 2 와 같은 lr, 같은 걸음 수입니다. 네 줄은 05 §3 그대로.
#
#      찍을 것
#        · 루프 안에서 epoch 0, 10, 50, 100, 499 일 때 loss.item()
#        · 학습이 끝난 뒤 학습용·시험용 R2 와 MSE
#            R2  : 05 §4 의 텐서용 R2(y, yhat) 함수를 복사해 와서 R2(Ytr, model(Ztr)), R2(Yte, model(Zte))
#            MSE : loss_fn 이 곧 MSE 입니다. loss_fn(model(Zte), Yte).item()
#            ★ 채점 줄들은 with torch.no_grad(): 안에 넣으세요 (05 §4). 구경만 하니까 기록할 필요 없음
#        · 가중치 4개와 절편
#            model.weight 는 (1, 4) 모양입니다. model.weight.squeeze(0).tolist() 로 4개짜리 리스트가 됩니다.
#            절편은 model.bias.item()
#
#      ★ 0 에서 출발시키기 — 05 에 없는 문법이라 그대로 드립니다 ★
#        실습 2 는 w, b 를 0 에서 출발했습니다. 파이토치는 난수에서 출발합니다.
#        숫자를 똑같이 맞추려면 모델을 만든 직후 이 두 줄을 넣으세요:
#            with torch.no_grad():
#                model.weight.zero_()     # 밑줄로 끝나는 함수 = 그 자리에서 값을 바꾼다
#                model.bias.zero_()
#        (no_grad 로 감싸는 이유: 이건 학습이 아니라 '초기 설정' 이라 기록할 필요가 없어서)
#      순서: torch.manual_seed(0) → 모델 → 0 으로 → 손실함수 → 옵티마이저 → 루프
#
#      ★ 답하세요 ★
#        (1) 실습 2 [C2] [C3] 의 숫자와 비교하세요. 어디까지 같습니까?
#        (2) 실습 2 는 기울기를 '밟아서', 오늘은 '식을 거꾸로 따라가서' 구했습니다.
#            방법이 다른데 왜 같은 숫자가 나옵니까? A3 의 결과로 설명하세요.
#        (3) epoch 0 의 손실 14214 는 A3 에서 본 숫자입니다. 왜 그렇습니까?
#      내 답: (1)
#             (2)
#             (3)
#
# [나와야 하는 출력]
#     epoch   0  손실 14214.1816
#     epoch  10  손실 508.2359
#     epoch  50  손실 308.6377
#     epoch 100  손실 308.2807
#     epoch 499  손실 307.9932
#     학습용 R² 0.7982  MSE 307.993
#     시험용 R² 0.6080  MSE 519.897
#     VIB-BOT_RMS  w = +24.692
#     VIB-BOT_PTP  w = +14.966
#     VIB-BOT_KUR  w = -5.908
#     VIB-TOP_RMS  w = -18.962
#     절편 b = 112.640
#     (실습 2: 0.7982 / 0.6080, MSE 307.993 / 519.897, +24.692 / +14.966 / -5.908 / -18.962, b 112.640)
# TODO
print("\nB1")

torch.manual_seed(0)

model = torch.nn.Linear(4, 1)

with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(500):
    opt.zero_grad()
    loss = loss_fn(model(Ztr), Ytr)
    loss.backward()
    opt.step()
    if epoch in (0, 10, 50, 100, 499):
        print(f" epoch {epoch} 손실 {loss.item():.4f}")


def R2(y, yhat):
    return (1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()).item()


with torch.no_grad():
    print(
        f"학습용 R2 {R2(Ytr, model(Ztr)):.4f} MSE {loss_fn(model(Ztr), Ytr).item():.3f}"
    )
    print(
        f"시험용 R2 {R2(Yte, model(Zte)):.4f} MSE {loss_fn(model(Zte), Yte).item():.3f}"
    )
    weights = model.weight.squeeze(0).tolist()
    for idx in range(4):
        print(f"{특징이름[idx]} w = {weights[idx]:.3f}")
    print(f"절편 b : {model.bias.item():.3f}")

# [B2] 05 맨 아래 실습 문제를 여기서 풉니다 — 몇 걸음이면 충분한가.
#      B1 과 같은 데이터(Ztr, Ytr), 같은 부품(Linear(4,1) / MSELoss / SGD lr=0.1)입니다. 다른 점은 둘.
#        · 0 으로 바꾸는 두 줄을 빼세요. 파이토치 기본값(난수 출발)으로 둡니다.
#        · 에폭을 5 / 50 / 500 / 2000 으로 바꿔 가며 네 번 학습합니다. (for 에폭수 in [5, 50, 500, 2000]: 로 감싸기)
#      ★ 매번 torch.manual_seed(0) → 모델 → 옵티마이저 순서로 처음부터 다시 만드세요 ★
#        모델을 한 번만 만들고 이어서 걸으면 '5 + 50 + …' 걸음이 됩니다. 출발점이 같아야 공정한 비교입니다.
#      각 줄에 학습용·시험용 R2, 가중치, 절편을 찍으세요.
#      그리고 첫 번째(5 에폭) 모델을 만든 직후, 학습 전 가중치·절편(= 출발점)도 한 번 찍으세요.
#
#      ★ 답하세요 ★
#        (1) 5 걸음에서 R2 가 음수입니다. 실습 2 §G 에서 R2 = 0 의 뜻을 배웠죠. 음수는 무슨 뜻입니까?
#            절편 b 를 보세요. 정답 평균(112.64)에 아직 얼마나 못 갔습니까?
#        (2) 2000 걸음 줄을 실습 4 [A2] [A3] 의 사이킷런 결과(0.7982 / 0.6070, +24.902 …)와 비교하세요.
#            무엇과 같습니까? 왜 걸음을 늘리면 사이킷런 답에 도착합니까?
#        (3) 50 걸음에서 시험용 R2 (0.6181) 가 500 걸음, 2000 걸음보다 높습니다.
#            그럼 50 걸음에서 멈추는 게 정답입니까?
#            힌트: 이 숫자를 보고 50 을 고르면, 실습 4 [E] 에서 말한 그 반칙이 됩니다.
#                 그리고 실습 4 [D1] 에서 씨앗만 바꿨을 때 시험용이 얼마나 흔들렸죠?
#      내 답: (1)
#             (2)
#             (3)
#
# [나와야 하는 출력]
#     출발점: w = [-0.0037  0.2682 -0.4115 -0.368 ]  b = -0.1926
#     에폭    학습용 R²   시험용 R²   가중치                                   절편
#        5   -0.1845    -0.4569    [+14.148 +12.277 -10.019  -4.426]   75.667
#       50    0.7978     0.6181    [+21.607 +17.623  -6.693 -18.700]  112.638
#      500    0.7982     0.6080    [+24.683 +14.976  -5.909 -18.963]  112.640
#     2000    0.7982     0.6070    [+24.902 +14.759  -5.869 -18.949]  112.640   ← 실습 4 사이킷런과 같음
# TODO
print("\nB2")

epoch_list = [5, 50, 500, 2000]
loss_fn = torch.nn.MSELoss()

print("에폭    학습용 R²   시험용 R²   가중치                                   절편")
for epoch_num in epoch_list:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # if epoch_num == 5:
    #     w0 = np.round(model.weight.squeeze(0).tolist(), 4)
    #     print(f"출발점 : w = {w0} b = {model.bias.item():.4f}")

    for epoch in range(epoch_num):
        opt.zero_grad()
        loss = loss_fn(model(Ztr), Ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        weights = model.weight.squeeze(0).tolist()
        w_str = " ".join(f"{v:+.3f}" for v in weights)
        print(
            f"{epoch_num} {R2(Ytr, model(Ztr)):.4f} {R2(Yte, model(Zte)):.4f} [{w_str}] {model.bias.item():.3f}"
        )


# [B3] 일부러 틀려 보기 — 05 §3 이 "제일 위험한 실수" 라고 한 것.
#      B1 을 처음부터 그대로 다시 하되(씨앗 0 → 새 모델 → 0 으로 → 옵티마이저, 500 에폭),
#      루프 안 손실 계산의 정답만 Ytr 대신 A3 에서 만든 1줄짜리 정답(모양 (399,))을 넣으세요.
#          loss = loss_fn(model(Ztr), 1줄짜리_정답)       ← 바뀌는 건 이 한 줄뿐
#      찍을 것
#        · epoch 0 과 499 의 손실
#        · 학습이 끝난 뒤 학습용·시험용 R2, 가중치, 절편
#          ★ R2 채점은 B1 과 똑같이 Ytr, Yte(세로 모양)로 하세요 ★ 학습만 틀리게 한 겁니다.
#        · 모양 세 개: model(Ztr).shape / 1줄짜리_정답.shape / (model(Ztr) - 1줄짜리_정답).shape
#
#      ★ 답하세요 ★
#        (1) 에러가 납니까? 무엇이 뜹니까?
#        (2) (model(Ztr) - 정답) 의 모양이 몇 × 몇입니까? 원래 몇 개짜리 오차여야 합니까?
#            이렇게 늘어나는 걸 '브로드캐스팅' 이라고 합니다. 모양이 안 맞으면 numpy·파이토치가
#            '알아서' 늘려서 맞춥니다. 그래서 에러가 안 납니다.
#        (3) 가중치가 전부 0, 절편이 112.640 입니다. 이 모델은 무엇을 예측합니까?
#            R2 가 정확히 0 이 나온 이유를 설명하세요. (실습 4 [F2] 의 alpha=100 과 같은 상황입니다)
#        (4) 이 실수가 에러보다 더 위험한 이유를 한 줄로 적으세요.
#      내 답: (1)
#             (2)
#             (3)
#             (4)
#
# [나와야 하는 출력]
#     UserWarning: Using a target size (torch.Size([399])) that is different to the input size
#                  (torch.Size([399, 1])). This will likely lead to incorrect results ...   (에러 아님, 경고만)
#     epoch   0  손실 14214.1836
#     epoch 499  손실 1526.4727
#     학습용 R² 0.0000 / 시험용 R² -0.0152
#     가중치 [-0. -0. -0. -0.]  절편 b = 112.640
#     모양: 예측 torch.Size([399, 1]) / 정답 torch.Size([399]) / 빼면 torch.Size([399, 399])
# TODO
print("\nB3")

torch.manual_seed(0)

model = torch.nn.Linear(4, 1)

with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(500):
    opt.zero_grad()
    loss = loss_fn(model(Ztr), ytr1)
    loss.backward()
    opt.step()
    if epoch in (0, 499):
        print(f" epoch {epoch} 손실 {loss.item():.4f}")

with torch.no_grad():
    print(f"학습용 R2 {R2(Ytr, model(Ztr)):.4f} 시험용 R2 {R2(Yte, model(Zte)):.4f}")
    weights = model.weight.squeeze(0).tolist()
    print(f"가중치 {np.round(weights, 3)} 절편 b : {model.bias.item():.3f}")
    print(
        f"모양 : 예측 {model(Ztr).shape} 정답 {ytr1.shape} 빼면 {(model(Ztr) - ytr1).shape}"
    )


# =====================================================================
# C. 분류 — 실습 3 을 파이토치로. 손실 이름 하나만 바뀐다     (05 §5)
# =====================================================================
특징c = ["VIB-MDE_1X", "VIB-MNDE_RMS", "VIB-FNDE_RMS", "VIB-MDE_NSY", "VIB-MDE_HF"]
# [C1] 이제부터는 배풍기 데이터 e 를 씁니다. (d 가 아닙니다)
#      정답 = 이 행이 2호기냐 아니냐. 2호기(EQP_CD 가 "P1-SP2-EXF01")면 1, 1호기면 0.
#
#      ① 입력과 정답 (실습 3 [A2])
#           Xc = e[특징c].values.astype(float)                              # (558, 5)
#           yc = (e["EQP_CD"] == "P1-SP2-EXF01").astype(float).values        # (558,)  합 = 279
#      ② 양쪽 편을 따로 섞어 7:3 으로 나누기 (실습 3 [B1]) — 그대로 쓰세요
#           rng = np.random.RandomState(3)
#           i1 = rng.permutation(np.where(yc == 1)[0])   # 2호기 279건
#           i0 = rng.permutation(np.where(yc == 0)[0])   # 1호기 279건
#           trc = np.concatenate([i1[:195], i0[:195]])   # 학습용 390건
#           tec = np.concatenate([i1[195:], i0[195:]])   # 시험용 168건
#      ③ 표준화 (실습 3 [B2]) — 학습용 Xc[trc] 의 평균·표준편차로 학습용·시험용 둘 다 변환
#      ④ 텐서로
#           Zc_tr, Zc_te ← 표준화한 학습용·시험용 입력, float32 텐서
#           Yc_tr        ← yc[trc] 를 float32 텐서로 + .reshape(-1, 1)   (0/1 도 float32 여야 합니다. 05 §5)
#           yc_te        ← yc[tec] 그대로 numpy. 채점에만 쓰니 텐서로 안 바꿉니다
#      ⑤ 학습: nn.Linear(5, 1) / torch.nn.BCEWithLogitsLoss() / SGD(lr=0.5), 2000 에폭
#           순서: torch.manual_seed(0) → 모델 → 0 으로(B1 의 두 줄) → 손실함수 → 옵티마이저 → 루프
#           루프 안에서 epoch % 400 == 0 일 때 손실을 찍으세요.
#      ⑥ 가중치를 크기순으로 찍기
#           w = 모델.weight.squeeze(0).tolist() 로 꺼낸 뒤
#           np.argsort(np.abs(w))[::-1] 순서대로 특징c 이름과 함께 찍고, 절편도 찍으세요.
#
#      ★ sigmoid 를 모델 뒤에 붙이지 마세요 ★  (붙이면 어떻게 되는지는 C3 에서 직접 봅니다)
#
#      ★ 답하세요 ★
#        실습 3 [C2] 와 비교하세요. 손실 줄, 가중치, 절편이 같습니까?
#        05 §5 는 "회귀와 다른 유일한 줄" 이라고 했습니다. B1 코드와 지금 코드를 나란히 놓고
#        정말 몇 줄이 다른지 세어 보세요. (분할·표준화 말고 학습 부분만)
#      내 답:
#
# [나와야 하는 출력]
#     epoch    0  손실 0.6931        ← 실습 3 은 0.6648. 아래 ※ 참고
#     epoch  400  손실 0.5146
#     epoch  800  손실 0.5076
#     epoch 1200  손실 0.5064
#     epoch 1600  손실 0.5062
#     표준화 눈금 가중치 (크기순):
#       VIB-FNDE_RMS  -4.515
#       VIB-MDE_HF    +2.475
#       VIB-MDE_NSY   -2.109
#       VIB-MDE_1X    +0.688
#       VIB-MNDE_RMS  -0.303
#     절편 b = -0.043
#     ※ epoch 0 만 다른 이유: 실습 3 의 학습() 은 '한 걸음 걸은 뒤' 손실을 찍었고,
#       표준 루프의 loss 는 '걸음 직전' 에 계산한 값입니다. 0.6931 = 전부 반반(확률 0.5)일 때의 손실.
#       400 부터는 한 걸음 차이가 소수 넷째 자리에 안 보일 만큼 작아서 똑같이 나옵니다.
# TODO
print("\nC1")
Xc = e[특징c].values.astype(float)
yc = (e["EQP_CD"] == "P1-SP2-EXF01").astype(float).values

rng = np.random.RandomState(3)
i1 = rng.permutation(np.where(yc == 1)[0])
i0 = rng.permutation(np.where(yc == 0)[0])
trc = np.concatenate([i1[: int(len(i1) * 0.7)], i0[: int(len(i0) * 0.7)]])
tec = np.concatenate([i1[int(len(i1) * 0.7) :], i0[int(len(i0) * 0.7) :]])

Xc_train = Xc[trc]
Xc_test = Xc[tec]
yc_train = yc[trc]
yc_test = yc[tec]

mu = Xc_train.mean(axis=0)
std = Xc_train.std(axis=0)

Zc_train = (Xc_train - mu) / std
Zc_test = (Xc_test - mu) / std

Zc_tr = torch.tensor(Zc_train, dtype=torch.float32)
Zc_te = torch.tensor(Zc_test, dtype=torch.float32)
Yc_tr = torch.tensor(yc_train, dtype=torch.float32).reshape(-1, 1)
Yc_te = yc_test

torch.manual_seed(0)

model = torch.nn.Linear(5, 1)

with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

loss_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.5)
for epoch in range(2000):
    opt.zero_grad()
    loss = loss_fn(model(Zc_tr), Yc_tr)
    loss.backward()
    opt.step()
    if epoch % 400 == 0:
        print(f"epoch : {epoch} / 손실 : {loss.item():.4f}")

with torch.no_grad():
    w = model.weight.squeeze(0).tolist()
    print("표준화 눈금 가중치 (크기순) : ")
    for j in np.argsort(np.abs(w))[::-1]:
        print(f"{특징c[j]} {w[j]:+.3f}")
    print(f"절편 b = {model.bias.item():.3f}")


# [C2] 시험용 확률을 구해서 임계값 0.5 / 0.4 / 0.3 / 0.2 의 표를 찍으세요. (실습 3 [E1] 과 같은 표)
#      ① 확률 꺼내기 (05 §5)
#           with torch.no_grad():
#               p = torch.sigmoid(모델(Zc_te)).squeeze(1).numpy()     # (168,) 짜리 2호기 확률
#      ② 임계값마다 판정 = (p >= 임계값).astype(int) 를 만들고, yc_te 와 비교해서 네 칸을 셉니다.
#           잡음 TP = 2호기인데 2호기라 함      놓침 FN = 2호기인데 1호기라 함
#           헛경보 FP = 1호기인데 2호기라 함    통과 TN = 1호기인데 1호기라 함
#           재현율 = TP / (TP + FN)   정밀도 = TP / (TP + FP)   정확도 = (TP + TN) / 168
#         실습 3 [C1] 의 네칸() 함수를 복사해 와서 쓰면 한 줄입니다:  TP, FN, FP, TN = 네칸(yc_te, 판정)
#
# [나와야 하는 출력]
#     임계값  잡음  놓침  헛경보  재현율  정밀도   (정확도)
#     0.5      69    15     28    0.82    0.71    (0.744)
#     0.4      75     9     39    0.89    0.66    (0.714)
#     0.3      80     4     47    0.95    0.63    (0.696)
#     0.2      82     2     51    0.98    0.62    (0.685)
#     (실습 3 [E1] 과 한 칸도 다르지 않습니다)
# TODO
print("\nC2")
print(f"임계값  잡음  놓침  헛경보  재현율  정밀도   (정확도)")
with torch.no_grad():
    p = torch.sigmoid(model(Zc_te)).squeeze(1).numpy()


def four(y, 판정):
    TP = int(((y == 1) & (판정 == 1)).sum())
    FN = int(((y == 1) & (판정 == 0)).sum())
    FP = int(((y == 0) & (판정 == 1)).sum())
    TN = int(((y == 0) & (판정 == 0)).sum())  # .sum() 으로 True 개수 세기
    return TP, FN, FP, TN


for conf in [0.5, 0.4, 0.3, 0.2]:
    verdict = (p >= conf).astype(int)

    TP, FN, FP, TN = four(Yc_te, verdict)

    재현율 = TP / (TP + FN) if TP + FN else 0
    정밀도 = TP / (TP + FP) if TP + FP else 0
    정확도 = np.mean(verdict == Yc_te)

    print(
        f"{conf}    {TP}    {FN}    {FP}    {재현율:.2f}    {정밀도:.2f}    ({정확도:.3f})"
    )

# [C3] 일부러 틀려 보기 — sigmoid 두 번 누르기.
#      C1 의 ⑤ 학습을 새 모델로 처음부터 그대로 다시 하되(씨앗 0 → 모델 → 0 으로 → … 2000 에폭),
#      루프 안 손실 계산 한 줄만 이렇게 바꾸세요.
#          loss = loss_fn(torch.sigmoid(모델(Zc_tr)), Yc_tr)     ← sigmoid 를 한 번 씌워서 넣음
#      (= "손실에는 확률을 넣어야 할 것 같아서" 붙인 상황. 실무에서 아주 흔한 실수입니다)
#      찍을 것
#        · 400 에폭마다 손실, 그리고 2000 걸음이 끝난 뒤의 손실 (위 식으로 no_grad 안에서 한 번 더 계산)
#        · 시험용 채점: C2 의 ① 과 똑같이 sigmoid 를 '한 번' 씌운 확률로, 임계값 0.5 에서
#          정확도 / 잡음 / 놓침 / 헛경보
#
#      ★ 답하세요 ★
#        (1) 2000 걸음 뒤 손실이 몇입니까? C1 (0.5062) 과 비교하세요.
#        (2) BCEWithLogitsLoss 안에서는 무슨 일이 일어납니까? 이름의 Logits 가 무슨 뜻이었죠?
#            그럼 지금 모델 출력은 sigmoid 를 몇 번 지나갑니까?
#        (3) sigmoid 를 두 번 지나면 확률이 0.5 ~ 0.731 사이에서만 나옵니다. (sigmoid(0)=0.5, sigmoid(1)=0.731)
#            이 상태로 손실을 줄일 수 있는 한계가 있는 이유를 설명하세요.
#            힌트: 정답이 0 인 1호기에게 "확률 0.5 이하" 라고 말할 방법이 있습니까?
#        (4) 이것도 에러가 안 납니다. B3 와 공통점이 무엇입니까?
#      내 답: (1)
#             (2)
#             (3)
#             (4)
#
# [나와야 하는 출력]
#     epoch    0  손실 0.7241
#     epoch  400  손실 0.6557
#     epoch  800  손실 0.6520
#     epoch 1200  손실 0.6499
#     epoch 1600  손실 0.6478
#     (2000 걸음 후 손실 0.6451)
#     시험용 (임계값 0.5)  정확도 0.655 / 잡음(TP) 26  놓침(FN) 58  헛경보(FP) 0
#     (C1 은 정확도 0.744 / 잡음 69 놓침 15)
# TODO
print("\nC3")
Xc = e[특징c].values.astype(float)
yc = (e["EQP_CD"] == "P1-SP2-EXF01").astype(float).values

rng = np.random.RandomState(3)
i1 = rng.permutation(np.where(yc == 1)[0])
i0 = rng.permutation(np.where(yc == 0)[0])
trc = np.concatenate([i1[: int(len(i1) * 0.7)], i0[: int(len(i0) * 0.7)]])
tec = np.concatenate([i1[int(len(i1) * 0.7) :], i0[int(len(i0) * 0.7) :]])

Xc_train = Xc[trc]
Xc_test = Xc[tec]
yc_train = yc[trc]
yc_test = yc[tec]

mu = Xc_train.mean(axis=0)
std = Xc_train.std(axis=0)

Zc_train = (Xc_train - mu) / std
Zc_test = (Xc_test - mu) / std

Zc_tr = torch.tensor(Zc_train, dtype=torch.float32)
Zc_te = torch.tensor(Zc_test, dtype=torch.float32)
Yc_tr = torch.tensor(yc_train, dtype=torch.float32).reshape(-1, 1)
Yc_te = yc_test

torch.manual_seed(0)

model = torch.nn.Linear(5, 1)

with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

loss_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.5)
for epoch in range(2000):
    opt.zero_grad()
    # loss = loss_fn(model(Zc_tr), Yc_tr)
    loss = loss_fn(torch.sigmoid(model(Zc_tr)), Yc_tr)
    loss.backward()
    opt.step()
    if epoch % 400 == 0:
        print(f"epoch : {epoch} / 손실 : {loss.item():.4f}")

with torch.no_grad():
    final_loss = loss_fn(torch.sigmoid(model(Zc_tr)), Yc_tr).item()
    print(f"2000걸음 후 손실 : {loss.item():.4f}")
    p = torch.sigmoid(model(Zc_te)).squeeze(1).numpy()
    conf = 0.5
    verdict = (p >= conf).astype(int)

    TP, FN, FP, TN = four(Yc_te, verdict)

    재현율 = TP / (TP + FN) if TP + FN else 0
    정밀도 = TP / (TP + FP) if TP + FP else 0
    정확도 = np.mean(verdict == Yc_te)

    print(
        f"시험용 임계값({conf})    정확도 {정확도:.3f} / 잡음(TP) {TP}  놓침(FN) {FN}  헛경보(FP) {FP}"
    )

# [C4] 도구가 바뀌어도 누수는 그대로다 — 실습 3 [F1] 의 세션 단위 분할을 파이토치로.
#      e 의 한 행은 파형 2초 조각이고, 같은 계측세션(SESS_ID)의 조각들은 서로 거의 같은 데이터입니다.
#      (실습 3 [F] 머리말 참고. 세션은 8개, 호기당 4번 계측)
#      이번엔 각 호기의 4번째 세션을 통째로 시험용으로 빼고 나머지 6개 세션으로 학습합니다.
#
#      ① 분할 — 그대로 쓰세요 (실습 3 [F1] 과 같은 코드)
#           세션 = e["SESS_ID"].values
#           시험세션 = ["V24-SP1-04", "V24-SP2-04"]
#           tr2 = np.where(~np.isin(세션, 시험세션))[0]     # 학습용 418건
#           te2 = np.where(np.isin(세션, 시험세션))[0]      # 시험용 140건
#      ② 표준화를 Xc[tr2] 의 평균·표준편차로 '다시' 하고 텐서로 바꾸세요. (C1 의 ③ ④ 와 같은 방법, trc → tr2)
#      ③ C1 의 ⑤ 와 같은 방법으로 새 모델을 학습하고, C2 의 ① ② 와 같은 방법으로 임계값 0.5 에서 채점하세요.
#         (시험용 140건: 2호기 70 / 1호기 70)
#
#      ★ 먼저 예상하고 적으세요 ★  파이토치로 바꿨으니 결과가 달라질까요?
#      내 예상:
#
#      ★ 답하세요 ★
#        실습 4 [C1] 에서 "Pipeline 이 막아 주는 누수와 열 누수는 종류가 다르다" 고 했습니다.
#        이 세션 누수는 파이토치가 막아 줄 수 있습니까? 누가 막아야 합니까?
#      내 답:
#
# [나와야 하는 출력]
#     학습용 418 건 / 시험용 140 건
#     epoch    0  손실 0.6931
#     epoch  400  손실 0.5193
#     epoch  800  손실 0.5123
#     epoch 1200  손실 0.5089
#     epoch 1600  손실 0.5068
#     시험용 (임계값 0.5)  정확도 0.500 / 잡음(TP) 0  놓침(FN) 70  헛경보(FP) 0   (실습 3 [F1] 과 같음)
# TODO
print("\nC4")

세션 = e["SESS_ID"].values
시험세션 = ["V24-SP1-04", "V24-SP2-04"]
tr2 = np.where(~np.isin(세션, 시험세션))[0]
te2 = np.where(np.isin(세션, 시험세션))[0]
print(f"학습용 {len(tr2)}건 / 시험용 {len(te2)}건")

Xs_train, Xs_test = Xc[tr2], Xc[te2]
ys_train, ys_test = yc[tr2], yc[te2]

mu_s = Xs_train.mean(axis=0)
sd_s = Xs_train.std(axis=0)

Zs_tr = torch.tensor((Xs_train - mu_s) / sd_s, dtype=torch.float32)
Zs_te = torch.tensor((Xs_test - mu_s) / sd_s, dtype=torch.float32)
Ys_tr = torch.tensor(ys_train, dtype=torch.float32).reshape(-1, 1)

torch.manual_seed(0)
clf_s = torch.nn.Linear(5, 1)
with torch.no_grad():
    clf_s.weight.zero_()
    clf_s.bias.zero_()
bce_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(clf_s.parameters(), lr=0.5)

for epoch in range(2000):
    opt.zero_grad()
    loss = bce_fn(clf_s(Zs_tr), Ys_tr)
    loss.backward()
    opt.step()
    if epoch % 400 == 0:
        print(f"epoch {epoch:4d} 손실 {loss.item():.4f}")

with torch.no_grad():
    p = torch.sigmoid(clf_s(Zs_te)).squeeze(1).numpy()
판정 = (p >= 0.5).astype(int)

TP, FN, FP, TN = four(ys_test, 판정)
정확도 = (TP + TN) / len(ys_test)
print(
    f"시험용 (임계값 0.5)  정확도 {정확도:.3f} / 잡음(TP) {TP}  놓침(FN) {FN}  헛경보(FP) {FP}"
)

# =====================================================================
# D. 미니배치 — 파이토치를 쓰는 두 번째 이유                   (05 §6)
# =====================================================================
# [D1] B1 (압연 데이터 Ztr, Ytr / Linear(4,1) / MSELoss / SGD lr=0.1 / 0 에서 출발) 을 미니배치로 바꾸세요.
#      바뀌는 건 '데이터를 32건씩 잘라서 한 걸음씩' 걷는다는 것 하나입니다. 40 에폭.
#
#      ① 순서: torch.manual_seed(0) → 모델 → 0 으로 → 옵티마이저 → 로더 → 루프
#           from torch.utils.data import TensorDataset, DataLoader
#           로더 = DataLoader(TensorDataset(Ztr, Ytr), batch_size=32, shuffle=True)    (05 §6)
#      ② 루프 모양 — 바깥 for 는 에폭, 안쪽 for 는 배치. 네 줄은 안쪽에 들어갑니다.
#           걸음 = 0
#           for epoch in range(40):
#               for zb, yb in 로더:          # zb = 이번 32건의 입력, yb = 그 정답
#                   (네 줄 — 손실은 loss_fn(model(zb), yb))
#                   걸음 += 1
#               if epoch in (0, 4, 9, 19, 39):
#                   (여기서 찍기)
#      ③ 찍을 것 (epoch 0, 4, 9, 19, 39 마다, no_grad 안에서)
#           걸음 / 학습용 '전체' 손실 = loss_fn(model(Ztr), Ytr).item() / 학습용 R2 / 시험용 R2
#           ※ 루프 안의 loss 는 마지막 묶음(15건) 하나의 손실이라 비교용으로 못 씁니다. 전체로 다시 재세요.
#      ④ 그리고 len(로더) (= 한 에폭의 걸음 수) 와 마지막 묶음의 건수를 찍으세요.
#
#      다 되면 ① ~ ② 를 for 씨앗 in [0, 1, 2, 3, 4]: 로 감싸서 torch.manual_seed(씨앗) 만 바꿔 다섯 번 학습하고,
#      40 에폭 뒤 학습용·시험용 R2 를 다섯 줄로 찍은 뒤, 시험용의 최소·최대·평균을 찍으세요.
#
#      ★ 답하세요 ★
#        (1) 40 에폭 동안 몇 걸음을 걸었습니까? B1 (500 걸음) 과 비슷하게 맞춘 겁니다.
#            그런데 epoch 0 (13 걸음) 에서 이미 학습용 R2 가 얼마입니까?
#            B2 의 5 걸음, 50 걸음과 비교하세요. 왜 미니배치가 '같은 에폭 수로 더 많이 내려간다' 고 합니까?
#        (2) B1 은 마지막 손실이 매 에폭 조금씩 '내려가기만' 했습니다.
#            여기는 epoch 9 → 19 에서 손실이 오히려 올라갑니다. 왜 그렇습니까?
#            힌트: 한 걸음을 정할 때 399건 전부를 봅니까, 32건만 봅니까?
#        (3) 씨앗만 바꿨는데 시험용 R2 가 얼마에서 얼마까지 흔들립니까?
#            B1 은 씨앗을 바꾸면 흔들립니까? (0 에서 출발하고 전체로 걸으니까요)
#        (4) 이 데이터에 미니배치가 필요했습니까? 그럼 언제 필요합니까?
#      내 답: (1)
#             (2)
#             (3)
#             (4)
#
# [나와야 하는 출력]
#     한 에폭에 13 걸음 (399건 ÷ 32 = 12 묶음 + 마지막 15건 한 묶음)
#     epoch  0  걸음  13  학습용 손실 432.840  학습용 R² 0.7164  시험용 R² 0.6323
#     epoch  4  걸음  65  학습용 손실 319.243  학습용 R² 0.7909  시험용 R² 0.6226
#     epoch  9  걸음 130  학습용 손실 315.837  학습용 R² 0.7931  시험용 R² 0.5985
#     epoch 19  걸음 260  학습용 손실 320.720  학습용 R² 0.7899  시험용 R² 0.5686
#     epoch 39  걸음 520  학습용 손실 313.964  학습용 R² 0.7943  시험용 R² 0.5938
#
#     씨앗   학습용 R²   시험용 R²
#        0    0.7943     0.5938
#        1    0.7887     0.6474
#        2    0.7933     0.6157
#        3    0.7976     0.6197
#        4    0.7725     0.4902
#     시험용 최소 0.4902 / 최대 0.6474 / 평균 0.5934
#     (B1 전체 배치는 학습용 손실 307.993 에서 멈춤. 미니배치는 그 근처를 맴돕니다)
# TODO
print("\nD1")
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
로더 = DataLoader(TensorDataset(Ztr, Ytr), batch_size=32, shuffle=True)
model = torch.nn.Linear(4, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

걸음 = 0
for epoch in range(40):
    for zb, yb in 로더:
        opt.zero_grad()
        loss = loss_fn(model(zb), yb)
        loss.backward()
        opt.step()
        걸음 += 1
    if epoch in (0, 4, 9, 19, 39):
        with torch.no_grad():
            print(
                f"epoch {epoch} 걸음 {걸음} 학습용 손실 {loss_fn(model(Ztr), Ytr).item():.3f} 학습용 R2 {R2(Ytr, model(Ztr)):.4f} 시험용 R2 {R2(Yte, model(Zte)):.4f}"
            )

print("\n씨앗   학습용 R2   시험용 R2")
시험점수 = []
for 씨앗 in [0, 1, 2, 3, 4]:
    torch.manual_seed(씨앗)
    model = torch.nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    로더 = DataLoader(TensorDataset(Ztr, Ytr), batch_size=32, shuffle=True)

    for epoch in range(40):
        for zb, yb in 로더:
            opt.zero_grad()
            loss = loss_fn(model(zb), yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        r_tr, r_te = R2(Ytr, model(Ztr)), R2(Yte, model(Zte))
    시험점수.append(r_te)
    print(f"{씨앗:4d}    {r_tr:.4f}     {r_te:.4f}")

print(
    f"시험용 최소 {min(시험점수):.4f} / 최대 {max(시험점수):.4f} / 평균 {np.mean(시험점수):.4f}"
)

# [D2] 05 §6 은 Adam 을 쓰면서 정답 y 까지 표준화했습니다. 왜 그랬는지 직접 확인합니다.
#      D1 과 같은 압연 데이터, 같은 미니배치 루프입니다. 바뀌는 것:
#        · 옵티마이저: torch.optim.Adam(model.parameters(), lr=0.01)   (05 §6)
#        · 0 으로 바꾸는 두 줄을 빼고 기본 출발
#        · 100 에폭
#      순서: torch.manual_seed(0) → 모델 → 옵티마이저 → 로더 → 루프
#
#      정답만 바꿔서 두 번 돌리세요.
#        ① 정답을 그대로:  로더에 TensorDataset(Ztr, Ytr), 채점도 Ytr, Yte 로
#        ② 정답을 표준화해서 (05 §6 과 같은 코드):
#             y_m, y_s = Ytr.mean(), Ytr.std()              # 학습용 정답의 평균·표준편차
#             Ytr_s, Yte_s = (Ytr - y_m) / y_s, (Yte - y_m) / y_s
#           로더에 TensorDataset(Ztr, Ytr_s), 채점도 Ytr_s, Yte_s 로
#      epoch 0, 9, 49, 99 마다 학습용 R2 / 시험용 R2 / 절편(model.bias.item())을 찍으세요.
#      (② 를 표준화된 눈금으로 채점해도 되는 이유는 아래 ※)
#
#      ★ 답하세요 ★
#        (1) ① 의 절편이 100 에폭 동안 얼마까지 갔습니까? 가야 할 곳은 어디입니까? (정답 평균 112.64)
#        (2) Adam 은 '손잡이마다 보폭을 알아서 조절' 합니다. 대신 한 걸음의 크기가 대략 lr 근처입니다.
#            lr=0.01 로 112 까지 가려면 몇 걸음이 필요합니까? 100 에폭 = 1300 걸음이면 됩니까?
#        (3) SGD 는 같은 문제에서 왜 괜찮았습니까?
#            힌트: SGD 의 한 걸음 = lr × 기울기. A3 에서 본 b 방향 기울기는 몇이었죠?
#        (4) 그럼 실무에서 정답의 눈금이 클 때 어떻게 하는 게 안전합니까? 한 줄로 적으세요.
#      내 답: (1)
#             (2)
#             (3)
#             (4)
#
# [나와야 하는 출력]
#     ① 정답 그대로
#        epoch  0  학습용 R² -8.2924  시험용 R² -8.8119  절편 -0.063
#        epoch  9  학습용 R² -8.0345  시험용 R² -8.5821  절편 1.102
#        epoch 49  학습용 R² -7.0296  시험용 R² -7.6911  절편 6.216
#        epoch 99  학습용 R² -6.0006  시험용 R² -6.7552  절편 12.472
#     ② 정답 표준화 (학습용 평균 112.640, 표준편차 39.119)
#        epoch  0  학습용 R² 0.6430  시험용 R² 0.5032  절편 -0.091
#        epoch  9  학습용 R² 0.7940  시험용 R² 0.6317  절편 0.008
#        epoch 49  학습용 R² 0.7972  시험용 R² 0.6090  절편 0.009
#        epoch 99  학습용 R² 0.7980  시험용 R² 0.6156  절편 -0.002
#     ※ ② 의 예측을 원래 눈금(× 39.119 + 112.640)으로 되돌려 원래 정답으로 채점해도 시험용 R² 0.6156 으로 같습니다.
#       R² 는 '평균만 말하는 것보다 얼마나 나은가' 라서 눈금을 바꿔도 변하지 않습니다.
#     ※ 05 §6 은 표준편차를 Ytr.std() (unbiased) 로 구했습니다. 위 숫자도 그 방식입니다.
# TODO
print("\nD2")

# 정답을 그대로
print("정답을 그대로")
torch.manual_seed(0)
model = torch.nn.Linear(4, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
로더 = DataLoader(TensorDataset(Ztr, Ytr), batch_size=32, shuffle=True)
loss_fn = torch.nn.MSELoss()

걸음 = 0
for epoch in range(100):
    for zb, yb in 로더:
        opt.zero_grad()
        loss = loss_fn(model(zb), yb)
        loss.backward()
        opt.step()
        걸음 += 1
    if epoch in (0, 9, 49, 99):
        with torch.no_grad():
            print(
                f"epoch {epoch} 학습용 R2 {R2(Ytr, model(Ztr)):.4f} 시험용 R2 {R2(Yte, model(Zte)):.4f} 절편 {model.bias.item():.3f}"
            )

# 정답을 표준화해서
print("\n정답을 표준화해서")
torch.manual_seed(0)
model = torch.nn.Linear(4, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

y_m, y_s = Ytr.mean(), Ytr.std()
Ytr_s, Yte_s = (Ytr - y_m) / y_s, (Yte - y_m) / y_s

로더 = DataLoader(TensorDataset(Ztr, Ytr_s), batch_size=32, shuffle=True)
loss_fn = torch.nn.MSELoss()

걸음 = 0
for epoch in range(100):
    for zb, yb in 로더:
        opt.zero_grad()
        loss = loss_fn(model(zb), yb)
        loss.backward()
        opt.step()
        걸음 += 1
    if epoch in (0, 9, 49, 99):
        with torch.no_grad():
            print(
                f"epoch {epoch} 학습용 R2 {R2(Ytr_s, model(Ztr)):.4f} 시험용 R2 {R2(Yte_s, model(Zte)):.4f} 절편 {model.bias.item():.3f}"
            )

# =====================================================================
# E. 마무리 — 같은 일을 세 번 했다
# =====================================================================
# 실습 2(손) → 실습 4(사이킷런) → 오늘(파이토치). 같은 데이터, 같은 질문이었습니다.
#
#   1) 아래 표를 채우세요. 실습 2·4·5 의 코드에서 '그 일을 하는 줄' 을 찾아 적으면 됩니다.
#
#        하는 일          실습 2 (손)            실습 4 (사이킷런)        실습 5 (파이토치)
#        기울기 구하기     기울기_밟아보기()
#        한 걸음           w = w - lr * gw
#        손실 정하기       손실()
#        나누기·표준화     permutation, mu/sd
#        시험용 R²         0.6080                 0.6070                   (B1)        (B2 의 2000)
#
#   2) 오늘 '일부러 틀려 보기' 가 세 번 있었습니다 (B3, C3, D2-①).
#      셋 다 에러가 나지 않았습니다. 셋을 미리 알아챌 수 있는 습관을 하나씩 적으세요.
#      · B3 :
#      · C3 :
#      · D2 :
#
#   3) 05 §8 은 "이 과정의 표 데이터는 전부 사이킷런" 이라고 했습니다.
#      오늘 파이토치로 해 보니 실제로 그랬습니까? 그럼 파이토치는 언제 꺼내겠습니까? (한 줄)
#
# =====================================================================
