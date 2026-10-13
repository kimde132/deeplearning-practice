# =====================================================================
#  03. 분류 — "이 설비, 고장이야 아니야?" 를 맞히기 (로지스틱 회귀)
#  실행: python 03_분류_로지스틱회귀_실무네이밍.py   (수업코드 폴더에서)
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터"
)  # data/수업용데이터/
df = pd.read_csv(os.path.join(DATA_DIR, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")


# =====================================================================
# 1. 문제 정의 — 분류란 무엇인가, 그리고 이 데이터의 함정
# =====================================================================
feature_names = ["공기온도", "회전수", "토크", "공구마모"]
X = df[feature_names].values.astype(float)  # 입력은 02 와 똑같은 센서 4개
y = df["고장여부"].values.astype(
    float
)  # 정답만 바뀜: 고장 1.0 / 정상 0.0 (계산용으로 실수형)

print(
    "[1] 고장 개수:", int(y.sum()), "/ 전체", len(y), f"→ 고장 비율 {y.mean():.1%}"
)  # {:.1%} = 백분율로 소수 1자리

print(
    "    고장 난 설비의 평균 공구마모:",
    round(df[df["고장여부"] == 1]["공구마모"].mean(), 1),
    "/ 정상 설비:",
    round(df[df["고장여부"] == 0]["공구마모"].mean(), 1),
)
print(
    "    고장 난 설비의 평균 토크    :",
    round(df[df["고장여부"] == 1]["토크"].mean(), 1),
    "/ 정상 설비:",
    round(df[df["고장여부"] == 0]["토크"].mean(), 1),
)


# =====================================================================
# 2. 나누기 + 표준화 — 02 와 똑같이 (단, 고장이 양쪽에 골고루 들어가게)
# =====================================================================
rng = np.random.RandomState(3)  # 무작위 고정 (02 와 같은 이유)
fail_idx = rng.permutation(np.where(y == 1)[0])
normal_idx = rng.permutation(np.where(y == 0)[0])  # 정상 192대 번호를 섞음
train_idx = np.concatenate([fail_idx[:6], normal_idx[:134]])
test_idx = np.concatenate([fail_idx[6:], normal_idx[134:]])  # 시험: 고장 2 + 정상 58 = 60
X_train, X_test, y_train, y_test = (
    X[train_idx],
    X[test_idx],
    y[train_idx],
    y[test_idx],
)  # 번호로 행 골라내기 (02 2번)
print(
    "\n[2] 학습용",
    len(train_idx),
    "대 (고장",
    int(y_train.sum()),
    ") / 시험용",
    len(test_idx),
    "대 (고장",
    int(y_test.sum()),
    ")",
)

x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
Z_train = (X_train - x_mean) / x_std
Z_test = (X_test - x_mean) / x_std


# =====================================================================
# 3. 직선값을 확률로 — sigmoid
# =====================================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print("\n[3] sigmoid 맛보기 — 아무 수나 0~1 로")
for z in [-5, -2, 0, 2, 5]:
    print(f"    sigmoid({z:+d}) = {sigmoid(z):.3f}")  # {:+d} = 부호 붙인 정수


def predict_proba(Z, w, b):
    return sigmoid(
        Z @ w + b
    )  # 설비마다 '고장일 확률' 하나씩. 02 의 predict() 에 sigmoid 만 씌운 것


# =====================================================================
# 4. 확률용 손실 — 맞으면 벌 작게, 확신하고 틀리면 벌 크게
# =====================================================================
print("\n[4] 정답이 '고장'일 때, 내가 말한 고장 확률에 따른 벌점")
for p in [0.99, 0.7, 0.5, 0.1, 0.01]:
    print(f"    p={p:<5} → 벌점 {-np.log(p):.3f}")


def bce_loss(Z, y, w, b):
    p = np.clip(predict_proba(Z, w, b), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# =====================================================================
# 5. 학습 — 경사하강 그대로 (01·02 와 한 글자도 안 다름)
# =====================================================================
h = 0.0001


def numerical_gradient(Z, y, w, b):  # 02 4번와 동일 (편미분 5번 = 그래디언트)
    grad_w = np.zeros(len(w))
    for j in range(len(w)):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h
        w_minus[j] -= h
        grad_w[j] = (bce_loss(Z, y, w_plus, b) - bce_loss(Z, y, w_minus, b)) / (2 * h)
    grad_b = (bce_loss(Z, y, w, b + h) - bce_loss(Z, y, w, b - h)) / (2 * h)
    return grad_w, grad_b


def train(Z, y, lr=0.5, epochs=2000, verbose=True):
    w = np.zeros(Z.shape[1])
    b = 0.0
    for epoch in range(epochs):
        grad_w, grad_b = numerical_gradient(Z, y, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
        if (
            verbose and epoch % 400 == 0
        ):  # 400 바퀴마다 손실 출력. % = 나머지, 400 으로 나눠 떨어질 때만
            print(f"    epoch {epoch:4d}  손실 {bce_loss(Z, y, w, b):.4f}")
    return w, b


print("\n[5] 학습 (손실이 줄어드는 걸 보세요)")
w, b = train(Z_train, y_train)
print("    표준화 눈금 가중치:")
for name, weight in sorted(zip(feature_names, w), key=lambda t: -abs(t[1])):
    print(
        f"      {name:6s} {weight:+.3f}  {'클수록 고장 확률 ↑' if weight > 0 else '클수록 고장 확률 ↓'}"
    )


# =====================================================================
# 6. 확률 → 판정, 그리고 채점
# =====================================================================
p_test = predict_proba(Z_test, w, b)  # 시험용 60대의 고장 확률
y_pred = (p_test >= 0.5).astype(int)
print("\n[6] 시험용 60대 중 확률이 높은 순 8대")
rank_idx = np.argsort(-p_test)
print("    고장확률  판정  실제")
for i in rank_idx[:8]:
    print(f"    {p_test[i]:.3f}     {y_pred[i]}     {int(y_test[i])}")

accuracy = np.mean(
    y_pred == y_test
)  # y_pred == y_test 는 True/False 60개, 그 평균 = True 의 비율
print(f"\n    정확도: {accuracy:.3f}")


# =====================================================================
# 7. 정확도의 함정 — 네 칸으로 나눠 봐야 보인다
# =====================================================================
def confusion_counts(y, y_pred):
    TP = int(((y == 1) & (y_pred == 1)).sum())
    FN = int(((y == 1) & (y_pred == 0)).sum())
    FP = int(((y == 0) & (y_pred == 1)).sum())
    TN = int(((y == 0) & (y_pred == 0)).sum())  # .sum() 으로 True 개수 세기
    return TP, FN, FP, TN


TP, FN, FP, TN = confusion_counts(y_test, y_pred)
print("\n[7] 혼동행렬 (임계값 0.5)")
print(f"    고장 {int(y_test.sum())}대 중  잡음(TP) {TP}  놓침(FN) {FN}")
print(f"    정상 {int((y_test == 0).sum())}대 중 통과(TN) {TN}  헛경보(FP) {FP}")

recall = TP / (TP + FN) if TP + FN else 0
precision = TP / (TP + FP) if TP + FP else 0
print(f"    재현율 {recall:.2f}  정밀도 {precision:.2f}")

lazy_pred = np.zeros_like(y_test, dtype=int)
print(
    f"\n    [비교] 무조건 정상이라 답하면 → 정확도 {np.mean(lazy_pred == y_test):.3f}, 재현율 0.00 (고장 전부 놓침)"
)


# =====================================================================
# 8. 임계값 — 0.5 는 절대 규칙이 아니다
# =====================================================================
print("\n[8] 임계값을 바꾸면 (시험용 60대)")
print("    임계값  잡음  놓침  헛경보  재현율  정밀도")
for threshold in [0.5, 0.3, 0.2, 0.1]:
    y_pred_th = (p_test >= threshold).astype(int)
    TP, FN, FP, TN = confusion_counts(y_test, y_pred_th)
    rec = TP / (TP + FN) if TP + FN else 0
    prec = TP / (TP + FP) if TP + FP else 0
    print(
        f"    {threshold:<6}  {TP:^4}  {FN:^4}  {FP:^6}  {rec:.2f}    {prec:.2f}"
    )  # {:^4} = 4칸 가운데 정렬

# =====================================================================
# 실습 — 임계값을 더 내려 보기
# =====================================================================
# [문제] 8번 표에서 임계값 0.2 일 때 고장 2대를 다 잡고 헛경보가 5건이었습니다.
# 0.15 로 더 내리면 어떻게 될까요? 잡음 / 놓침 / 헛경보를 구하세요.
#
#   힌트: (p_test >= 0.15).astype(int) 로 판정을 만들고, confusion_counts() 함수에 넣으면 됩니다.
