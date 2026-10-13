# =====================================================================
#  02. 센서 여러 개로 예측하기 + 정직하게 채점하기 (train / test)
#  실행: python 02_선형회귀_다변수_train_test_실무네이밍.py   (수업코드 폴더에서)
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터"
)  # data/수업용데이터/
df = pd.read_csv(os.path.join(DATA_DIR, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")


# =====================================================================
# 1. 입력이 여러 개면 X 는 '표' — 행 = 설비 한 대, 열 = 센서 하나
# =====================================================================
feature_names = ["공기온도", "회전수", "토크", "공구마모"]
X = df[feature_names].values.astype(
    float
)  # df[리스트] = 그 열들만 뽑은 표 → .values = numpy 표(200행 × 4열)
y = df["공정온도"].values.astype(float)  # 정답은 여전히 숫자 200개 한 줄
print("[1] X 의 모양 (행, 열):", X.shape, " y 의 모양:", y.shape)
print(
    "    1번 설비의 센서값 4개:", X[0]
)  # X[0] = 0번 행 = 첫 설비의 센서값 4개 (00 미리보기 ⑦)
print("    1번 설비의 정답:", y[0])


# =====================================================================
# 2. 학습용 / 시험용 나누기 — 시험 문제를 미리 보지 않기
# =====================================================================
rng = np.random.RandomState(42)
shuffled_idx = rng.permutation(len(X))  # 0~199 를 무작위로 섞은 순서표. permutation = "섞어라"
n_train = int(len(X) * 0.7)  # 200 × 0.7 = 140.0 → int() 로 소수점 버려 140
train_idx, test_idx = (
    shuffled_idx[:n_train],
    shuffled_idx[n_train:],
)  # 섞은 순서의 앞 140개 번호 = 학습용, 뒤 60개 번호 = 시험용 (00 미리보기 ⑥ 슬라이스)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = (
    y[train_idx],
    y[test_idx],
)  # 정답도 '같은 번호'로 골라야 짝이 안 어긋납니다. 이게 제일 중요.
print("\n[2] 학습용", len(train_idx), "대 / 시험용", len(test_idx), "대")


# =====================================================================
# 3. 표준화 — 열마다 따로, 그리고 학습용의 평균·표준편차만 써서
# =====================================================================
print("\n[3] 열별 원래 크기 (학습용 평균):", X_train.mean(axis=0).round(1))


def mse_loss_raw_scale(X, y, w, b):
    return np.mean((y - (X @ w + b)) ** 2)


print("    표준화 안 하고 20걸음 걸어 보기:")
for lr in [0.1, 0.0000001]:
    w0, b0 = np.zeros(4), 0.0
    for _ in range(20):
        grad_w = np.array(
            [
                (
                    mse_loss_raw_scale(X_train, y_train, w0 + np.eye(4)[j] * 1e-4, b0)
                    - mse_loss_raw_scale(X_train, y_train, w0 - np.eye(4)[j] * 1e-4, b0)
                )
                / 2e-4
                for j in range(4)
            ]
        )
        grad_b = (
            mse_loss_raw_scale(X_train, y_train, w0, b0 + 1e-4)
            - mse_loss_raw_scale(X_train, y_train, w0, b0 - 1e-4)
        ) / 2e-4
        w0, b0 = w0 - lr * grad_w, b0 - lr * grad_b
    print(
        f"      lr={lr:<10} → 손실 {mse_loss_raw_scale(X_train, y_train, w0, b0):.2e}"
    )  # {:.2e} = 지수 표기 (1.18e+39 = 1.18 × 10³⁹)

x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)  # 열별 표준편차 4개 (퍼진 정도)
Z_train = (
    X_train - x_mean
) / x_std  # 열마다 (값 − 그 열 평균) ÷ 그 열 표준편차. numpy 가 열 맞춰 알아서 해 줍니다
Z_test = (X_test - x_mean) / x_std
print(
    "    표준화 후 학습용 열별 평균:", np.abs(Z_train.mean(axis=0)).round(3), "← 전부 0"
)
print("    표준화 후 학습용 열별 퍼짐:", Z_train.std(axis=0).round(3), "← 전부 1")
print(
    "    표준화 후 시험용 열별 평균:",
    Z_test.mean(axis=0).round(3),
    "← 0 이 아님. 학습용 자를 빌려 썼으니 당연",
)


# =====================================================================
# 4. 예측과 학습 — 01 과 똑같고, w 가 숫자 하나에서 '숫자 4개'가 될 뿐
# =====================================================================
def predict(Z, w, b):
    return Z @ w + b


def mse_loss(Z, y, w, b):  # 01 과 완전히 같음: (실제 − 예측)² 의 평균 = MSE
    return np.mean((y - predict(Z, w, b)) ** 2)


h = 0.0001


def numerical_gradient(Z, y, w, b):
    grad_w = np.zeros(len(w))
    for j in range(len(w)):  # j = 0,1,2,3 : j 번째 손잡이만 움직여 본다
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h  # j 번째만 살짝 키우고
        w_minus[j] -= h  # j 번째만 살짝 줄여서
        grad_w[j] = (mse_loss(Z, y, w_plus, b) - mse_loss(Z, y, w_minus, b)) / (
            2 * h
        )  # (오른쪽 손실 − 왼쪽 손실) ÷ 거리
    grad_b = (mse_loss(Z, y, w, b + h) - mse_loss(Z, y, w, b - h)) / (2 * h)  # b 도 한 번
    return grad_w, grad_b


def train(Z, y, lr=0.1, epochs=500):
    w = np.zeros(Z.shape[1])  # Z.shape[1] = 열 수 = 4. 가중치 4개를 0 에서 출발
    b = 0.0
    for _ in range(epochs):  # 500 바퀴 (에폭 500)
        grad_w, grad_b = numerical_gradient(Z, y, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b


w, b = train(Z_train, y_train)
print("\n[4] 학습 완료 — 표준화 눈금의 가중치")
for name, weight in zip(feature_names, w):
    print(
        f"    {name:6s} w = {weight:+.3f}"
    )  # {:+.3f} = 부호 붙여 소수 3자리 (00 미리보기 ①)
print(f"    절편 b = {b:.3f}")


# =====================================================================
# 5. 채점 — 학습용 점수와 시험용 점수를 '나란히'
# =====================================================================
def mse(y, y_pred):  # 손실과 같은 식. 채점용으로 이름만 따로
    return np.mean((y - y_pred) ** 2)


def r2_score(y, y_pred):  # 01 9번의 R². 1 에 가까울수록 좋음, 0 = 평균만 말하는 수준
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)


train_pred = predict(Z_train, w, b)
test_pred = predict(Z_test, w, b)
print("\n[5] 채점")
print(
    f"    학습용(train)  R² {r2_score(y_train, train_pred):.4f}   손실 {mse(y_train, train_pred):.3f}"
)
print(
    f"    시험용(test)   R² {r2_score(y_test, test_pred):.4f}   손실 {mse(y_test, test_pred):.3f}"
)
print("""
    읽는 법 — 두 점수를 '같이' 봐야 합니다:
      둘이 비슷하고 둘 다 좋음        → 건강. 규칙을 배운 것
      학습용만 좋고 시험용이 뚝 떨어짐  → 과적합. 답안지를 외운 것
      둘 다 나쁨                     → 과소적합. 모델이 너무 단순하거나 센서가 부족
    어림 기준: R² 차이가 0.05 안이면 정상, 0.10 넘으면 의심. (법칙은 아니고 경보선)
""")
print("    시험용 앞 6대: 예측 vs 실제")
for pred, actual in zip(test_pred[:6], y_test[:6]):
    print(f"      예측 {pred:7.2f}   실제 {actual:7.2f}   (차이 {actual - pred:+.2f})")


# =====================================================================
# 5-1. 과적합을 눈으로 — 학습용을 딱 5대만 주면?
# =====================================================================
print("[5-1] 학습용을 5대만 써서 학습하면? (손잡이 5개 = 데이터 5대)")
# w5, b5 = train(Z_train[:5], y_train[:5], epochs=2000)
w5, b5 = train(Z_train[:20], y_train[:20], epochs=2000)
print(
    # f"    학습용 5대 R² {r2_score(y_train[:5], predict(Z_train[:5], w5, b5)):.4f}   ← 만점. 5대를 통째로 외웠다"
    f"    학습용 20대 R² {r2_score(y_train[:20], predict(Z_train[:20], w5, b5)):.4f}   ← 만점. 5대를 통째로 외웠다"
)
print(
    # f"    시험용 60대 R² {r2_score(y_test, predict(Z_test, w5, b5)):.4f}   ← 140대로 배웠을 때(0.78)보다 뚝. 이게 과적합"
    f"    시험용 60대 R² {r2_score(y_test, predict(Z_test, w5, b5)):.4f}   ← 140대로 배웠을 때(0.78)보다 뚝. 이게 과적합"
)
print("    가중치도 달라짐:", w5.round(2), "← 140대 때는", w.round(2))


# =====================================================================
# 6. 어느 센서가 중요한가 — 표준화 가중치 크기로 읽는다
# =====================================================================
print("\n[6] 영향력 순위 (가중치 절댓값 큰 순)")
for name, weight in sorted(zip(feature_names, w), key=lambda pair: -abs(pair[1])):
    print(
        f"    {name:6s} {weight:+.3f}   {'높을수록 공정온도 ↑' if weight > 0 else '높을수록 공정온도 ↓'}"
    )  # 한 줄 if (00 미리보기 ⑤)


# =====================================================================
# 6-1. 센서를 하나씩 빼 보면 — 중요도를 눈으로, 그리고 과소적합을 만나는 자리
# =====================================================================
print(
    f"\n[6-1] 센서 하나 빼고 다시 학습 (전부 쓰면 학습용 {r2_score(y_train, train_pred):.4f} / 시험용 {r2_score(y_test, test_pred):.4f})"
)
for drop_idx in range(4):
    keep_cols = [j for j in range(4) if j != drop_idx]
    w_sub, b_sub = train(Z_train[:, keep_cols], y_train)
    r2_train = r2_score(y_train, predict(Z_train[:, keep_cols], w_sub, b_sub))
    r2_test = r2_score(y_test, predict(Z_test[:, keep_cols], w_sub, b_sub))
    print(f"    {feature_names[drop_idx]:4s} 빼면 → 학습용 {r2_train:.4f} / 시험용 {r2_test:.4f}")

# =====================================================================
# 실습 — 데이터를 얼마나 줄이면 외우기 시작하나
# =====================================================================
# [문제] 5-1 에서 5대만 주니 학습용 R2 가 1.0 (완전 암기) 이 나왔습니다.
# 그럼 20대를 주면 어떨까요? 학습용과 시험용 R2 를 둘 다 구해 보세요.
#
# 학습용 R2 = 0.8574
# 시험용 R2 = 0.7736
#
#   힌트: train(Z_train[:20], y_train[:20], epochs=2000) 으로 배우고,
# r2_score 와 predict 함수를 그대로 쓰면 됩니다.
