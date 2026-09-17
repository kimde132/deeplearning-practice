# 실습 환경 만들기

수업 전에 미리 해 주세요. 라이브러리 설치가 5~10분 걸립니다.

**`deeplearning-practice` 폴더 열기 → venv 만들기 → 라이브러리 설치** 순서입니다.
venv 는 이 폴더 안에 만들고, 수업 때 칠 코드도 여기에 씁니다.

---

## 1. deeplearning-practice 폴더 열기

VS Code → **파일 → 폴더 열기** → **`deeplearning-practice`** 폴더 선택
("이 폴더의 작성자를 신뢰합니까?" 가 뜨면 **신뢰함**)

### 터미널 열기

VS Code 위쪽 메뉴 **터미널 → 새 터미널** (단축키 `Ctrl + ~`, 맥은 `Control + ~`)

아래에 검은 창이 뜹니다. **이미 `deeplearning-practice` 안에 들어와 있습니다.**
앞으로 나오는 명령은 전부 여기에 한 줄씩 치시면 됩니다.

> venv 도 여기에 만들고, 수업 때 칠 코드도 여기에 씁니다.
> **폴더를 옮기거나 이름을 바꾸면 venv 가 깨집니다.** 그대로 두세요.

---

## 2. 파이썬 버전 확인

```
python --version
```

맥에서 안 되면 `python3 --version`.

**3.12 이상**이어야 합니다. 3.9 나 3.11 이 나오면 https://www.python.org/downloads/ 에서
3.14 를 설치하세요. (윈도우는 설치 첫 화면에서 `Add python.exe to PATH` 체크 필수,
설치 후 VS Code 껐다 켜기)

---

## 3. venv 만들고 켜기

**윈도우**
```
python -m venv .venv
```
```
.venv\Scripts\activate
```

**맥**
```
python3 -m venv .venv
```
```
source .venv/bin/activate
```

터미널 맨 앞에 **`(.venv)`** 가 붙어야 합니다.

```
(.venv) C:\...\deeplearning-practice>      <- 켜진 상태
C:\...\deeplearning-practice>              <- 꺼진 상태. 이러면 안 됩니다
```

**만들기는 한 번, 켜기는 터미널 열 때마다** 입니다.

VS Code 에서 `Ctrl+Shift+P` (맥 `Cmd+Shift+P`) → `Python: Select Interpreter` →
`.venv` 선택을 한 번 해 두면 다음부터 자동으로 켜집니다.

---

## 4. 라이브러리 설치

`(.venv)` 확인하고, 아래를 **한 줄로** 복사해서 붙여넣으세요.

```
pip install pandas==3.0.5 numpy==2.5.3 scikit-learn==1.9.0 torch==2.14.0 matplotlib==3.11.1 joblib==1.6.0
```

**5~10분 걸립니다.** torch 가 수백 MB 라서 그렇습니다.
노란 `WARNING` 이나 맨 끝에 뜨는 `[notice] A new release of pip is available` 은
무시하셔도 됩니다. 빨간 `ERROR` 만 아니면 됩니다.

| | 버전 | 하는 일 |
|---|---|---|
| pandas | 3.0.5 | 표 다루기 |
| numpy | 2.5.3 | 숫자 계산 |
| scikit-learn | 1.9.0 | 머신러닝 |
| torch | 2.14.0 | 딥러닝 (파이토치) |
| matplotlib | 3.11.1 | 그래프 |
| joblib | 1.6.0 | 모델 저장 |

**버전(`==` 뒤 숫자)을 빼고 치지 마세요.** 그러면 최신 버전이 깔려서 사람마다 결과가 달라집니다.

---

## 5. 확인

아래를 **한 줄로** 복사해서 붙여넣으세요.

```
python -c "import sys,pandas,numpy,sklearn,torch,matplotlib,joblib as j;print(sys.version.split()[0],'|',pandas.__version__,numpy.__version__,sklearn.__version__,torch.__version__,matplotlib.__version__,j.__version__)"
```

이렇게 나오면 끝입니다.

```
3.14.6 | 3.0.5 2.5.3 1.9.0 2.14.0 3.11.1 1.6.0
```

- 맨 앞 파이썬은 **3.12 이상**이면 뒷자리가 달라도 됩니다
- 나머지 여섯 개는 **위와 똑같아야** 합니다

`ModuleNotFoundError` 가 나오면 그 라이브러리가 안 깔린 겁니다. 4번을 다시 하세요.

여기까지 되면 준비 끝입니다.
수업 때는 이 폴더 안에 `.py` 파일을 만들어 코드를 쓰게 됩니다.
(VS Code 왼쪽 파일 목록에서 마우스 오른쪽 → 새 파일)

---

## venv 가 뭔가

**이 수업 도구만 담아 두는 상자**입니다.

라이브러리를 그냥 설치하면 컴퓨터 전체에 깔려서, 다른 파이썬 프로그램과 버전이 충돌합니다.
그래서 폴더 하나(`.venv`)를 만들어 거기에만 넣습니다.
컴퓨터의 다른 파이썬은 영향을 안 받고, 나중에 그 폴더만 지우면 깨끗이 정리됩니다.

| | 명령 | 언제 |
|---|---|---|
| 만들기 | `python -m venv .venv` | 딱 한 번 |
| 켜기 | `.venv\Scripts\activate` (윈도우)<br>`source .venv/bin/activate` (맥) | 터미널 열 때마다 |
| 도구 넣기 | 4번의 `pip install ...` | 딱 한 번 |
| 끄기 | `deactivate` | 안 해도 됨 |
| 버리기 | `.venv` 폴더 삭제 | 꼬였을 때 |

꼬이면 그냥 지우고 3번부터 다시 하세요. 컴퓨터에 아무 영향 없습니다.

```
rm -rf .venv                  (맥)
rmdir /s /q .venv             (윈도우)
```

`.venv` 는 점으로 시작해서 숨김 폴더입니다. 탐색기에 안 보이는 게 정상입니다.

---

## 버전을 고정한 이유

`pip install pandas` 로 깔면 그날 최신 버전이 들어갑니다. 사람마다 버전이 달라지면
출력 모양이 다르고, 점수가 달라지고, 에러가 납니다.
수업에서 "여기 0.7425 나오시죠?" 할 때 모두가 같은 숫자를 봐야 합니다.

`==` 는 "정확히 이 버전" 이라는 뜻입니다.

---

## 안 될 때

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | `.venv` 안 켬 | 맨 앞에 `(.venv)` 보이게 하고 다시 |
| `Could not find a version that satisfies` | 파이썬 3.12 미만 | 2번 참고 |
| `python 은 명령어가 아닙니다` (윈도우) | PATH 에 없음 | 재설치 시 PATH 체크, VS Code 새로 열기 |
| `이 시스템에서 스크립트를 실행할 수 없으므로` (윈도우) | PowerShell 보안 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 치고 `Y` |
| `python3` 은 되는데 `python` 이 안 됨 (맥) | 정상 | `.venv` 켜면 `python` 됩니다 |
| 한글 깨짐 (윈도우) | 터미널 인코딩 | `chcp 65001` |
| `pip install` 이 멈춤 | 회사 네트워크 차단 | 휴대폰 핫스팟으로 |
| 뭘 해도 이상함 | venv 꼬임 | `.venv` 지우고 3번부터 다시 |

안 되면 화면 캡처해서 보내 주세요.

---

## 요약

| | 윈도우 | 맥 |
|---|---|---|
| 버전 확인 | `python --version` | `python3 --version` |
| venv 만들기 (1회) | `python -m venv .venv` | `python3 -m venv .venv` |
| venv 켜기 (매번) | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| 라이브러리 (1회) | `pip install pandas==3.0.5 numpy==2.5.3 scikit-learn==1.9.0 torch==2.14.0 matplotlib==3.11.1 joblib==1.6.0` | 좌동 |
