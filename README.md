# 수치해석 기법의 구현과 오차 분석

**기간**

- E1–E2: 2024.03.25 – 2024.04.14
- E3–E4: 2024.05.02 – 2024.05.26

## 개요

Taylor 급수, 근 찾기, 선형 연립방정식, 회귀를 NumPy로 구현하고 오차를 측정한다(역행렬, 조건수,
고유값 등 일부 계산은 `numpy.linalg` 사용). 네 실험은 서로 독립적이다. 각 표는 요약이며, 설정·정의와
전체 결과는 [`docs/experiment-log.md`](docs/experiment-log.md)(이하 로그)에 있다.

| 실험 | 주제 | 내용 |
| --- | --- | --- |
| E1 | Taylor 급수 | `sin(x)`의 `π/6` 전개, 차수 0–10 |
| E2 | 근 찾기 | `(x−1)(x−2)(x−3)`의 근 3, 이분법·고정점 반복·Newton-Raphson |
| E3 | 선형 연립방정식 | 대각이 `δ = 10⁻ⁿ`인 4×4 행렬, 풀이법 5종, 반올림 연구 |
| E4 | 분포 회귀 | 경험적 CDF에 모델 5종 적합, seed 100개 |

## 결과

### E1: Taylor 급수

<table>
<tr>
<td>

| 차수 n | 상대오차 (%) |
| ---: | ---: |
| 0 | 0.8574 |
| 1 | 0.001243 |
| 2 | 3.575e-06 |
| 3 | 2.586e-09 |
| 4 | 4.469e-12 |
| 5 | 2.201e-14 |
| 6–10 | 0 |

</td>
<td><img src="results/e1_taylor/polynomials.png" alt="그림 1" width="640"></td>
</tr>
</table>

### E2: 근 찾기

<table>
<tr>
<td>

| 방법 | 15번째 기록의<br>상대오차 (%) |
| :--- | ---: |
| 이분법 | 0.0005086 |
| 고정점 반복 | 3.134 |
| Newton-Raphson | 0 |

Newton-Raphson은 iteration 7부터 `x = 3.0`이다.

</td>
<td><img src="results/e2_roots/errors.png" alt="그림 2" width="480"></td>
</tr>
</table>

### E3: 선형 연립방정식

<div align="center">

*네 성분 중 가장 큰 상대오차 (%). Gauss-Seidel은 이완 1과 0.9 모두 같은 결과다.*

| n (δ = 10⁻ⁿ) | 조건수 | 단순 가우스 소거 | partial pivoting | Gauss-Seidel |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.94 | 4.049e-14 | 3.522e-14 | nan |
| 8 | 13.42 | 1.488e-06 | 2.220e-14 | nan |
| 15 | 13.42 | 11.18 | 2.220e-14 | nan |
| 16 | 13.42 | nan | 3.331e-14 | nan |

<img src="results/e3_linear/errors.png" alt="그림 3" width="640">

</div>

- 조건수는 모든 `n`에서 13.42–14.94이지만, pivoting 없는 소거의 오차는 `n = 15`에서 11.18%까지
  커진다. pivoting을 쓰면 모든 `n`에서 8.218e-14% 이하다.
- Gauss-Seidel은 반복 행렬의 spectral radius가 모든 `n`에서 6.830e+04 이상이고, 모든 실행이 nan으로
  끝났다.

### E4: 분포 회귀

<table>
<tr>
<td>

| 모델 | `R²` 평균 | 최고 `R²`<br>seed 수 |
| :--- | ---: | ---: |
| 선형 | 0.9643 | 0 / 100 |
| 2차 다항식 | 0.9724 | 6 / 100 |
| 지수 | 0.9774 | 15 / 100 |
| 거듭제곱 | 0.8609 | 0 / 100 |
| **sigmoid** | **0.9934** | **79 / 100** |

</td>
<td><img src="results/e4_regression/r2.png" alt="그림 4" width="552"></td>
</tr>
</table>

- seed 0에서는 지수 모델의 `R²`가 가장 높지만, seed 100개에서는 sigmoid가 79개에서 가장 높다.
