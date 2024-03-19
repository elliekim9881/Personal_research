# Sensor fusion

## 다양한 센서데이터를 통합하여 환경에 대한 이해를 향상시키는 기술. 여러 센서의 데이터를 병합하여 특정 장면, 특정 개체에 대한 개념화를 만드는 프로세스.한번에 여러 센서의 출력을 결합, 추출할 때 센서 퓨전을 통해 더 나은 모델 제공 가능하다.

- 각각의 개별 센서들의 데이터는 강점과 약점을 모두 지니고 있기 때문에 각각 개별 센서데이터의 장점을 활용, 불확실성을 줄이고 정확한 모델을 얻고자 한다.

![Untitled](Personal_research\sensor_fusion\Untitled.png)

## 1980’s

## Sensor fusion의 개념 도입

### 주요 연구 과제

- 서로 다른 유형의 **데이터를 효율적으로 통합**하는 방법에 대한 연구
- 여러 센서에서 수집된 데이터의 시간적 정렬을 맞추는 문제에 대한 연구
- 컴퓨팅 자원 제한, **실시간 데이터 처리**(데이터 처리 속도)에 대한 연구

**[An introduction to multisensor data fusion**  Hall, David L and Llinas, James 1997 IEEE](https://ieeexplore.ieee.org/abstract/document/554205)

- 센서퓨전 기본 개념 및 과제, 응용 분야에서의 중요성 제시
- 표적 인식, 전장 감시(군사), 자율 주행 차량 유도, 제어(자율 주행), 모니터링, 의료 진단 분야로의 응용 제시

## 1990~2000’s

## 알고리즘 도입

### 주요 연구 과제

- 센서 별 다른 시간, 공간, 해상도를 가진 **데이터를 효과적으로 통합**하는 방법에 대한 연구
- 대형 데이터 처리 능력 및 **실시간** 분석에 대한 연구

컴퓨터 기술의 전반적인 발전으로 (하드웨어) 센서 퓨전 기술 본격적인 발전.

다양한 알고리즘과 프레임워크가 도입.

- **Kalman Filter**

**[A New Approach to Linear Filtering and Prediction Problems R. E. Kalman 1960](https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction)**

- [**An introduction to the Kalman filter. Welch, Greg, and Gary Bishop 1995**](https://perso.crans.org/club-krobot/doc/kalman.pdf)
    
    피드백 제어를 통하여 추정.
    
    ![Untitled](Personal_research\sensor_fusion\Untitled1.png)
    
    시간 업데이트 방정식 & 측정 업데이트 방정식
    
    - 시간 업데이트 : 현재 상태, 오차 공분산 추정치를 시간을 기준으로 예측
    - 측정값 업데이트 : 새로운 측정값을 예측치에 통합, 사후 추정치를 얻음 (피드백 담당)
    
    = 새로운 측정값을 반복적으로 통합하고 추정치를 업데이트함으로써 시간 경과에 따른 상태를 추적하고 추정한다
    

노이즈가 포함된 측정값을 바탕으로 시스템의 상태를 연속적으로 예측. 다양한 센서 데이터를 결합하여 위치,  속도 등을 추정하는데 사용= 탐색.(스마트폰, 위성)

예측을 가져와서 업데이트에 따라 수정, 원하는 정확도에 도달할 때까지 반복,(반복적인 계산 필요로함) - 동적 시스템의 현재 상태 추정 목적

- **Extended Kalman Filter, EKF : 선형이 아닌 센서데이터 선형화를 위한 접근방식. (비선형 시스템에 칼만 필터를 적용하기 위함)**

![Untitled](Personal_research\sensor_fusion\Untitled2.png)

미분을 통해 비선형 모델을 선형화(jacobian matrix) > 칼만 필터 계산 

- **Unscented Kalman Filter, UFK : 선형화 과정 생략, 샘플링을 통한 근사화**

![Untitled](Personal_research\sensor_fusion\Untitled3.png)

- **Particle Filter : 비선형 필터, 파티클로 시스템의 확률 분포와 상태를 모두 표현.**

![Untitled](Personal_research\sensor_fusion\Untitled4.png)

## 2000~

## 기술 응용, 확장

### 주요 연구 과제

- **대규모 데이터 처리, 분석**
- ML, DL 활용한 고급 알고리즘 개발 : 자동분류, 패턴인식, 예측 모델링
- 더욱 복잡한 환경에서 **실시간으로 데이터 처리,** 의사 결정

**[Multi-Sensor Fusion in Automated Driving: A Survey Wang, Zhangjing, Yu Wu, and Qingqing Niu 2019 *Ieee Access*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8943388)**

**[Sensor and Sensor Fusion Technology in Autonomous Vehicles: A Review Yeong, De Jong, et al 2021](https://www.mdpi.com/1424-8220/21/6/2140)**
자율 주행 자동차 : 레이더, 카메라, 라이다(LIDAR), 울트라소닉 센서 등 다양한 센서의 데이터를 통합하여 주변 환경을 정확하게 인식하고, 안전한 운전 결정 내릴 수 있도록 함. > 현재 보급된 반자율

**[A survey of data fusion in smart city applications Lau, Billy Pik Lik, et al 2019**  *Information Fusion* 52](https://www.sciencedirect.com/science/article/abs/pii/S1566253519300326)
스마트 시티 구현 :  교통 관리, 안전 모니터링, 에너지 관리(도시 운영 효율 상승)

**[Data fusion and multiple classifier systems for human activity detection and health monitoring: Review and open research directions Nweke, Henry Friday, et al 2019**  *Information Fusion* 46](https://www.sciencedirect.com/science/article/abs/pii/S1566253518304135)

의료 : 웨어러블 장치와 의료 모니터링 장비에서 수집된 다양한 생체 신호를 통합, 건강상태 실시간 모니터링

## Recent

late fusion 

early fusion

slow fusion 

single frame

joint fusion

어떤 퓨전이 어떻게 좋을까

데이터 세트별로 비교 

1. *궁극적 목표 [ 인간의 감지 능력(뇌, 신경, 근육) 모방 & 그 이상]* 
2. *발전 과정에서(시간순) 공통적인 연구 과제 : 데이터의 효율적인 통합과 실시간 데이터 처리*
3. *하드웨어 ( 감지 기술, 컴퓨팅 파워) , 소프트웨어 (디지털 신호 처리)* 

- 🤣
    
    결국 자율주행은 도로 위 모든 자동차가 자율주행화 되었을 때 비로소 적용 될 수 있지 않을까..
    
    통신, 하드웨어, 보급