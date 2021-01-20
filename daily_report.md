# Daily Report

1/4 Mon.
- jetbot 초기 설정 및 부팅  
<img src="https://user-images.githubusercontent.com/47997946/103622166-a30b5c80-4f79-11eb-8190-2c74340478f2.png" width="30%"></img>

- 무선랜 설치  
<img src="https://user-images.githubusercontent.com/47997946/103621726-fd57ed80-4f78-11eb-8f4f-111bc2bfbd92.png" width="30%"></img>
<br/>

1/5 Tue.
- jetbot ssh 연결 - [docker](https://jetbot.org/master/software_setup/docker.html)
- jetbot 동작 확인 (모터, 카메라)
- 문제점 : 배터리가 보드만 연결 시에는 작동하나 모터까지 연결할 경우 전원이 꺼짐.
- jetbot 예제들 공부
    1. 기본 동작 확인 : 바퀴의 움직임 확인
    2. 충돌 회피 (Collision avoidance) : 장애물이 있는 이미지와 없는 이미지 학습으로 장애물 피하기
    3. 물체 따라가기 (Object following) : 훈련된 이미지로 해당 물체 따라가게 함
    4. 길 따라가기 (Line tracking) : 길이 주어졌을 때 갈 방향 학습시켜 길을 따라가게 함
- AWS DeepRacer tutorial 공부 : 트랙을 벗어나지 않으면 reward를 주는 방식으로 강화학습
- jetson community 프로젝트들을 살펴봄 - [link](https://developer.nvidia.com/embedded/community/jetson-projects)
    1. [open-source autocar](https://developer.nvidia.com/embedded/community/jetson-projects#autocar) : 컨트롤러로 jetbot이 움직일 때마다 화면, 이동 방향을 저장하여 학습
    2. 물체 인식 : 얼굴, 관절, 과일 등등
    3. [ROS NAVIBOT](https://developer.nvidia.com/embedded/community/jetson-projects#ros_navbot) : indoor mapping
    
<br/>

1/6 Wed.
- jetson community 프로젝트들을 이어서 살펴봄 - [link](https://developer.nvidia.com/embedded/community/jetson-projects)
    1. 물체 인식
          - [알파벳 sign language 인식](https://developer.nvidia.com/embedded/community/jetson-projects#sign_language_recognition)
          - [자동차 번호판 인식](https://developer.nvidia.com/embedded/community/jetson-projects#license_plate_recog)
          - [제스쳐 설명](https://developer.nvidia.com/embedded/community/jetson-projects#tsm_online) : 손을 인식해 취하고 있는 제스쳐에 대한 인식 및 설명
          - [pothole 인식](https://developer.nvidia.com/embedded/community/jetson-projects#ai_pothole_detector)
          - [TIPPER](https://developer.nvidia.com/embedded/community/jetson-projects#tipper) : CNN을 통해 공이 날라가는 초기 상태 캡쳐 사진들로 훈련시켜 (classification) 타자에게 공이 날라오기 전에 스트라이크 존에 공이 들어오는지 미리 알려줌
          - [AI Thermometer](https://developer.nvidia.com/embedded/community/jetson-projects#ai_thermometer) : 물체 인식해 그 물체의 평균 온도 알려줌
          - [Shoot your shot](https://developer.nvidia.com/embedded/community/jetson-projects#shoot_your_shot) : 다트를 던지는 사람의 관절 등의 자세에 따른 다트판의 점수 위치 예측
          - [잡초 제거 로봇](https://developer.nvidia.com/embedded/community/jetson-projects#nindamani)
          - [홈트레이닝 정확도](https://developer.nvidia.com/embedded/community/jetson-projects#mixpose)
          - [Reading eye for blind](https://developer.nvidia.com/embedded/community/jetson-projects#reading_eye) : 500명의 손글씨 모아 손글씨와 프린트 글씨 모두 읽어줌
          - [Deepclean](https://developer.nvidia.com/embedded/community/jetson-projects#deepclean) : 손이 쓸고간 자리 기록 -> 이 담에 청소할 곳 (코로나 때문에 나온 듯!)
          - [DBSE-monitor](https://developer.nvidia.com/embedded/community/jetson-projects#dbse_monitor) : 졸음운전과 스마트폰 사용 방지, 표정으로 감정 인식하여 노래 선곡, 사각지대 알림
          - [앉은 자세 교정](https://developer.nvidia.com/embedded/community/jetson-projects#posture_corrector) : 앉은 자세 잘못되면 경고
          
    2. 자율주행
          - [outdoor mapping](https://developer.nvidia.com/embedded/community/jetson-projects#panther)
          - [indoor mapping](https://developer.nvidia.com/embedded/community/jetson-projects#orbslam2_bebop2_nano) by 드론
          - Traffic cone 피해 주행하기  
            [1)](https://developer.nvidia.com/embedded/community/jetson-projects#jetbot_traffic_cones) : 딥러닝모델(AlexNet)에 cone을 인식하게 하여 cone으로 길이 막혔을때 피해서 주행하게 함  
            [2)](https://developer.nvidia.com/embedded/community/jetson-projects#3d_traffic_cone_detection) : 3D  
          - [Self driving truck](https://developer.nvidia.com/embedded/community/jetson-projects#induction_charger_autonomous_truck) : jetbot으로 자율주행 구현
          - [Self driving hoverboard](https://developer.nvidia.com/embedded/community/jetson-projects#robaka_2)
          - [jetbot 실제 환경 강화학습](https://developer.nvidia.com/embedded/community/jetson-projects#reinforcement_jetbot)
          - [platooning](https://developer.nvidia.com/embedded/community/jetson-projects#platooning_demonstrator) : Bounding box 크기에 따라 거리 및 속도 조절
          
    3. IoT
        - [SHAIDES](https://developer.nvidia.com/embedded/community/jetson-projects#shaides) : AI와 IoT를 결합하여 앞에 보이는 전자기기를 컨트롤(on/off) 가능
        
    4. 기타
        - [Cheersbot](https://developer.nvidia.com/embedded/community/jetson-projects#cheersbot) : 가속도 인식해 같이 짠해주고, 인공지능 스피커 탑재해 대화 가능
        - [Rescue bot](https://developer.nvidia.com/embedded/community/jetson-projects#self_nav_search_rescue_bot) : 특정 위치로 이동해 물체 인식하면 잡아서 시작 위치로 돌아와 내려놓음 (구조 활동과 비슷)
    
<br/>
    
1/7 Thur.
- 전날 살펴본 예시 중 [jetbot 실제 환경 강화학습](https://developer.nvidia.com/embedded/community/jetson-projects#reinforcement_jetbot)의 설명을 더 살펴본 후 VAE와 SAC로 훈련시켰음을 확인함 -> 추후 더 알아봐야 할 것
- 배터리에 jetson nano 보드와 모터를 동시에 연결했을 때 바로 꺼지는 것을 모터 usb를 보드에 연결하고 보드만 배터리에 연결하여 해결함
- 그래서 basic 구동 성공 후 collision avoidance 실행 중 jupyter notebook이 갑자기 멈추는 현상 발생함 -> 부팅 새로 필요할 듯
- [edwith MOVE37 강의](https://www.edwith.org/move37/joinLectures/25196) 강화학습 공부 진행 중  
    - Ch.1 : 강화학습 개요 (용어, 행동-가치/상태-가치 함수), 벨만 방정식, 마르코프체인, 마르코프 결정 과정 (MDP)  
            - 벨만 방정식:  
            <img src="https://user-images.githubusercontent.com/47997946/103871802-de8c5f00-5110-11eb-88ee-6e5fddb08bd0.png" width="50%"></img>
    - Ch.2 : 동적계획법 (가치반복, 정책반복 알고리즘) -> 최적화  
            - 가치반복 알고리즘: 각 칸의 가치를 수렴할 때까지 계산하여 최적의 방법 찾음  
            - 정책반복 알고리즘: 임의의 정책의 리워드를 바탕으로 정책을 업데이트하며 최적의 정책 찾음  
            
<br/>
    
1/11 Mon.
- 연구주제 설정을 위한 아이디어 탐색 (마인드맵 - 구글 코글)
    - [젯봇](https://coggle.it/diagram/X_ujT44ja19ZhWAI/t/%EC%A0%AF%EB%B4%87-image)
    - [강화학습](https://coggle.it/diagram/X_ulwgfKJimvn5Cg/t/%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5)
- jetbot image를 다운받아 SD카드에 flash하여 부팅 다시 함 -> 데이터 훈련하고 그 훈련시킨 모델을 불러올 때 실행이 멈추는 현상 해결 -> 간단히 collision avoidance 잘 작동함 확인 완료
- [edwith MOVE37 강의](https://www.edwith.org/move37/joinLectures/25196) 강화학습 공부 진행 중  
    - Ch.3 : 모델프리 강화학습 (환경을 알지 못해도 최적의 정책 찾기)  
              > 몬테카를로 방법 (무작위로 정책 시도) + 입실론-그리디 (무작위 선택 시 새로운 탐색과 기존 지식 활용 중 선택)  
              > Q-learning (Q[s][a] 테이블을 만들어 평균값 계산 -> 가치 V(s)) => 최적의 정책 결정

<br/>
    
1/12 Tue.
- 연구주제 설정을 위한 아이디어 탐색 (마인드맵 - 구글 코글) (이어서)
    - 아이디어: 젯봇 2대 활용 - 한 대가 line tracking을 할 때 사람 대신 잘 작동하는지를 다른 한 대가 감시 / 나아가 트랙을 벗어나려고 하면 알리기
- [edwith MOVE37 강의](https://www.edwith.org/move37/joinLectures/25196) 강화학습 공부 진행 중  
    - Ch.4 : 시간차 학습 - 몬테카를로 방법의 경우 무작위 정책이 끝이 나야만 거꾸로 계산해 Q 테이블을 채워 가치 계산함. 하지만 복잡한 문제의 경우 끝까지 하기 어려움. 따라서 동적계획법을 차용해 매 타임 스텝마다 다음 Q값을 샘플링하여 현재의 Q값을 예측하며, 실제 다음 Q값과의 차이(예측 오차)만큼 업데이트함.  
    - Ch.5  
        : Augmented Random Search (ARS) Training
        - 로봇의 동작 학습의 시간을 현저히 줄여줌
        - 무작위 가우시안 노이즈를 만들어 현재 가중치에 노이즈를 더한 값과 뺀 값을 각각 구함
        - 두 개의 각 가중치로 하나의 에피소드에서 테스트 해 보상을 계산함 (각 r+, r-)
        - {alpha * (r+ - r-) * 노이즈} 값만큼 가중치를 업데이트 해줌  
        
        : 칼만 필터  
        - 간접적, 불확실한 측정치로 시스템의 상태를 추정할 때 사용되는 최적의 추정 알고리즘
        - 이전 상태의 측정 값을 바탕으로 예측한 현재 상태 값과 현재 상태 측정 값을 종합하여 현재 상태를 추정함 (측정치가 센서의 불확실성으로 인해 항상 정확한 값을 준다고 보기 어렵기 때문)
        - 자율주행차의 경우, 여러 개의 센서의 정보를 종합해 현재 위치 도출해냄

<br/>
    
1/13 Wed.
- Robotics 강의 영상 사이트 찾음 [link](https://www.theconstructsim.com/)
- 강화학습 시뮬레이터 관련 네이버 DEVIEW2020 영상 시청 - [link](https://tv.naver.com/v/16969158?fbclid=IwAR3DicE9mzKe0BWB-Mw_dKKg0KGDoe3JRzWLkpIgm85Wq6FabgAlrKwK5cM)
- 연구주제 설정을 위한 아이디어 탐색 (마인드맵 - 구글 코글) (이어서)
    - 아이디어: 강화학습을 활용하여 코로나 추이 예측 - state: 진행 현황, action: 예측, reward: 예측 정확도 - [데이터](https://bit.ly/2Uq634A), [참고](https://paperswithcode.com/paper/curating-a-covid-19-data-repository-and)
- [edwith MOVE37 강의](https://www.edwith.org/move37/joinLectures/25196) 강화학습 공부 진행 중  
    - Ch.6  
        : 딥 Q러닝  
        - Q러닝의 Q[s][a] 테이블 대신 신경망(DQN)을 이용함 ((s, a) 쌍이 너무 많을 경우 모두 저장 불가능)
        - 현재 상태에서 액션을 통해 보상 등의 결과를 얻고, 손실 = {Q(s, a) - 보상}^2을 계산
        - SGD로 손실을 최소화 하는 방식으로 신경망(Q의 가중치) 업데이트함 / 수렴할 때까지 반복  
        
        : 듀얼 DQN (DDQN)
        - 신경망에서 Q값을 저장하는 DQN과 달리 가치와 보상을 따로 저장한 후 마지막 레이어에서 Q값으로 합쳐짐
        - DQN보다 안정적이며 성능도 좋음
        - 단 역전파 시 값이 나누어져 있어 문제가 발생하지만, 이를 해결하는 방법이 있음 (* 역전파: 마지막 레이어에서부터 첫번째 레이어까지 출력값의 손실을 바탕으로 가중치를 업데이트)

    - Ch.7  
        : 메타 러닝 
        - learning to learn / AI가 하나 또는 여러 개의 AI를 최적화함  
        
        : 진화 알고리즘 
        - 진화론에서의 개념을 가져온 것으로, 주어진 문제의 여러 솔루션 중 우수한 솔루션들만 남기고 나머지는 제거함. 그리고나서 전이 또는 교배로 변화를 주어 이 과정을 반복함.  
        
        : 신경 진화 
        - 딥러닝의 신경망의 구조, 형태 등의 디테일을 최적화하는 메타러닝 기술로 진화 알고리즘을 사용함  
        
        : 정책 검색 알고리즘 
        - MDP를 해결하기 위한 정책을 직접 학습하는 방법 (ex) 정책 경사  
                * Q러닝은 행동 반복 알고리즘으로 Q[s][a] 테이블을 채움으로써 최고의 가치를 만드는 행동들을 취해 최종 정책 결정함 (off-policy)   
                반면에, 정책 검색 알고리즘은 특정 정책으로부터 변화를 줌으로써 최종 정책 찾음 (on-policy)  
                * 액터크리틱 기법 - 크리틱이 현재 정책의 가치를 추정(정책 평가)하고, 액터는 이 값을 이용해 정책 경사(정책 발전)를 적용함. 따라서 정책 검색 알고리즘(정책 경사 사용함)과 정책 반복 알고리즘(정책 평가 후 발전) 모두에 속함.  

<br/>
    
1/14 Thur.
- 연구주제 설정을 위한 아이디어 탐색 (마인드맵 - 구글 코글) (이어서)
    - 아이디어: 가고 싶은 여행지 입력하면 최적 경로를 제공
    - [edwith MOVE37 강의](https://www.edwith.org/move37/joinLectures/25196) 강화학습 공부 진행 중  
    - Ch.8  
        : 정책 경사
        - model-free & on-policy 방법으로, 정확한 Q값을 계산하지 않고 정책의 기대되는 값보다 보상이 더 커지는 방향으로 정책을 수정해나가는 방식 (정책 검색 알고리즘 중 하나)
        - 정책 신경망 : 각각의 행동을 취하게 될 확률로 이루어짐
        - 정책 경사 계산 : 무작위 정책에서 시작 -> 환경에서 몇가지 행동 추출 -> 보상이 기대한 것보다 크면 해당 행동을 취할 확률을 높이고 작으면 확률을 낮춤 -> SGD로 정책 신경망 업데이트
        * Q러닝과의 큰 차이점은 정책 경사에서는 명시적인 탐색이 필요하지 않다는 점이다.
        - 수렴 속도도 빠르고 고차원의 문제 상황에서 효율적이나, 단점은 global 최저점이 아닌 local 최저점에서 머물러 수렴해버릴 수 있다는 것이다
        
        : 진화 정책 경사 (EPG)
        - 정책 경사의 메타러닝 방식으로, 단순히 정책을 손실 함수의 값에 따라 업데이트하는 내부 루프와 더불어 정책의 보상 값에 따라 그 보상 값이 더 커질 수 있도록 손실함수의 파라미터를 수정하는 외부 루프를 추가한 방식
        - 내부 루프는 외부 루프가 제안하는 변화된 손실 함수에 정책을 최적화하기 때문에, 지역 최저점에 머물 수 있는 정책 경사의 단점을 보완함  
        <img src="https://user-images.githubusercontent.com/47997946/104547436-e056a500-5671-11eb-8980-a9f44778946d.png" width="30%"></img>
        
    - Ch.9
        : Actor Critic
        - 현재 환경에서 행동을 결정하는 actor 네트워크과 현재 상태의 가치를 계산하는 critic 네트워크로 나뉘어 최적화함
        - DDPG (Deep Deterministic Policy Gradient) : Q러닝이 action이 연속적일 경우 사용 가능 / model-free & off-policy
        - A2C (Advantage Actor Critic) : policy loss와 value loss를 각각 actor와 critic가 줄여나가도록 함. 즉 actor critic 방법에 actor가 정책 경사를 따르도록 한 것. / 정책 경사의 단점 보완함
        - A3C (Asychronous A2C) : 한 정책마다 학습 데이터를 독립적이고 동일한 (i.i.d) 분포를 갖게 하기 위해 여러 환경에서 parallel하게 train시킴 (단 더 큰 버퍼 필요)
        - Continuous AC : action이 discrete하지 않고 continuous할 경우 (실제 상황에서는 continuous한 경우가 대다수) 3가지 부분을 바꿔주면 된다.
            - 우선 정책에서 각 행동을 취할 확률을 softmax 함수로 이산적으로 구하는 대신에 평균 mu와 표준편차 sigma 값을 구하도록하여 연속적인 행동을 취학 확률은 mu와 sigma의 정규 분포를 따르도록 한다. (sigma가 작아질수록 탐색을 더 적게 하는 것)
            - policy loss와 entropy bonus 부분도 정책이 연속적이므로 마찬가지로 정규분포를 따르도록 한다.
        - Proximal Policy Optimization (PPO) : AC을 업그레이드 한 것으로 부드럽게 점진적인 경사도 업테이트를 갖도록 한 것
            - AC의 문제점 : 하이퍼파라미터 조정에 학습이 매우 민감함, 과도한 정책 업데이트가 야기하는 아웃라이어 데이터가 학습 프로세스 전체를 망가뜨릴 수 있음
            - PPO가 업그레이드한 것 : 
                1. Generalized Advantage Estimation (GAE)으로 보상의 손실과 분산을 줄여 학습을 보다 부드럽고 안정적으로 진행
                2. Surrogate Policy Loss 이용 (이전 확률과 새 확률의 비율을 활용해 policy loss 조절하는 느낌)
                3. Mini-batch Updates

<br/>
    
1/15 Fri.
- 연구주제 설정을 위한 아이디어 탐색 (마인드맵 - 구글 코글) (이어서)
    - 추가 아이디어: 젯봇으로 들어오는 쓰레기 분리수거
- 아이디어 정리
    1. 자율주행이 잘 되고 있는지 감시하는 젯봇
        - 자동차의 안전장치 또는 통제 및 제어 시스템으로 활용 가능할 것 같음
        - line tracking을 우선 구현 필요
    2. (정책에 따른) 코로나 추이 예측
        - 데이터 쉽게 구할 수 있음
        - 주가 예측 등을 참고하면 좋을 듯
        - 그러나 환경설정이 중요함
    3. 여행 경로 추천
        - 데이터 구하기 어려움 -> 이 아이디어는 제외
    4. 젯봇으로 분리 수거
        - 시뮬레이터(nvidia issac sim) 학습 필요
        - 사실 인공지능(이미지 분류)을 이용한 분리수거 이미 꽤 나와있음
        - 구현에 성공하면 택배 분류나 헌 옷 분류(의류 수거)에 활용할 수 있을 것 같음
- 계획 : 위 아이디어 중 2, 4번 중심으로 병행 (1번의 경우는 직접 젯봇을 구동시켜보면서 환경이나 상황을 살펴볼 예정)
    - 아이디어별 계획
        - 2번 : actor critic code 공부 및 주가 예측 등에서 어떻게 쓰이는지 확인 -> 환경 설정 (변수 설정 등) -> 데이터 정리 -> actor critic 개선 및 모델 확정 -> 예측 (역강화학습?)
        - 4번 : 시뮬레이터 사용법 공부 -> 젯봇에 맞게 환경 설정 -> 강화학습 -> 젯봇에 적용
    
    - 주차별 계획
        - 3주차 : 2번) actor critic 공부 및 주가 예측 등에서 어떻게 쓰이는지 확인 / 4번) 시뮬레이터 익히기
        - 4주차 : 2번) 코로나 환경 설정 & 데이터 정리 / 4번) 젯봇 설정해서 강화학습 전 테스트
        - 5주차 : 2번) actor critic code 완성시켜 테스트 / 4번) 강화학습
        - 6주차 : 2번) 강화학습 & 문제점 파악 / 4번) 문제점 파악 및 분석 (* 설 연휴)
        - 이후, 개선 및 발표 준비
        
                
<br/>
    
1/18 Mon.
- Ubuntu 환경 초기 설정  
    - Anaconda 설치, Pycharm 설치, 한영 키 변환 등 초기 설정
    - 『강화학습 입문 파이썬 예제와 함께하는』의 내용에 따라 환경 구성 - [github 링크](https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python), [책 링크](http://www.yes24.com/Product/Goods/77254784)
        - OpenAI Gym과 여러 라이브러리를 포함한 universe 환경 설치
            > sudo apt-get update  
            > sudo apt-get install golang libcupti-dev libjpeg-turbo8-dev make tmux htop chromium-browser git cmake zlib1g-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig  
            > conda install pip six libgcc swig  
            > conda install opencv  
            > pip install gym==0.7.0  
            > conda install -c conda-forge tensorflow  

        - 기본 시뮬레이션 동작
        
- 『강화학습 입문 파이썬 예제와 함께하는』 공부
    - ch2
        : Gym을 활용한 강화학습 순서
        - env 생성 -> episode마다 reset -> step마다 env render, action_space에서 행동 선택, 기록   
        - env.step(action) -> observation: 환경의 관찰, reward: 이전 행동으로부터 얻은 보상, done: 에피소드 완료 여부, info: 디버깅에 유용한 정보
        
        : tensorflow 기초 지식
        - 변수: tf.variable 이후 tf.global_variables_initializer() 함수로 할당
        - 상수: tf.constant
        - 플레이스홀더: type와 shape를 정의하고 값은 할당 못 함
        - 계산 그래프(ex. tf.multiply) 이후 tf.session과 sess.run을 해야 실행 
        - tensorboard로 확인
  
- 주간 계획
    - 『강화학습 입문 파이썬 예제와 함께하는』 ch 10 A3C, ch 11 DDPG 공부
    - 『수학으로 풀어보는 강화학습 원리와 알고리즘』 4장 A2C, 5장 A3C, 6장 PPO, 7장 DDPG 공부
    - 간단하게 state, action, reward를 갖는 강화학습 모델 완성 (custom data로 gym 환경 구성)
    
- Nvidia Isaac Sim 설치
    1. Prepare local requirement : Nvidia driver 설치
    2. Run Omniverse Isaac Sim container on Ubuntu
        - Isaac Sim package 다운로드 후 실행
        - Nucleus Server 설치 : [문제점] user name/pwd 설정 후 sudo docker run 실행되지 않음 
        - add the Omniverse Isaac Sim assets to the Nucleus Server : [문제점] nucleus server 연결 후 Isaac Sim 실행하여 연결이 되는 것까지는 확인 완료. 그러나 수정이 안됨 (새 폴더 만들기 불가) // 위의 문제때문인지는 모르겠음 => [해결] mount를 이용하는 2번째 방법으로 하니 문제 없이 됨
        - (내일 이어서) Isaac Sim 첫 실행

<br/>
    
1/19 Tue.
- 코로나 관련 데이터셋 목록
    - KT빅데이터플랫폼
        > fpopl.csv - 행정동별 유동인구 데이터  
        > adstrd_master.csv - 8자리 행정동 코드 데이터  
        > card.csv - 업종 별 결재금액 데이터  
        > delivery.csv - 배달 호출 정보 데이터  
        > index.csv - 품목 별 소비지수 데이터  
        
    - Data Science for COVID-19 (DS4C) - [description](https://github.com/jihoo-kim/Data-Science-for-COVID-19)
        > 코로나 확진자 데이터
    
    - corona board - [github](https://github.com/jooeungen/coronaboard_kr)
        >  kr_daily.csv - 한국의 코로나 현황 (사망자 수, 확진자 수)  
        >  kr_regional_daily.csv - 지역 별 현황  
            
 - State, Action, Reward 정리
 State: kr_daily.csv(확진자 수), fpopl.csv(행정동별 유동인구 데이터)
 Action: 증가/감소 (비율마다 나누기)
 Reward: 예측 성공 시 1, 실패 시 0
        
- Nvidia Isaac Sim 설치 (이어서)  
    < Local Workstation Deployment >
    1. Prepare local requirement : Nvidia driver 설치 (어제 완료)
    2. Run Omniverse Isaac Sim container natively on Ubuntu
        - Isaac Sim package 다운로드 후 실행 (어제 완료)
        - Nucleus Server 설치 :  
            [문제점] user name/pwd 설정 후 sudo docker run 실행되지 않음  
            [해결] nvidia container toolkit 설치하여 해결 (-> 그런데 실행은 되긴 되는데 제대로 연결되는지 여부 확인 필요)
        - add the Omniverse Isaac Sim assets to the Nucleus Server :  
            [문제점] nucleus server 연결 후 Isaac Sim 실행하여 연결이 되는 것까지는 확인 완료. 그러나 수정이 안됨 (새 폴더 만들기 불가) // 위의 문제때문인지는 모르겠음  
            [해결] mount를 이용하는 2번째 방법으로 하니 문제 없이 됨  
        - Isaac Sim 첫 실행 : Isaac Robotics의 Jetbot Sample 시뮬레이션 Play까지는 됨  
            [문제점] Play 후 키보드 w, a, s, d로 젯봇을 움직여야하는 데 움직이지 않음
            [원인] 위에서 mount를 이용하여 data assets를 가져오는 대신 1번째 방법이었던 package를 이용해 가져와야 함 (mount에 있는 파일은 구 버전) // 하지만 docker 문제를 해결한 시점에서도 여전히 폴더를 만들 수 있는 권한이 없어 불가능  
            [대안] python code 실행 가능한지 확인
    3. Run Omniverse Isaac Sim container on local Ubuntu
        - log-in to NGC : API key 만들어 완료
        - (yet) enable xserver access
        - (yet) run  
        
    < Remote Workstation Deployment >
    1. Container requirements : Nvidia driver, docker, nvidia container toolkit 이미 설치 완료
    2. Access remote Ubuntu workstation
        - Install SSH server
        - Check remote IP address : 129.254.85.131
        - Access remote workstation : ssh jwk@129.254.85.131  
            > ssh <remote_workstation_username>@<remote_workstation_ip_address>
    3. Running headless container
        - Check requirements 및 log-in to NGC 이미 완료
        - (yet) Run  
    4. Omniverse Kit Remote Client  
        - Download Omniverse Kit Remote Client (Linux ver.) 완료
        - Connect to Omniverse Issac Sim  
            [문제점] AWS ?

<br/>
    
1/20 Wed.
- Nvidia Isaac Sim 다시 설치
    - 컴퓨터에서 local로 서버를 만들어 사용하는 방법은 실행 시 에러가 발생하여 AWS 서버를 사용하는 방법으로 변경하기로 하였다.
        - localhost의 사용 권한이 없어 파일을 추가할 수 없고, 그로 인해 예제와 python 코드 모두 동작하지 않는다.
        - AWS 서버 생성하였으나 Nucleus 서버 연결되지 않음.
        - Isaac Sim 구동은 어려워 보임.  
- custom data로 gym 환경 구성 - [참고 링크](https://www.kaggle.com/hwsiew/custom-gym-environment-for-reinforcement-learning)
    - 코로나 추이 예측에 앞서, 간단한 list data로 env를 구성하고 test해보고 있다.
    - gym environment 구성 방법에 대해 공부하고 있다. - [링크](https://gym.openai.com/envs/#classic_control)
        - 함수
            > init : state, action, observation space 정의  
            <img src="https://user-images.githubusercontent.com/59794238/105142921-7ee57900-5b3e-11eb-8f53-cb2fb4b96b95.png" width="30%"></img>  
            > step : 한 단계 동안의 과정 (한 action을 선택하였을 때의 결과, reward return)  
            > render : 환경의 동작 과정 표시  
        - 변수
            > spaces.Discrete : 이산적 변수  
            > spaces.Box : 연속 변수  
    - 역강화학습으로 가치 함수를 알아낸 후, 이 함수를 토대로 예측하는 것이 필요해보인다.
    
 - ROS를 사용하여 jetbot simulation 하는 방법 - [link](https://www.google.com/search?channel=fs&client=ubuntu&q=jetbot+ros+reinforcement+learning)
