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
        - 2번 : actor critic code 공부 및 주가 예측 등에서 어떻게 쓰이는지 확인 -> 환경 설정 (변수 설정 등) -> 데이터 정리 -> actor critic 개선 및 모델 확정 -> 예측
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
    - 『수학으로 풀어보는 강화학습 원리와 알고리즘』 4장 A2C, 5장 A3C, 6장 PPO, 7장 DDPG 공부 - [github 링크](https://github.com/pasus/Reinforcement-Learning-Book)
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
    
 - ROS를 사용하여 jetbot simulation 하는 방법 - [link](https://www.google.com/search?channel=fs&client=ubuntu&q=jetbot+ros+reinforcement+learning)
 
 - Isaac Sim 대신 [jetbot 실제 환경 강화학습](https://developer.nvidia.com/embedded/community/jetson-projects#reinforcement_jetbot)라도 구동해보기로 함
    - jetbot(192.168.0.22)에 [github](https://github.com/masato-ka/airc-rl-agent)의 installation 진행 중

<br/>
    
1/21 Thur.
- [jetbot 실제 환경 강화학습](https://developer.nvidia.com/embedded/community/jetson-projects#reinforcement_jetbot) 구동  
    [에러] 어제 설치 중 AttributeError: module 'enum' has no attribute 'IntFlag' 에러 발생 하여 설치 실패  
    [해결] sudo pip uninstall -y enum34 통해 해결 후 설치 재시도 중  
    [문제] 여전히 에러 발생.  설치 완료 실패
    
- 강화학습을 이용한 예측 관련 논문 공부
    1. 강화학습을 이용한 주가 예측 - [link](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=227431a61bc0493effe0bdc3ef48d419&outLink=K)  
    : 주가 데이터에 강화학습을 적용하여 장기 보유, 단기 보유 중 장기 보유가 수익이 높다는 것을 알아냈다. 하지만 정해진 데이터 안에서만 동작하고, 이후 데이터를 예측하지는 못한다. 관련 연구로 재귀 강화학습, 심층 강화학습을 활용하여 주가를 예측하는 논문을 소개하여 찾아보았다.

    2. A Multi-objective Deep Reinforcement Learning Approach for Stock Index Future’s Intraday Trading - [link](https://ieeexplore.ieee.org/abstract/document/8283307)  
    : 주식 시장 특성을 Deep nueral network로 파악하고 순환 신경망에 의해 강화 학습 방법을 구현해 거래 결정을 내리는 여러 기법을 혼합한 예측 모델을 사용한다.

    3. Reinforcement Learning via RecurrentConvolutional Neural Networks - [link](https://paperswithcode.com/paper/reinforcement-learning-via-recurrent)  
    : 강화학습에 RNN을 적용하여 MDP model을 효과적으로 학습한다. (좀 더 공부하기)

    4. Using Reinforcement Learning in the Algorithmic Trading Problem - [link](https://paperswithcode.com/paper/using-reinforcement-learning-in-the)  
    : 1번의 논문과 유사한 내용. 정해진 데이터에 대해 여러 강화학습 모델을 사용하여 그들 중 가장 높은 수익률을 얻는 모델을 발견하였다. (주가와 같은 robust data를 다룰 때 LSTM, dropout layer를 추가하거나 cost function을 찾기 위해 NN을 사용하는 것은 성능 향상에 도움이 되지만 보상 함수를 바꾸거나 뉴런 수를 증가시키거나 데이터를 합치는 것은 도움이 되지 않는다.)

    5. Inverse Reinforcement Learning 관련 논문 - [이해에 도움이 된 코드](https://github.com/eliemichel/IRL/blob/master/notebooks/IRL.ipynb)  
    : IRL은 학습 후 그 데이터의 가치 함수를 찾는 것. 데이터의 특징을 알아보는 것으로, 내가 원하는 예측과는 다르다.


<br/>
    
1/22 Fri.
- 코로나 추이 예측 환경 구성
    - up, down 중 어느 것인지 배팅하는 환경 구성을 완료하였고 baseline 라이브러리의 ACKTR을 사용해 학습해보았다. 
    - 결과 (data = [0,9,7,4,3,5])  
    <img src="https://user-images.githubusercontent.com/59794238/105461235-4aa1c200-5cd0-11eb-820f-8fdf52e6ba2c.png" width="100%"></img>
    
- [jetbot 실제 환경 강화학습](https://developer.nvidia.com/embedded/community/jetson-projects#reinforcement_jetbot)의 racer 설치 결국 실패

- 대신, Jetbot ROS로 그냥 실행해보기로 함 - [설치 link](https://github.com/dusty-nv/jetbot_ros) : 설치 및 모터 & 카메라 테스트 완료
    1) ROS 시작 : terminal 열어서 
        > $ roscore  
        
    2) 모터 움직이기 
        - start jetbot_motors node : 새 terminal 열어서  
            > $ rosrun jetbot_ros jetbot_motors.py  
        - test motorcommands : 또 새 terminal 열어서  
            > $ rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "forward"  
            > $ rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "backward"  
            > $ rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "left"  
            > $ rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "right"  
            > $ rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "stop"  
            
    3) 카메라 사용
        - start the jetbot_camera node : 새 terminal 열어서  
            > $ rosrun jetbot_ros jetbot_camera  
            (이러면 카메라로 들어오는 비디오 프레임은 /jetbot_camera/raw 토픽으로 발행됨. 이때 타입은 sensor_msgs::Image이며 BGR8 형식으로 인코딩된 이미지임.)
        - rqt_image_view 도구를 이용하여 이미지 보기  
            > $ rqt_image_view  
            > * rqt_image_view 실행 후 뜬 창에서 왼쪽 위에 빈 상자 클릭해 /jetbot_camera/raw 구독(선택)
            > <img src="https://user-images.githubusercontent.com/47997946/105465315-1af5b880-5cd6-11eb-9d47-bfa098392ac9.png" width="30%"></img>
            
- Isaac Sim 대신 Gazebo(오픈 소스 3D 로봇 시뮬레이터) 설치 완료 및 실행 성공  
    - 버전: gazebo11
    - 실행 방법:
        > $ cd workspace/catkin_ws/src/jetbot_ros/gazebo  
        > $ gazebo  
    - 계획: gazebo를 통해 전에 찾았던 [예시들](https://www.google.com/search?channel=fs&client=ubuntu&q=jetbot+ros+reinforcement+learning) (ROS로 Jetbot simulation) 해보기   

<br/>

1/25 Mon.
- Multi-Agent Reinforcement Learning / Reinforcement learning for Covid-19 자료 정리
    - Multi-Agent Reinforcement Learning
        1. Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning - [link](https://arxiv.org/abs/2006.07169)  
        : 서로 다른 policy의 결과를 모아 학습하는 Shared Experience Actor-Critic(SEAC)를 활용하여 Multi-Agent RL 문제를 해결  
        : Level-Based Foraging, Multi-Robot Warehouse 환경과 그 환경에 대한 SEAC 적용 방법이 있는 코드 공유 - [코드 link](https://github.com/uoe-agents)  
        : local gradient를 공유하고 동시에 비슷한 policy를 학습하여 기존 RL 알고리즘(IAC, SNAC)보다 학습 속도가 빠르고 최대 보상값이 큼  
        : 기존에는 하나의 기기에 대해서만, 혹은 알고리즘을 활용하여 Robotic mobile fullfillment system(RMFS)를 해결하였는데 이를 multi-agent에 대해 해결함    
    - Reinforcement learning for Covid-19
        1. Optimal policy learning for COVID-19 prevention using reinforcement learning - [link](https://journals.sagepub.com/doi/full/10.1177/0165551520959798)  
        : 매 월 코로나에 대비한 정책을 행하였을 때, 그 정책에 따른 reward를 반환하는 가상 환경을 만들어 효율적인 정책을 찾음  
        : 정책은 testing, sanitization, lock down의 3가지로, 정책을 시행할 때 큰 비용이 들기 때문에 좀 더 효율적으로 코로나 대응을 할 수 있는 방법을 찾아봄  
        2. EPIDEMIOPTIM: A TOOLBOX FOR THE OPTIMIZATION OF CONTROL POLICIES IN EPIDEMIOLOGICAL MODELS - [link](https://arxiv.org/pdf/2010.04452.pdf)  
        : 전염병의 모델 (ex. SEIR 모델), cost function, 사용할 최적화 알고리즘 종류, 시행할 방역 대책을 선택하였을 때 강화학습을 통해 최적의 정책을 찾아주는 시뮬레이터 같은 프로그램 제공 - [github](https://github.com/flowersteam/EpidemiOptim)  
        (Example: SEIRAH 모델에서 health & economy cost를 고려하여 COVID-19에 대비할 lockdown 정책 최적화 with DQN)
        3. Reinforcement learning for Covid- 19: Simulation and Optimal Policy - [link](https://towardsdatascience.com/reinforcement-learning-for-covid-19-simulation-and-optimal-policy-b90719820a7f)  
        : 기존의 전염병 수학적 모델(compartment model)은 강화학습을 적용시키기 어려운 부분이 있어 (action을 취했을 때의 환경의 reaction을 확인할 수 없음 등) 새롭게 강화학습 모델 디자인을 하여 시뮬레이션을 해봄 
        4. MONTREAL.AI JOINT TASK FORCE ON PANDEMIC COVID-19 - [link](https://montrealartificialintelligence.com/covid19/)  
        : Covid-19와 관련된 다양한 open code 등 있음  
        
<br/>

1/26 Tue.
- 다뤄볼만한 RMFS 문제
 1. 하나의 agent에 대해 알고리즘 방식과 강화학습 방식을 실행해보고 비교
 2. SEAC에 curriculum learning, Hierarchical Deep Reinforcement Learning 등 다른 방법을 적용하여 단계적으로 학습하게 함
 3. jetbot을 활용해 충돌하지 않고 짐 옮기는 과정 실행 (가상 환경, 실제 환경) - 위아래양옆뿐만 아니라 다양한 방향으로 이동?
 
 - 개발 목표
 1) jetbot simulation 환경에서 RMFS 환경 구성 후 배포  
 2) SEAC 논문 코드를 보고, 학습 과정을 이해하고 학습 이후 동작 모습을 rendering  
 
- Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning - [코드 link](https://github.com/uoe-agents)  
: robotic_warehouse/robotic_warehouse/에서 python3 warehouse.py로 학습 과정 확인  
: robotic_warehouse/tests/의 test_movement.py로 이동 시 거리, 초기값 등 확인  

- DeepSoccer - [github link](https://github.com/kimbring2/DeepSoccer)  
: 강화학습을 할 수 있는 가상 환경 참고  
    - Gazebo error  
        > dusty-nv 레퍼지토리 맨 아래 해보기 
    - 12: build jetson-inference git clone 뒤부터, deepsoccer git clone 한 후 catkin_make, source하기

- EPIDEMIOPTIM([github](https://github.com/flowersteam/EpidemiOptim)) 실행 및 코드 확인
    1. 설치: EpidemiOptim/requirements.txt대로 미설치되어있던 torch와 gym, pymoo 추가로 설치
    2. 실행: EpidemiOptim/readme.md의 예시 실행  
    - Base condition: SEIRAH model (for the epodemiological model (transition function)) - in EpidemiOptim/epidemioptim/environments/gym_envs/epidemic_discrete.py,  
                          health cost function based on death toll,  
                          economy cost function based on GDP loss,  
                          1e6 training steps,  
                          optimize for Pareto front  
    * 1st experiment
        > python train.py --config goal_dqn --expe-name goal_dqn_study --trial_id 0  
        - Experiment condition: goal-dqn (for learning algorithm)  
        
    * 2nd experiment
        > python train.py --config dqn --expe-name dqn_study --trial_id 0  
        - Experiment condition: goal-dqn (for learning algorithm)  
            
    3. 결과 확인: 각각 1st, 2nd experiment 결과  
    - Jupyter Notebook으로 epidemioptim/analysis/Visualization EpidemiOptim.ipynb를 통해 visualization 확인 가능
    <img src="https://user-images.githubusercontent.com/47997946/105819271-50164980-5ffb-11eb-84bf-f5ece6977ad3.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/47997946/105819450-8eac0400-5ffb-11eb-8903-0d15ec78b746.png" width="45%"></img>  
    - 결과 해석을 논문 조사 좀 더 필요  
    
    4. 코드 구조 확인
    
- RL for COVID-19 : EPIDEMIOPTIM의 어느 부분을 바꿀지 고민 필요
    - Agent - Learning model: 현재는 DQN 뿐. -> Actor Critic으로?
    - Agent - Action: 현재는 lockdown on/off 뿐 -> 사회적 거리두기 단계 조절, 유흥업소 폐쇄 등 추가?
    - Environment - Epidemiological model: SEIRAH 모델 -> 다른 모델? / Probabilistic Programming 도입?
    - Environment - Cost function: health(death toll) & economy(GDP loss) cost -> simplify?
  
<br/>

1/27 Wed.
- DeepSoccer
    - Gazebo error
        - Jetbot model 불러오기 (dusty-nv repository)
        > mkdir .gazebo/models , gazebo model editor에서 링크 추가하여 해결
        - Deepsoccer CMake error
        > gazebo-ros 관련 문제가 있어 [issue](https://github.com/kimbring2/DeepSoccer/issues/9)를 올려두었다.
 
 - Isaac Sim
    - 권한 설정 문제 해결
        : 이전에 파일이 만들어지지 않는 문제가 발생하였는데, Devtalk Community에 같은 문제에 대한 [해결방법](https://forums.developer.nvidia.com/t/cannot-create-folder-in-localhost-create-a-folder-and-nothing-happen/160694/6)이 나와 있어 참고하여 해결하였다.
    - 아래 사진과 같이 강화 학습 예제가 잘 동작함을 확인하였다. 내일은 코드를 구체적으로 살펴보고 RMFS 가상 환경을 구현할 방법을 구체화할 예정이다.  
        <img src="https://user-images.githubusercontent.com/59794238/105962401-1c9ef200-60c3-11eb-80aa-4ed54ea0f2a6.png" width="45%"></img>
        
- EPIDEMIOPTIM([github](https://github.com/flowersteam/EpidemiOptim)) 재실행 및 알고리즘 확인
    1. 재설치: 컴퓨터를 옮김에 따라 역시나 미설치되어있던 torch와 gym, pymoo 추가로 설치  
    2. 재실행: 각 알고리즘(learing algorithm) 별로 'first_algorithm_name'의 experiment로 저장  
    3. 알고리즘 분석  
    - DQN: health cost와 economy cost의 비율(1-β:β)을 나타내기 위한 변수 β 값이 주어지면 그에 맞는 손실 값을 최소화하는 솔루션을 찾음 by Actor Critic with deep Q러닝  
    - Goal-DQN: DQN이 β값이 주어졌을 때의 하나로 정해지는 cost function에 최적화한 정책 Π(s)을 찾는 반면, 두 개의 Q-network를 이용하여 β에 따라서도 최적화되는 정책 Π(s, β) 찾음   
    - Goal-DQN-Constraint: Goal-DQN에 각 손실값의 최대값(M_eco, M_health)이라는 constraint를 주어 손실 값이 그 주어진 최대값을 넘으면 -1의 reward 값을 주는 방식으로 솔루션 찾음  
    - NSGA-ii: 경제 분야에서 많이 쓰이는 최적화 방법인 Pareto front 방식 사용. multi-objective algorithm(이라는데 이게 뭔지 잘 모르겠음).  
    - 각 알고리즘의 결과 해석하는 법 알아내 논문의 결과들 모두 확인함.  
    
- RL for COVID-19 : 바꿔볼 옵션들 생각
    1. data를 우선 우리나라 데이터로
    2. lockdown을 우리나라는 시행한 적이 없음. 대신 사회적 거리두기 단계 조정. -> lockdown on/off 대신에 다양한 action 추가  
    3. action 다양화에 따른 epidemiological model 수정 필요 -> 후보 - SQEIR  
    +) 가능하다면, probabilistic programming과의 결합도 해보면 좋을 듯.

<br/>

1/28 Thur.
- EPIDEMIOPTIM([github](https://github.com/flowersteam/EpidemiOptim)) 분석
    * 문제 발생: 어제 돌려놓은 4개의 알고리즘 중 NSGA-ii에서 에러 발생  
        > Traceback (most recent call last):  
        >   File "train.py", line 77, in <module>  
        >     train(**kwargs)  
        >   File "train.py", line 66, in train  
        >     algorithm.learn(num_train_steps=params['num_train_steps'])  
        >   File "../epidemioptim/optimization/nsga/nsga.py", line 191, in learn  
        >     self.res_eval = self.evaluate(n=self.n_evals_if_stochastic if self.stochastic else 1, all=True)  
        >   File "../epidemioptim/optimization/nsga/nsga.py", line 205, in evaluate  
        >     for w in self.res.X:  
        > AttributeError: 'NoneType' object has no attribute 'X'  
    -> 해당 NSGA-ii 알고리즘이 경제하겡서의 최적화 방법을 활용한 알고리즘이므로 향후 연구에 쓰일 일이 없을 것 같아 해결하지 않음. (파라미터 설정이 더 필요한 것 같은데 해당 알고리즘에 대해 잘 모르겠어서 왜 에러가 발생하는지 못 앟아채는 것 같음)  
    
- RL for COVID-19 : 바꿔볼 옵션들 생각
    1. data를 우선 우리나라 데이터로
    2. lockdown을 우리나라는 시행한 적이 없음. 대신 사회적 거리두기 단계 조정. -> lockdown on/off 대신에 다양한 action 추가  
    3. action 다양화에 따른 epidemiological model 수정 필요 -> 후보 - SQEIR  
    4. (추가) learning algorithm DDPG 추가 (*DDPG가 DQN보다 continuous space에서 더 효과적인 것으로 알려져 있음)  
    +) 가능하다면, probabilistic programming과의 결합도 해보면 좋을 듯.
    
- Optimal policy learning for COVID-19 prevention using reinforcement learning - [link](https://journals.sagepub.com/doi/full/10.1177/0165551520959798) 분석
    - base로 사용한 데이터 없음. (그냥 보상 값으로 알고리즘별 결과 비교)
    - 바이러스 확산 모델 사용 안함 (epidemiological model 사용 X)
    - 대신 다양한 알고리즘 비교: Q-러닝, SARSA, DQN, DDPG

- RL algorithm 및 Pytorch 공부

- 중간 보고 자료 제작 중 (앞으로의 연구 주제 및 방향 확정)

<br/>

1/29 Fri.
- 연구 진행 상황 중간 보고 자료 제출

<br/>

2/1 Mon.
- Isaac Sim 공부
1. youtube
    - [Isaac Sim 2020: Deep Dive](https://www.youtube.com/watch?v=KGtGe87lY60) : AWS로 Isaac Sim을 실행하는 방법과 외부 로봇을 import, 조작할 수 있는 방법 소개
    - [Isaac Sim 2020.2 Omniverse Robotics App](https://www.youtube.com/watch?v=BM9q8vnjU6w) : Isaac Sim tutorial 내용 소개
2. 공식 tutorial - [link](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
- Sample Applications
    - CAD에서 Sample을 불러오고 stage, layer를 활용해 센서 부착, 물리 효과 추가 등을 하는 방법
    - Simple Robot Navigation   
        > setup: _create_robot()으로 환경 구성, _on_setup_fn()으로 로봇 컨트롤러 구성  
        > update: 명령에 따라 목적지로 이동  
- Rigging a Robot : CAD에서 부품을 불러온 후 결합하는 방법  
- Assets
    - warehouse sample 환경  
    <img src="https://user-images.githubusercontent.com/59794238/106408143-2b015b00-6481-11eb-96e2-5c393449da22.png" width="45%"></img>
    - STR 로봇  
    /Isaac/Robots/STR  
    <img src="https://user-images.githubusercontent.com/59794238/106408461-dca08c00-6481-11eb-88f4-8590603ac644.png" width="45%"></img>  
- Python Samples
    - Basic Time Stepping : CONFIG로 env 불러오기 → kit.play (simulation 시작) → callback 정의 → kit.update
    - Synthetic Data Generation
        - 임의로 box 만들고 이미지 처리를 하여 보여줌  
        <img src="https://user-images.githubusercontent.com/59794238/106423921-ae807380-64a4-11eb-9499-b69f5473a24a.png" width="45%"></img>  
        - 이미지 학습도 가능  
        <img src="https://docs.omniverse.nvidia.com/app_isaacsim/_images/isaac_synth-data_train.gif" width="45%"></img>   
    - Reinforcement Training
        - 기존의 gym과 비슷하게 동작한다. 하지만 기존 언어와 명령어가 다른 점이 많아 각각 어떤 내용인지 확인해야겠다.
        - jetbot_train.py, jetracer_train.py의 학습 과정 이해하기 + [robotic_warehouse](https://github.com/uoe-agents/robotic-warehouse)의 warehouse.py 내용 적용하기
 
 - RL for COVID-19: git remote repository from EpidemiOptim - [github](https://github.com/hyerinshelly/RL_COVID-19_Korea)
    - RL_COVID-19_Korea/epidemiOptim/environment/cost_functions 내 파일 추가 (변경 전 파일 이름 앞아 모두 'korea_' 붙임)
        - Add korea_multi_cost_death_gdp_controllable.py
            - 기존 파일에서 사망률 0.02로 변경
        - Add Korea ver. cost in RL_COVID-19_Korea/epidemiOptim/environment/cost_functions/costs
            - korea_death_toll_cost.py
            - korea_gdp_recess_cost.py : epidemiological model 수정 후 compute_current_infected & compute_cost 함수 식 수정 필요
    - RL_COVID-19_Korea/epidemiOptim/environment/models에 파일 for SQEIR 모델 추가 : sqeir_model.py
    
<br/>

2/2 Tue.
- Isaac Sim 공부
    - Reinforcement Training
        - jetbot_train.py : 흰색 벽과의 거리가 0이 되도록 강화 학습  
        <img src="https://user-images.githubusercontent.com/59794238/106545047-fb6b5500-654b-11eb-874b-f0ffeb67c360.png" width="30%"></img>  
        - jetracer_train.py : 주어진 선에서 벗어나지 않고 트랙을 돌도록 강화 학습  
        <img src="https://user-images.githubusercontent.com/59794238/106545400-b267d080-654c-11eb-9d6d-bda5629ac81b.png" width="30%"></img>  
    - Simple Navigation RL: RFMS 환경을 만들기에 앞서, 예제에 있는 Simple Navigation을 강화학습 형태로 구현해보려고 한다.
        - 예제 코드 주소 : isaac-sim-2020.2.2007-linux-x86_64-release/_build/linux-x86_64/release/exts/omni.isaac.samples/omni/isaac/samples
    - warehouse.py, jetbot_train.py, jetracer_train.py의 코드를 보고 각각 함수 내용을 정리하였다.  
        <img src="https://user-images.githubusercontent.com/59794238/106567805-bcea9000-6575-11eb-8a44-23bccf9de932.jpg" width="30%"></img>
        <img src="https://user-images.githubusercontent.com/59794238/106567809-bf4cea00-6575-11eb-8113-0d17fd20414b.jpg" width="30%"></img>  

- RL for COVID-19: epidemiological model & action 부분 스케치
    - 우리나라에 맞는 action 계획 세우기
        - 우리나라 거리두기 방침 확인 - 출처: 질병관리청 ([link](http://ncov.mohw.go.kr/socdisBoardList.do?brdId=6&brdGubun=64&dataGubun=641))
        - 1, 1.5, 2, 2.5, 3단계에 따른 1인당 평균 접촉자수 c_0 조절 계획 (3단계일 경우 c_0 ~= 0, 1단계일 경우 값 정해야함 -> 통계치 찾아보기)
            * EpimediOptim에서는 lockdown on/off에 따라 전염률 b 계산함. (역시나 최소는 0) 그러나 여기서는 전염 정도를 '전염률(바이러스의 특정, 고정값, 베타) * 1인당 접촉자수(action에 따라 변하는 값, c_0)'로 계산할 계획.
            
    - SQEIR model 스케치:  
    <img src="https://user-images.githubusercontent.com/47997946/106572180-75ff9900-657b-11eb-91ab-8cab5fcc8823.png" width="50%"></img> <img src="https://user-images.githubusercontent.com/47997946/106572351-a9422800-657b-11eb-8ed7-a996d57b3609.png" width="40%"></img>  
        * q : 확진자와 접촉 후 감염된 사람 중 격리될 비율(확률)  
    
    - 정해야 하는 파라미터들: (통계 더 찾아봐야 할 듯)  
        D_e : 확진자와 접촉 후 증상이 발현되는데 걸리는 평균 일 수  
        D_I : 증상 발현 후 회복되기까지 걸리는 평균 일 수  
        델타 : 격리되어 있지 않던 사람이 증상 발현 후 격리될 비율 (90퍼는 넘게 해야하지 않을까?)  
        
<br/>

2/3 Wed.
- Isaac Sim으로 간단한 프로그램 만들기  
    : STR 로봇이 A 지점에서 B 지점으로 이동하면 보상을 주는 간단한 강화학습 프로그램을 만들어보았다.  
    - Isaac Sim에서 데이터를 가져오는 방법  
        <img src="https://user-images.githubusercontent.com/59794238/106861912-c3aa0c00-6709-11eb-9083-0d405e025f18.png" width="50%"></img>
        <img src="https://user-images.githubusercontent.com/59794238/106861941-c9075680-6709-11eb-9742-fb5c69525be7.png" width="40%"></img>  
         위와 같은 방식으로 Pose, linear_velocity, local_linear_velocity, angular_velocity를 얻는다. Pose의 p는 position, r는 rotation을 의미한다.
    - 간단한 강화학습 프로그램 제작  
    : jetbot, jetracer 예제를 바탕으로 코드 내용을 이해하고 바꾸어 아래와 같이 STR을 warehouse env에서 사용하는 프로그램을 제작하였다.  
        <img src="https://user-images.githubusercontent.com/59794238/106862278-429f4480-670a-11eb-8174-5dda60680f2b.png" width="60%"></img>  
         그런데 python에서 새로운 물체(warehouse의 짐)를 추가한다거나 옮길 수 있는 방법을 모르겠어서 현실적으로 RMFS를 구현하기는 어려워보인다. (Isaac Sim에서 사용할 수 있는 함수에 대한 설명이 부족하다.) Isaac Sim 대신 ROS를 활용하거나 jetbot에서 직접 사용하는 방법으로 바꿔야 할 것 같다.  

- RL for COVID-19:
    - SQEIR model 파라미터 값 확정: (+ SQEIR model diagram 수정)  
    <img src="https://user-images.githubusercontent.com/47997946/106719919-318cff80-6646-11eb-884f-82c325e04da1.png" width="50%"></img> <img src="https://user-images.githubusercontent.com/47997946/106720041-56817280-6646-11eb-8d13-10ae11f0607e.png" width="45%"></img>  
    
        * Reference  
            - 인구수: [행정안전부](https://jumin.mois.go.kr/) (*2020.1.20. 첫 확진자 발생 -> 2020.2. 인구수로 가져옴)  
            - 각 compartment 별 초기값: [질병관리청 감염병포털](http://www.kdca.go.kr/npt/biz/npp/portal/nppIssueIcdView.do?issueIcdSn=176)  
            - 베타(transmision rate): coronaboard_kr/kr_daily.csv (첫 발생 이후 2달 후인 2020.3.20.을 기준으로 확진자수/검사수로 계산함. 그 이후는 증상이 없어도 검사를 받은 경우가 많거나 사회적 거리두기로 인한 수치 감소의 영향이 있을 것으로 판단되었기 때문  
            - c_0(1인 일 평균 접촉자 수): 기준값 2개(양 극단)를 잡고 균일하게 나눔 -> (거리두기 0, 1, 1.5, 2, 2.5, 3단계) = (40, 25, 20, 15, 10, 5)   
                - 거리두기를 하지 않을 때 40명 - [source](https://www.aimspress.com/article/10.3934/mbe.2020153)  
                - 거리두기 3단계 (10인 이상 집합 금지)에서 5명으로 설정  
            - q: 정리된 수치를 찾지는 못했지만 한 해동안 감염경로 불명 비율이 20%를 웃도는 것 같음  
            - D_e: [coronaqna.com의 '잠복기' 관련 글](https://www.coronaqna.com/incubation-period-of-covid-19)의 출처 [[5](https://www.nejm.org/doi/full/10.1056/NEJMoa2001316)]  
            - D_I: [경북대의대 연구 결과](https://m.health.chosun.com/svc/news_view.html?contid=2020062902672)  
            - D_T: 계산된 수치는 없지만, 확진자 동선을 보면 주로 증상이 발현되고 병원에 가 검사를 받고 확진이 되기까지 2~3일 걸린 것을 바탕으로 함.  

    - SQEIR diagram과 파라미터 설정에 맞춰 RL_COVID-19_Korea/epidemiOptim/environment/models/sqeir_model.py 코드 수정 중  

<br/>

2/4 Thu. 
- 공항 등에서 활용되는 로봇 주차에 대해 조사
RMFS 환경과 유사하게, Jetbot을 활용해 로봇 주차 환경을 만들 수 있겠다는 생각이 들었다. Ray, stan이라는 공항 주차 로봇이 이미 있다고 한다. - [link](https://blog.naver.com/tech-plus/221608574562)  
<img src="https://user-images.githubusercontent.com/59794238/106863766-52b82380-670c-11eb-9b01-9ef9b9844b0c.png" width="50%"></img>  

- openai ROS 공부  
    - [DeepSoccer](https://github.com/kimbring2/DeepSoccer)에서 사용한 openai ROS에 대해 찾아보고 tutorial을 해보았다. - [tutorial](http://wiki.ros.org/openai_ros/TurtleBot2%20with%20openai_ros)
    - training script와 env script가 서로 독립적(학습 방식, 학습 환경 따로)으로 되어있고 training script는 python으로 되어 있어 GPU를 사용한 강화학습이 가능하다.
    - 아래 사진과 같이 벽 앞에서 부딪히지 않고 회피하는 예제를 해보고 있는데 에러가 발생하여 동작하지 않는 중이다.  
    <img src="https://user-images.githubusercontent.com/59794238/106863601-197fb380-670c-11eb-9acf-90857418c716.png" width="50%"></img>  
   
- RL for COVID-19:
    - SQEIR modeling 완료: RL_COVID-19_Korea/epidemiOptim/environment/models/sqeir_model.py 코드 작성 완료
    - Plan: epidemiological model 완성 후 cost function 수정하면 될 것이라고 생각했는데, 단순 사망자수로 계산되는 health cost와 달리 현재 방역 정책이 몇단계인가에 따라 결정되는 economy cost의 경우 state에 'previous_lockdown_state, current_lockdown_state'가 포함되어 있는 것을 확인함. 이 부분 수정을 위해서는 gym_env에 또 korea ver.을 추가해야함. 따라서 state 설정을 먼저 마무리한 후 economy cost의 수식만 바꿔주면 될 듯.
    - gym env. 파일 수정 중
        - RL_COVID-19_Korea/epidemiOptim/environment/gym_env/get_env.py line 51(env 초기 설정)까지 수정 완료
        - Next to do: RL_COVID-19_Korea/epidemiOptim/environment/gym_env/epidemic_discrete.py 수정 후 get_env.py 마저 완성하기

<br/>

2/5 Fri. 
- openai ROS 공부  
    - tutorial
        TurtleBot2Maze-v0 환경에서 로봇이 미로를 탈출하는 tutorial을 구현하였다. 환경을 불러오지 못하는 문제가 있었는데, 아래와 같이 코드를 작성해 환경 정보가 담긴 turtlebot_maze.py에 추가하여 해결하였다.  
        > gym.error.UnregisteredEnv: No registered env with id: TurtleBot2Maze-v0  
        
        ```
        register(  
        id='TurtleBot2Maze-v0',  
        entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_maze:TurtleBot2MazeEnv',  
         )  
        ```
        
        <img src="https://user-images.githubusercontent.com/59794238/106981490-6f546a00-67a5-11eb-9301-892e460fac4e.png" width="50%"></img>  
        결과: EP: 500 - [alpha: 0.1 - gamma: 0.7 - epsilon: 0.55] - Reward: -127     Time: 1:22:42   
                
    - 새로운 환경 만드는 방법
    1. AI RL script : 학습 방법
        - yaml config file에 task_and_robot_environment_name, ros_ws_abspath 입력
        - AI RL script에 아래와 같이 입력  
            ```
            from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment # Init OpenAI_ROS ENV task_and_robot_environment_name = rospy.get_param('/turtlebot2/task_and_robot_environment_name')  
            env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)  
            ```
            
    2. Task environment : 전체 학습 환경
        - task_envs에 mkdir [환경 이름]/config/, parameter 내용이 저장된 yaml 파일을 config에 놓기
        - World launch : 'task_env/your_task/your_task.py'를 생성하여 robot을 제외한 env을 불러온다. init에서 yaml의 params load, launch script
        - Task environment registration : openai_ros_common.py에서 사용할 시뮬레이션 환경을 ‘git embedded files’로 입력, task_envs_list.py에서 환경 register
    3. Robot Environment : 로봇
        - 다른 코드를 참고하여 robot_envs 폴더에 YOUR_ROBOT.py 파일을 만들고 ROSLauncher로 spawn robot
        - Task environment registration과 같이 openai_ros_common.py 변경
        
- jetbot 환경 설치  
    [jetbot_ros](https://github.com/dusty-nv/jetbot_ros)를 참고하여 로봇 환경을 설치하려 하였는데, jetson NvInfer.h를 찾지 못한다는 에러와 catkin_make가 되지 않는 에러가 발생하였다. TensorRT가 필요해보이는데, jetbot에서는 잘 동작하고 있기 때문에 데스크탑에서 설치하는 것은 포기하였다. jetbot 로봇 대신 임시적으로 turtlebot을 활용해 환경을 구현해야겠다.  
   
<br/>

2/8 Mon.  
- openai ROS 다른 환경 test 및 공부
    - [다른 tutorial 환경](https://bitbucket.org/theconstructcore/openai_examples_projects/src/master/) 실행
        - Turtlebot3  
            <img src="https://user-images.githubusercontent.com/59794238/107170220-2568bf80-6a03-11eb-9d5a-f7d4ac9b9e36.png" width="50%"></img>  

        - error
            - 예제에서 제공하는 환경들이 python 2에서 만들어져 대부분 에러가 발생한다. (turtlebot3만 python3로 실행 가능)
            > ImportError: dynamic module does not define module export function (PyInit__tf2)  
            - 아래 에러는 Task Environment 파일의 timestep_limit을 max_episode_steps로 바꾸어 해결하였다.  
            > TypeError: __init__() got an unexpected keyword argument 'timestep_limit'  
            - 아래 에러를 해결하기 위해 config 파일에서 경로를 지정해주었다.
            > AssertionError: You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: 'YOUR/SIM_WS/PATH'
  
            ```
            task_and_robot_environment_name: 'TurtleBot2Maze-v0'  
            ros_ws_abspath: "/home/jwk/catkin_ws"  
            ```
            
            - re-register 문제: task_env안의 파일의 register 위에 코드 추가
            > gym.error.Error: Cannot re-register id: TurtleBot3World-v0  
            
            ```
            import gym
            
            env_dict = gym.envs.registration.registry.env_specs.copy()  
            for env in env_dict:  
                if 'TurtleBot3World-v0' in env:  
                    print("Remove {} from registry".format(env))  
                    del gym.envs.registration.registry.env_specs[env]  
            ```
            
            - 강제종료 후에도 gazebo가 실행되는 문제
            > alias killgazebo="killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient"
            
    - turtlebot2, turtlebot3의 환경과 학습 과정 공부
        - 환경 정보 (turtlebot2_maze.py)  
            <img src="https://user-images.githubusercontent.com/59794238/107185542-02033c00-6a26-11eb-88f1-8a2ab57661c3.png" width="50%"></img>  
        - 학습 과정 (start_training.py)  
            <img src="https://user-images.githubusercontent.com/59794238/107185203-56f28280-6a25-11eb-964e-610b7de848f4.png" width="50%"></img>  
        - 직접 만든 gazebo 환경을 my_turtlebot2_training에 적용  
            <img src="https://user-images.githubusercontent.com/59794238/107187456-61168000-6a29-11eb-8cfc-a5bbad49e0a5.png" width="50%"></img>  
            1. 환경 : gazebo 실행 후 제작, worlds 파일에 .world로 저장
            2. 경로 지정 : turtlebot2_maze.py에서 roslaunch되는 start_world_maze_loop_brick.launch -> world_file 경로 수정
- 계획 변경
    1. gazebo 환경 제작 방법 공부 - [tutorial](http://gazebosim.org/tutorials)
    2. 박스가 놓아져 있을 때 로봇이 빈 장소를 찾아 짐을 놓고 돌아오는 프로그램 제작 (가장 간단한 방법)
    3. (추가) 로봇 수 늘리기, 빈 자리 늘리기, 실제 로봇에 적용

- 시뮬레이터에서 학습한 모델을 실제 세상에서 활용하는 방법  
    - [jetbot line tracing](https://developer.nvidia.com/blog/training-your-jetbot-in-isaac-sim/)  
    isaac sim 안에서는 jetbot의 카메라 정보를 바탕으로 학습하고 모델을 저장한다. 이 모델을 사용하여 실제 세상에서도 카메라로 본 정보를 바탕으로 예측, 이동한다.  
    카메라 정보, 실제 세상 간에 카메라로 보이는 모습의 차이가 있기 때문에 [VAE를 활용](https://github.com/masato-ka/airc-rl-agent)해 차이를 줄인다.  
    - [Deepsoccer](https://kimbring2.github.io/2020/10/08/deepsoccer.html): Lidar sensor 정보 이용, 원리는 같다.  

- RL for COVID-19: gym env. 파일 수정 중
    - 기존 gym env를 수정한 버전인 RL_COVID-19_Korea/epidemiOptim/environment/gym_env/korea_epidemic_discrete.py를 만들고 plotting data 전까지의 함수들 짬.
        - State 설정 : SQEIR model의 compartment + [previous_distancing_level, current_distancing_level] + [culmulative_cost_{}.format(0, 1)]
        - Action 수 : dim_action = 6 (0~5) <- 각각 차례로 사회적거리두기 0, 1, 1.5, 2, 2.5, 3단계 의미함
        - _compute_c(times_since_start) : 주어진 시간부터 현재까지의 사회적거리두기 단계에 따른 일 평균 접촉수(c) 일 수대로 리스트에 담아 리턴함
        - _update
        - reset()
            * 의문점: reset()에서 env_state를 return할 때 normalization_factor을 이용해 normalize하는 것을 확인하였으나 왜 하는 것이며 그 factor 설정을 어떻게 하는 것인지 모르겠음.
                > self.normalization_factors = [self.model.current_internal_params['N']] * len(self.model.internal_states_labels) + \
                >                              [1, 1, self.model.current_internal_params['N'], 150, 1]  # not sure also. 150?
        - update_with_action(action)
        - step(action)
        
        * data plotting을 위한 함수만 남겨둠  
        * 중간중간 확실하지 않은 부분에는 '# not sure'이라는 주석 달아놓음 -> 나중에 필요 시 확인  
        
    - 다음 plan:
        - 남은 plotting 부분 완성
        - RL_COVID-19_Korea/epidemiOptim/environment/gym_env/get_env.py 마저 완성하기
        - cost function 마무리
         
<br/>

2/9 Tue.  
- [Gazebo tutorial](https://youtube.com/playlist?list=PLK0b4e05LnzbHiGDGTgE_FIWpOCvndtYx) 내용
    1. Worlds 파일 만들기 → my_world.launch, empty_world.world 생성
    2. [gazebo_models](https://github.com/osrf/gazebo_models)에서 공개된 모델 open source들을 이용할 수 있음, url 부분을 변경하여 추가 (include)
        - static: 고정, false면 play했을 때 중력 영향
        - size로 크기 변경
        - model 폴더에 meshes/something.stl 추가 → mesh를 추가하면 물체의 근처에만 box가 생기게 됨
        - texture 추가: materials 추가, 그 안에 scripts, textures 폴더 추가하고 scripts에 .material, textures에 .png 추가하면 모델에 색이 칠해짐.
    3. robot 정보 : urdf 파일, .xacro에 로봇 정보
        - .urdf → robot name 지정 → link, joint로 구성
        - .launch에서 초기 위치 지정, node 불러옴
        - rviz에서 확인 가능, rviz 변경 사항을 save as config로 저장하고 경로 지정하면 반영됨
        - child link로 물체 연결, node 안에서 use_gui를 true로 설정 → joint 위치 변경 가능
    4. sdf는 특정 모델의 내용을 담음 (ex. chair의 각 요소들) .world에서 sdf 파일을 가져와 실행 가능
    

- RL for COVID-19: 코드 마무리
    - RL_COVID-19_Korea/epidemiOptim/environment/gym_env/korea_epidemic_discrete.py data plotting 부분 완성
        - Changed plotting function in RL_COVID-19_Korea/epidemioptim/utils.py
    - RL_COVID-19_Korea/epidemiOptim/environment/gym_env/get.env.py 맞춰 완성
    - RL_COVID-19_Korea/epidemiOptim/environment/gym_env/run_distrib_env.py 맞춰 완성
    - RL_COVID-19_Korea/epidemioptim/environments/cost_functions/costs/korea_economy_cost.py 복잡한 GDP 손실 계산 대신 총 격리된 사람의 수(number of people in S_q, E_q, I_q and in death)에 economy cost 비례하도록 함
    - cost function 수정에 따라 RL_COVID-19_Korea/epidemioptim/environments/cost_functions/get_cost_function.py & korea_multi_cost_death_economy_controllable.py 완성
    - 최종 RL_COVID-19_Korea/epidemioptim/train.py 파라미터 맞춤 및 완성
        - 새로 만든 gym env, epidemiological model, cost function을 반영하도록 새로운 파라미터 담은 파일 만듦. (RL_COVID-19_Korea/epidemiOptim/configs/korea_dqn.py)
            - 역시나 해당 korea_dqn.py 파라미터 불러오도록 RL_COVID-19_Korea/epidemiOptim/configs/get_params.py 수정
        - train.py 실행 시 '--config korea_dqn'으로 하면 됨
            > python train.py --config korea_dqn --expe-name korea_dqn_study --trial_id 0
    
    - Next To Do: train.py 실행해보기!!!

<br/>

2/15 Mon.
- [gazebo_models](https://github.com/osrf/gazebo_models)의 공개된 모델 중 사용할만한 모델 추리기
    - robot  
    warehouse robot: <img src="https://user-images.githubusercontent.com/59794238/107336872-fcc4f080-6afc-11eb-9bd3-8787ad2785f5.png" width="15%"></img>, 
    cart_front_steer: <img src="https://user-images.githubusercontent.com/59794238/107894559-59982f00-6f73-11eb-86c4-bf1240fe8cc4.png" width="20%"></img>  
    - parking  
    parking_garage: <img src="https://user-images.githubusercontent.com/59794238/107894602-7cc2de80-6f73-11eb-9d3b-78a777d7e2c6.png" width="20%"></img>, 
    prius_hybrid: <img src="https://user-images.githubusercontent.com/59794238/107894624-8a786400-6f73-11eb-8c0a-b944df33b5d6.png" width="20%"></img>,  
    hatchback: <img src="https://user-images.githubusercontent.com/59794238/107894589-6fa5ef80-6f73-11eb-98a6-66984c885c60.png" width="20%"></img>,
    suv: <img src="https://user-images.githubusercontent.com/59794238/107894637-96fcbc80-6f73-11eb-827f-7f6c815ea536.png" width="20%"></img>  
    - ware  
    cardboard_box: <img src="https://user-images.githubusercontent.com/59794238/107894657-a2e87e80-6f73-11eb-890b-0ed75d1a85a0.png" width="20%"></img>, 
    euro_pallet: <img src="https://user-images.githubusercontent.com/59794238/107894576-6452c400-6f73-11eb-8e74-63fcebed84f8.png" width="20%"></img>  
    
- [warehouse_world](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world)  
    <img src="https://user-images.githubusercontent.com/59794238/107896966-38d3d780-6f7b-11eb-9051-3ee4f32f776c.png" width="40%"></img>  
    
- 아래 사진과 같은 warehouse world 환경을 만들었다.  
<img src="https://user-images.githubusercontent.com/59794238/107916839-50768480-6faa-11eb-8af1-f9ffddfe1a38.png" width="40%"></img>
    - 물품을 놓는 부분을 box로 표시하였고 필요에 따라 붉은색, 초록색으로 변경할 수 있다.
    - 물품이 쌓여 있는 앞 부분이 station으로, 로봇이 근처에 도달하면 보상을 주는 방식으로 만들 계획이다.
    - 아래 사진과 같이 물품을 운반하는 로봇과 물품이 필요하다. 직접 만들어보고 안 될 경우 turtlebot을 사용할 것이다.  
    <img src="https://www.roboticsbusinessreview.com/wp-content/uploads/2018/05/FetchWarehouse.jpg" width="40%"></img>  
    - 그 전에 먼저 [RWARE 환경](https://github.com/uoe-agents/robotic-warehouse)에서 어떻게 필요한 물품을 선택, 결정하여 운반하고 보상을 부여하는지 공부해야 한다.
    
