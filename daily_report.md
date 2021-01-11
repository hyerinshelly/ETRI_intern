# 작업 일지

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
    1. Ch.1 : 강화학습 개요 (용어, 행동-가치/상태-가치 함수), 벨만 방정식, 마르코프체인  
            - 벨만 방정식:  
            <img src="https://user-images.githubusercontent.com/47997946/103871802-de8c5f00-5110-11eb-88ee-6e5fddb08bd0.png" width="50%"></img>
    2. Ch.2 : 동적계획법 (가치반복, 정책반복 알고리즘) -> 최적화  
            - 가치반복 알고리즘: 각 칸의 가치를 수렴할 때까지 계산하여 최적의 방법 찾음  
            - 정책반복 알고리즘: 임의의 정책의 리워드를 바탕으로 정책을 업데이트하며 최적의 정책 찾음  
            
<br/>
    
1/11 Mon.
- 연구주제 설정을 위한 아이디어 탐색 (마인드맵 - 구글 코글)
    - [젯봇](https://coggle.it/diagram/X_ujT44ja19ZhWAI/t/%EC%A0%AF%EB%B4%87-image)
    - [강화학습](https://coggle.it/diagram/X_ulwgfKJimvn5Cg/t/%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5)
- jetbot image를 다운받아 SD카드에 flash하여 부팅 다시 함 -> 데이터 훈련하고 그 훈련시킨 모델을 불러올 때 실행이 멈추는 현상 해결 -> 간단히 collision avoidance 잘 작동함 확인 완료
- [edwith MOVE37 강의](https://www.edwith.org/move37/joinLectures/25196) 강화학습 공부 진행 중  
    3. Ch.3 : 모델프리 강화학습 (환경을 알지 못해도 최적의 정책 찾기) 
              > 몬테카를로 방법 (무작위로 정책 시도) + 입실론-그리디 (무작위 선택 시 새로운 탐색과 기존 지식 활용 중 선택)
              > Q-learning (Q[s][a] 테이블을 만들어 평균값 계산 -> 가치 V(s)) => 최적의 정책 결정

