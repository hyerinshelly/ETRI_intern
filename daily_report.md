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
- jetson community 프로젝트들을 살펴봄 - [link](https://developer.nvidia.com/embedded/community/jetson-projects#ros_navbot)
