# Daily Report
## Reinforcement Learning for Finding COVID-19 Policy with Various Actions

1/25 Mon.
- Reinforcement learning for Covid-19 자료 정리
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

<br/>

1/28 Thur, 1/29 Fri.
- 연구 진행 상황 중간 보고 자료 제출 (앞으로의 연구 주제 및 방향 확정)

<br/>

2/1 Mon.
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

- RL for COVID-19:
    - SQEIR modeling 완료: RL_COVID-19_Korea/epidemiOptim/environment/models/sqeir_model.py 코드 작성 완료
    - Plan: epidemiological model 완성 후 cost function 수정하면 될 것이라고 생각했는데, 단순 사망자수로 계산되는 health cost와 달리 현재 방역 정책이 몇단계인가에 따라 결정되는 economy cost의 경우 state에 'previous_lockdown_state, current_lockdown_state'가 포함되어 있는 것을 확인함. 이 부분 수정을 위해서는 gym_env에 또 korea ver.을 추가해야함. 따라서 state 설정을 먼저 마무리한 후 economy cost의 수식만 바꿔주면 될 듯.
    - gym env. 파일 수정 중
        - RL_COVID-19_Korea/epidemiOptim/environment/gym_env/get_env.py line 51(env 초기 설정)까지 수정 완료
        - Next to do: RL_COVID-19_Korea/epidemiOptim/environment/gym_env/epidemic_discrete.py 수정 후 get_env.py 마저 완성하기

<br/>

2/8 Mon.  
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

2/10 Wed. 
- RL for COVID-19: dqn 알고리즘 부분 코드 훑어봄

<br/>

2/16 Tue.
- RL for COVID-19: train.py 실행 시도
    > $ python train.py --config korea_dqn --expe-name korea_dqn_study --trial_id 0

    - costs의 __init__.py 파일에서 economy cost의 클래스명을 바꾼 것이 수정되지 않아 수정함
    - sqeir_model.py에서 run_n_step함수 내의 미분방정식을 계산해주는 odeint 함수 실행 시 타입 에러 발생
        > z = odeint(self.internal_model, current_state, np.linspace(0, n, n + 1), args=self._get_model_params())
        ```
          File "../epidemioptim/environments/models/sqeir_model.py", line 225, in run_n_steps
            z = odeint(self.internal_model, current_state, np.linspace(0, n, n + 1), args=self._get_model_params())
          File "/home/iot/anaconda3/lib/python3.8/site-packages/scipy/integrate/odepack.py", line 241, in odeint
            output = _odepack.odeint(func, y0, t, args, Dfun, col_deriv, ml, mu,
        TypeError: Cannot cast array data from dtype('O') to dtype('float64') according to the rule 'safe'
        ```
        (원인) 'print(current_state.dtype)'으로 확인해보니 기존 epidemiOptim과는 달리 odeint의 parameter 값으로 들어가는 current_state의 타입이 float64가 아닌 object였음.
        그래서 타입 확인을 넘어 current_state 자체를 출력하도록 했더니 다음과 같은 에러 발생함.  
        ```
        Traceback (most recent call last):
          File "train.py", line 78, in <module>
            train(**kwargs)
          File "train.py", line 51, in train
            env = get_env(env_id=params['env_id'],
          File "../epidemioptim/environments/gym_envs/get_env.py", line 16, in get_env
            env = gym.make(env_id, **kwargs)
          File "/home/iot/anaconda3/lib/python3.8/site-packages/gym/envs/registration.py", line 145, in make
            return registry.make(id, **kwargs)
          File "/home/iot/anaconda3/lib/python3.8/site-packages/gym/envs/registration.py", line 90, in make
            env = spec.make(**kwargs)
          File "/home/iot/anaconda3/lib/python3.8/site-packages/gym/envs/registration.py", line 60, in make
            env = cls(**_kwargs)
          File "../epidemioptim/environments/gym_envs/epidemic_discrete.py", line 48, in __init__
            self.normalization_factors = [self.model.current_internal_params['N_av']] * len(self.model.internal_states_labels) + \
        KeyError: 'N_av'
        ```
        (분석) 보면 불러야하는 korea_epidemic_discrete.py가 아닌 기존 것을 불러옴. gym_env가 잘못 불러지므로 current_state가 이상할 수 밖에 없음.  
        (해결 방법) get_env.py의 main 부분에서 env_id를 바꿔주지 않은 것을 발견함. 따라서 아래와 같이 바꿔주니 current_state가 7개의 숫자를 갖는 배열임을 확인함.  
        > env = get_env(env_id='KoreaEpidemicDiscrete-v0', params=dict(cost_function=cost_function, model=model, simulation_horizon=simulation_horizon))  
        
        (문제) 하지만 여전히 타입에러 발생함. 즉 여전히 current_state의 타입은 object. 
        
<br/>

2/17 Wed.
- RL for COVID-19: train.py 실행 시도 (이어서)
    - sqeir_model.py에서 run_n_step함수 내의 미분방정식을 계산해주는 odeint 함수 실행 시 타입 에러 발생  
        (어제 문제 해결) 타입 설정이 어디서 잘못되나를 파악해보려 했으나 실패. 그냥 아래와 같이 panda의 to_numeric 함수를 이용해 강제로 타입 설정해줌.
        > current_state = pd.to_numeric(current_state)
        
    - 그런데 어제 gym env 불러오는 것을 수정해줬음에도 여전히 이전의 env에 접근하며 어제와 같은 에러 발생함.  
        (원인) gym env가 __init__.py에 register 되지 않음 + korea_dqn에 담긴 env id가 수정 필요.  
        (해결) 아래와 같이 만든 gym register 해줌  
        > register(id='KoreaEpidemicDiscrete-v0',
        >          entry_point='epidemioptim.environments.gym_envs.korea_epidemic_discrete:KoreaEpidemicDiscrete')
    
    - env의 state label 중 level_c를 사용하지 않고 distancing_level를 사용했어서 남아있던 level_c 설정을 삭제하여 아래 에러 해결함  
        ```
          File "../epidemioptim/environments/gym_envs/korea_epidemic_discrete.py", line 114, in _update_env_state
            level_c=self.level_c)
        AttributeError: 'KoreaEpidemicDiscrete' object has no attribute 'level_c'
        ```
        
    - 마찬가지로 level_c에 해당하는 normalizing_factor 값을 삭제해 array 길이 차이로 인한 아래의 에러 해결
        ```
          File "../epidemioptim/environments/gym_envs/korea_epidemic_discrete.py", line 185, in reset
              return self._normalize_env_state(self.env_state)
            File "../epidemioptim/environments/gym_envs/korea_epidemic_discrete.py", line 304, in _normalize_env_state
              return (env_state / np.array(self.normalization_factors)).copy()
        ValueError: operands could not be broadcast together with shapes (11,) (12,)
        ```
    
    - 드디어 learning(알고리즘 학습) 부분이 돌아가기 시작하였으나 아래의 에러 발생
        ```
        Traceback (most recent call last):
          File "train.py", line 78, in <module>
            train(**kwargs)
          File "train.py", line 67, in train
            algorithm.learn(num_train_steps=params['num_train_steps'])
          File "../epidemioptim/optimization/dqn/dqn.py", line 400, in learn
            episodes = run_rollout(policy=self,
          File "../epidemioptim/optimization/shared/rollout.py", line 64, in run_rollout
            action, q_constraints = policy.act(augmented_state, deterministic=eval)
          File "../epidemioptim/optimization/dqn/dqn.py", line 322, in act
            state = ag.Variable(torch.FloatTensor(state).unsqueeze(0))
        TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        ```
        (원인) 전에 current_state의 타입이 float64이어야 하는데 object였던 것과 동일하게 모든 state가 numeric type이 아닌 object type이 되어 문제 발생하는 것 같음.  
        (분석) 따라서 전에는 당장 문제가 되는 current_state의 타입만 강제로 바꿔주어 닥친 에러를 해결했지만, 근본적으로 state의 타입 설정을 고쳐주어야 할 것으로 판단됨.  
        (해결) dqn.py에서 핵습을 하는 핵심 부분인 act 함수에서 기존 바디 내용 실행 전 state의 타입을 예전과 비슷하게 강제로 numeric type이 되도록 함.
            > state = pd.to_numeric(state)
        
    - 학습이 돌아가는 듯 보였지만, 계속해서 값(r1인 듯?)이 illegal하다는 warning 메세지가 계속 뜨고 convergence fail 메세지도 뜨는 듯 하더니 결국 숫자 타입이 맞지 않아 cost를 계산할 수 없다는 메세지의 에러 발생함. (자동 종료됨)
        ```
        ... # 생략
        
         intdy--  t (=r1) illegal      
              in above message,  r1 =  0.1000000000000D+01
              t not in interval tcur - hu (= r1) to tcur (=r2)       
              in above,  r1 =  0.0000000000000D+00   r2 =  0.0000000000000D+00
        AttributeError: 'float' object has no attribute 'sqrt'

        The above exception was the direct cause of the following exception:

        Traceback (most recent call last):
          File "train.py", line 78, in <module>
            train(**kwargs)
          File "train.py", line 67, in train
            algorithm.learn(num_train_steps=params['num_train_steps'])
          File "../epidemioptim/optimization/dqn/dqn.py", line 429, in learn
            new_logs, eval_costs = self.evaluate(n=self.n_evals_if_stochastic if self.stochastic else 1)
          File "../epidemioptim/optimization/dqn/dqn.py", line 464, in evaluate
            new_logs, costs = self.compute_eval_score(eval_episodes, eval_goals)
          File "../epidemioptim/optimization/dqn/dqn.py", line 478, in compute_eval_score
            costs_std = np.std(costs[ind_g], axis=0)
          File "<__array_function__ internals>", line 5, in std
          File "/home/iot/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 3496, in std
            return _methods._std(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
          File "/home/iot/anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py", line 237, in _std
            ret = um.sqrt(ret, out=ret)
        TypeError: loop of ufunc does not support argument 0 of type float which has no callable sqrt method
        ```
        (원인? 분석?) 검색해서 찾아보니 dtype이 object인 걸 억지로 numeric하게 바꿨을 경우 그럴 수 있댄다.. 그럼 어떻게 하지,,  
        (해결?) 알고보니 위의 문제에서 act함수 돌리기전 state의 타입이 이미 float64였다.. 잘못된 해결 방법이었다. (그런데 왜 에러 내용이 바뀐거지??) 아래와 같이 다시 원점이다..  
        ```
        ... # 생략
        
         lsoda--  at t (=r1) and step size h (=r2), the    
               corrector convergence failed repeatedly     
               or with abs(h) = hmin   
              in above,  r1 =  0.0000000000000D+00   r2 =  0.4713305315119-109
         lsoda--  warning..internal t (=r1) and h (=r2) are
               such that in the machine, t + h = t on the next step  
               (h = step size). solver will continue anyway
              in above,  r1 =  0.0000000000000D+00   r2 =  0.0000000000000D+00
         intdy--  t (=r1) illegal      
              in above message,  r1 =  0.1000000000000D+01
              t not in interval tcur - hu (= r1) to tcur (=r2)       
              in above,  r1 =  0.0000000000000D+00   r2 =  0.0000000000000D+00
        Traceback (most recent call last):
          File "train.py", line 78, in <module>
            train(**kwargs)
          File "train.py", line 67, in train
            algorithm.learn(num_train_steps=params['num_train_steps'])
          File "../epidemioptim/optimization/dqn/dqn.py", line 404, in learn
            episodes = run_rollout(policy=self,
          File "../epidemioptim/optimization/shared/rollout.py", line 64, in run_rollout
            action, q_constraints = policy.act(augmented_state, deterministic=eval)
          File "../epidemioptim/optimization/dqn/dqn.py", line 326, in act
            state = ag.Variable(torch.FloatTensor(state).unsqueeze(0))
        TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        ```
        
<br/>

2/18 Thur.
- RL for COVID-19: train.py 실행 시도 (이어서)
    - 어제의 에러(아래) 문제
        ```
          File "../epidemioptim/optimization/dqn/dqn.py", line 326, in act
            state = ag.Variable(torch.FloatTensor(state).unsqueeze(0))
        TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        ```
        (분석) 다시 'print(state.dtype)'을 넣고 끝까지 돌려보니 처음에는 float64로 잘 돌아가다가 갑자기 object로 바뀌더니 위와 같은 에러 발생. 아마도 한 스텝 한 스텝 돌릴 때는 문제가 없는데 에피소드로 모아 저장을 하고 roll out을 할 때 담긴 augmented_state의 타입이 object인 것으로 파악됨.  
        (시도) torch.from_numpy() 사용? -> 역시나 안됨  
        (해결 방안) 만약 타입이 object이면 torch.FloatTensor(list(state).unsqueeze(0)) 하게 함.  
        
        (추가) 마찬가지로 tensor.FloatTensor을 쓰는 데에서 같은 에러(아래) 발생하는 곳도 마찬가지로 FloatTensor 안에 list화 시킴.  
        ```
        Traceback (most recent call last):
          File "train.py", line 78, in <module>
            train(**kwargs)
          File "train.py", line 67, in train
            algorithm.learn(num_train_steps=params['num_train_steps'])
          File "../epidemioptim/optimization/dqn/dqn.py", line 432, in learn
            update_losses.append(self.update())
          File "../epidemioptim/optimization/dqn/dqn.py", line 354, in update
            losses = self._update(self.batch_size)
          File "../epidemioptim/optimization/dqn/dqn.py", line 229, in _update
            rewards = [- ag.Variable(torch.FloatTensor(c_func.scale(c))) for c_func, c in zip(self.cost_function.costs, costs.transpose())]
          File "../epidemioptim/optimization/dqn/dqn.py", line 229, in <listcomp>
            rewards = [- ag.Variable(torch.FloatTensor(c_func.scale(c))) for c_func, c in zip(self.cost_function.costs, costs.transpose())]
        TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        ```
        
    - 결국 위의 에러 전의 에러로 돌아왔다.
        ```
        /home/iot/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
          return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
        AttributeError: 'float' object has no attribute 'sqrt'

        The above exception was the direct cause of the following exception:

        Traceback (most recent call last):
          File "train.py", line 78, in <module>
            train(**kwargs)
          File "train.py", line 67, in train
            algorithm.learn(num_train_steps=params['num_train_steps'])
          File "../epidemioptim/optimization/dqn/dqn.py", line 442, in learn
            new_logs, eval_costs = self.evaluate(n=self.n_evals_if_stochastic if self.stochastic else 1)
          File "../epidemioptim/optimization/dqn/dqn.py", line 477, in evaluate
            new_logs, costs = self.compute_eval_score(eval_episodes, eval_goals)
          File "../epidemioptim/optimization/dqn/dqn.py", line 491, in compute_eval_score
            costs_std = np.std(costs[ind_g], axis=0)
          File "<__array_function__ internals>", line 5, in std
          File "/home/iot/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 3496, in std
            return _methods._std(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
          File "/home/iot/anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py", line 237, in _std
            ret = um.sqrt(ret, out=ret)
        TypeError: loop of ufunc does not support argument 0 of type float which has no callable sqrt method
        ```
        (해결) np.mean, np.std를 적용시키는 costs(numpy array)의 타입을 float을 담은 object가 아닌 자체가 float64가 되도록 아래와 같이 costs를 만들 때 지정해줌.  
            > # costs = np.array([np.sum(e['costs'], axis=0) for e in eval_episodes]) # before
            > costs = np.array([np.sum(e['costs'], axis=0) for e in eval_episodes], dtype=np.float64)
        
    - 드디어 아래와 같이 잘 돌아가는 중... 그러나
        ```
        Episode: 2650.00
            Best score so far: inf
            Eval score: nan
            Loss 1: nan
            Loss 2: nan
            Train, Cost 1: nan
            Train, Cost 2: nan
            Train, Aggregated cost: nan
            Eval, g: [0.25], C1: nan +/- nan, C2: nan +/- nan, Agg: nan +/- nan
            Eval, g: [0.5], C1: nan +/- nan, C2: nan +/- nan, Agg: nan +/- nan
            Eval, g: [0.75], C1: nan +/- nan, C2: nan +/- nan, Agg: nan +/- nan
            Eval, g: [0.], C1: nan +/- nan, C2: nan +/- nan, Agg: nan +/- nan
            Eval, g: [1.], C1: nan +/- nan, C2: nan +/- nan, Agg: nan +/- nan
        ```
        (문제) 수렴을 전혀 하지 않고 있다. best score이 inf라니.. 전혀 학습이 되지 않고 있다.  
        (분석1) 기존 모델의 학습(아래 참조)과 비교해보면 명확히 다르다.  
        ```
        Episode: 2450.00
            Best score so far: 35.26
            Eval score: 44.17
            Loss 1: 13.03
            Loss 2: 2.20
            Train, Cost 1: 6155.39
            Train, Cost 2: 79.15
            Train, Aggregated cost: 44.31
            Eval, g: 0.5, C1: 2494.96 +/- 2725.36, C2: 84.50 +/- 6.39, Agg: 44.17 +/- 44.17
        ```  
        (분석2) 아래 warning이 계속 반복된다.
        ```
         intdy--  t (=r1) illegal      
              in above message,  r1 =  0.1000000000000D+01
              t not in interval tcur - hu (= r1) to tcur (=r2)       
              in above,  r1 =  0.0000000000000D+00   r2 =  0.0000000000000D+00
         lsoda--  warning..internal t (=r1) and h (=r2) are
               such that in the machine, t + h = t on the next step  
               (h = step size). solver will continue anyway
              in above,  r1 =  0.0000000000000D+00   r2 =  0.0000000000000D+00
        ... # 중략
         lsoda--  at t (=r1) and step size h (=r2), the    
               corrector convergence failed repeatedly     
               or with abs(h) = hmin   
              in above,  r1 =  0.0000000000000D+00   r2 =  0.2918205452041-109
        ```  
        (분석3) train을 실행하고 초반에 이런 RuntimeWarning이 발생함을 확인하였다. cost가 인구수에 비례하게 해놨는데 인구수(S, S_q, E, E_q)에서 overflow가 발생했으니 당연히 cost 값이 올바르지 않고 이로 인해 best score 값도 inf였던 것이다. overflow의 원인을 찾는 것이 급한 일이 되었다. overflow가 일어 나는 4개의 compartment에만 공통으로 들어가 있는 값은 beta0, q, c, S이다.   
        ```
        Logging to:  /home/iot/바탕화면/RL_COVID-19_Korea/data/results/KoreaEpidemicDiscrete-v0/DQN/korea_dqn_study_1500/
        ../epidemioptim/environments/models/sqeir_model.py:63: RuntimeWarning: overflow encountered in double_scalars
          dSdt = - beta*c*S*I/N - (1-beta)*c*q*S*I + S_q/14
        ../epidemioptim/environments/models/sqeir_model.py:63: RuntimeWarning: invalid value encountered in double_scalars
          dSdt = - beta*c*S*I/N - (1-beta)*c*q*S*I + S_q/14
        ../epidemioptim/environments/models/sqeir_model.py:65: RuntimeWarning: overflow encountered in double_scalars
          dS_qdt = - S_q/14 + (1-beta)*c*q*S*I
        ../epidemioptim/environments/models/sqeir_model.py:67: RuntimeWarning: overflow encountered in double_scalars
          dEdt = - (1-q)*beta*c*S*I/N - E/De
        ../epidemioptim/environments/models/sqeir_model.py:69: RuntimeWarning: overflow encountered in double_scalars
          dE_qdt = - q*beta*c*S*I/N - E_q/De
         lsoda--  warning..internal t (=r1) and h (=r2) are
               such that in the machine, t + h = t on the next step  
               (h = step size). solver will continue anyway
              in above,  r1 =  0.0000000000000D+00   r2 =  0.0000000000000D+00
      ```  
      (시도) dSdt와 dS_qdt 수식에 치명적인 실수를 발견하였다. 전제 인구수 비례로 나누지 않은 항이 있었다. 하지만 이를 고쳤는데도 아래처럼 RuntimeWarning은 그대로다. 역시 beta0, q, c, S가 용의자인 것인가.  
      ```
      Logging to:  /home/iot/바탕화면/RL_COVID-19_Korea/data/results/KoreaEpidemicDiscrete-v0/DQN/korea_dqn_study_1700/
      ../epidemioptim/environments/models/sqeir_model.py:63: RuntimeWarning: overflow encountered in double_scalars
        dSdt = - beta0*c*S*I/N - (1-beta0)*c*q*S*I/N + S_q/14
      ../epidemioptim/environments/models/sqeir_model.py:63: RuntimeWarning: invalid value encountered in double_scalars
        dSdt = - beta0*c*S*I/N - (1-beta0)*c*q*S*I/N + S_q/14
      ../epidemioptim/environments/models/sqeir_model.py:65: RuntimeWarning: overflow encountered in double_scalars
        dS_qdt = - S_q/14 + (1-beta0)*c*q*S*I/N
      ../epidemioptim/environments/models/sqeir_model.py:67: RuntimeWarning: overflow encountered in double_scalars
        dEdt = - (1-q)*beta0*c*S*I/N - E/De
      ../epidemioptim/environments/models/sqeir_model.py:69: RuntimeWarning: overflow encountered in double_scalars
        dE_qdt = - q*beta0*c*S*I/N - E_q/De
      ```  
      (분석4) 아니면 초기 숫자가 너무 작은 탓일 수도 있을 것 같다. 사실 나의 모델링에는 격리와 관련한 파라미터가 들어있는데 가져온 초기 설정 값은 첫 국내 코로나 확진자가 발생했을 당시이다. 초기값을 첫 거리두기 시작 날짜로 가져와 봐야겠다.  
      (시도) 아래와 같이 2020.3.23. 기준으로 초기값 설정했더니 그 아래와 같은 에러 발생함.  
      > self._all_initial_state_distribs = dict(E0=LogNormalDist(params=[2000, 2000 * self.noise], stochastic=self.stochastic),  
      >                                         I0=DiracDist(params=2850, stochastic=self.stochastic),  
      >                                         R0=DiracDist(params=3277, stochastic=self.stochastic),  
      >                                         S_q0=DiracDist(params=13261, stochastic=self.stochastic),  
      >                                         E_q0 = DiracDist(params=367, stochastic=self.stochastic),  
      >                                         I_q0 = DiracDist(params=5684, stochastic=self.stochastic),  
      >                                         )  
      
      ```
      ../epidemioptim/utils.py:223: RuntimeWarning: overflow encountered in exp
        return np.exp(float(samples)) if n == 1 else np.exp(samples)
      Traceback (most recent call last):
        File "train.py", line 78, in <module>
          train(**kwargs)
        File "train.py", line 39, in train
          model = get_model(model_id=params['model_id'],
        File "../epidemioptim/environments/models/get_model.py", line 22, in get_model
          return SqeirModel(**params)
        File "../epidemioptim/environments/models/sqeir_model.py", line 134, in __init__
          super().__init__(internal_states_labels=['S', 'S_q', 'E', 'E_q', 'I', 'I_q', 'R'],
        File "../epidemioptim/environments/models/base_model.py", line 46, in __init__
          self.reset()  # reset model to initial states and parameters
        File "../epidemioptim/environments/models/base_model.py", line 59, in reset
          self._sample_initial_state()
        File "../epidemioptim/environments/models/sqeir_model.py", line 184, in _sample_initial_state
          self.initial_state[k] = int(self.initial_state[k])
      OverflowError: cannot convert float infinity to integer
      ```  
      (해결) E0의 초기 분포 설정을 LogNormal이 아닌 DiracDist로 바꿔주니 (아래 참고) warning 없이 학습된다!  
      > self._all_initial_state_distribs = dict(E0=DiracDist(params=2000, stochastic=self.stochastic),  
      >                                         I0=DiracDist(params=2850, stochastic=self.stochastic),  
      >                                         R0=DiracDist(params=3277, stochastic=self.stochastic),  
      >                                         S_q0=DiracDist(params=13261, stochastic=self.stochastic),  
      >                                         E_q0 = DiracDist(params=367, stochastic=self.stochastic),  
      >                                         I_q0 = DiracDist(params=5684, stochastic=self.stochastic),  
      >                                         )  
      
      ```
      ---------------
      Episode: 50.00
            Best score so far: 293271.51
            Eval score: 293271.51
            Loss 1: 20793.33
            Loss 2: 776657024.00
            Train, Cost 1: 2188284.95
            Train, Cost 2: 503275715.94
            Train, Aggregated cost: 374186.80
            Eval, g: [0.25], C1: 2035818.49 +/- 549932.18, C2: 371467349.05 +/- 80806900.04, Agg: 145221.08 +/- 31645.99
            Eval, g: [0.5], C1: 2017006.52 +/- 496145.50, C2: 365690016.85 +/- 66374820.67, Agg: 282851.56 +/- 51407.27
            Eval, g: [0.75], C1: 2154528.71 +/- 444872.78, C2: 387925961.95 +/- 70776912.35, Agg: 448435.54 +/- 81815.00
            Eval, g: [0.], C1: 2062840.69 +/- 496083.70, C2: 390059974.69 +/- 62042188.92, Agg: 3173.60 +/- 763.21
            Eval, g: [1.], C1: 2117429.87 +/- 598995.93, C2: 381339247.97 +/- 97691072.32, Agg: 586675.77 +/- 150293.96

        ---------------
        Episode: 100.00
            Best score so far: 293271.51
            Eval score: 295775.05
            Loss 1: 9315.25
            Loss 2: 692999296.00
            Train, Cost 1: 2215952.72
            Train, Cost 2: 494826865.92
            Train, Aggregated cost: 411094.45
            Eval, g: [0.25], C1: 2151501.91 +/- 537339.90, C2: 381175938.12 +/- 88404445.23, Agg: 149088.63 +/- 34584.23
            Eval, g: [0.5], C1: 2078171.17 +/- 540452.68, C2: 368601745.08 +/- 85370647.86, Agg: 285138.40 +/- 66046.07
            Eval, g: [0.75], C1: 2099890.09 +/- 502277.52, C2: 376647052.57 +/- 84146273.41, Agg: 435400.40 +/- 97257.61
            Eval, g: [0.], C1: 2049168.68 +/- 474079.14, C2: 380721072.24 +/- 73204552.75, Agg: 3152.57 +/- 729.35
            Eval, g: [1.], C1: 2267234.22 +/- 533961.83, C2: 393961901.37 +/- 80761509.54, Agg: 606095.23 +/- 124248.48
      ```

<br/>

2/19 Fri.
- RL for COVID-19: train.py 실행 결과 확인 - 어제 하루 동안 돌린 결과를 plotting으로 통해 확인해보았다.
    - train 완료 터미널 화면:  
    <img src="https://user-images.githubusercontent.com/47997946/108446538-a2b4ef80-72a1-11eb-91c8-b78bd0f6d331.png" width="80%"></img>  
    
    - plotting 방법 모색 중: analysis/plots.py를 이용해보고자 함
        > $ python plots.py
        
        (문제1 & 해결) train.py와 달리 'from epidemioptim.~'로 모듈 불러오기가 안됨. 그래서 필요한 두 함수만 직접 복붙함.  
        (문제2) 아래와 같이 np.swapaxes 함수 사용 부분에서 dimension이 안맞는 에러가 나는 듯함.
        ```
        Traceback (most recent call last):
          File "plots.py", line 509, in <module>
            plot_multi_algo(RES_FOLDER)
          File "plots.py", line 406, in plot_multi_algo
            data = np.swapaxes(np.array(data), 0, 1)
          File "<__array_function__ internals>", line 5, in swapaxes
          File "/home/iot/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 594, in swapaxes
            return _wrapfunc(a, 'swapaxes', axis1, axis2)
          File "/home/iot/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 58, in _wrapfunc
            return bound(*args, **kwds)
        numpy.AxisError: axis2: axis 1 is out of bounds for array of dimension 1
        ```
        (분석) 아 찾아보니 이 함수는 알고리즘간 비교이다.. 내가 원하는 게 아니다..
    
    - plotting 방법 찾음: epidemioptim/analysis/Visualization EpidemiOptim.ipynb를 jupyter notebook을 통해 확인 가능
        - 우선 이전에 epidemioptim/analysis/notebook_utils.py을 바뀐 모델링에 맞게 수정함 (파일 path 등)
        - 실행 방법:  
            > (base) iot@Blade15:~/바탕화면/RL_COVID-19_Korea$ jupyter notebook
            이후 연결된 주피터 노트북의 epidemioptim/analysis/Visualization EpidemiOptim.ipynb 실행
        - 결과: 일단 beta = 0.5로 고정된 값(default)으로 무조건 결과 나타내게 해봄  
            <img src="https://user-images.githubusercontent.com/47997946/108475397-fb9d7b80-72d3-11eb-8f1b-d7c18789f274.png" width="70%"></img>
            * 고쳐야할 점: quarantine 수 범위, 베타 값에 따른 결과, 파일 path 정리  
            
    - beta 0~0.8까지 0.05 단위로 돌려놓음. (*나중에 0.0->0 폴더 이름 수정 필요)
    - 그리고 그래프가 다 넘어가서 최대치를 아래와 같이 바꿈 (순서대로 new deaths, total deaths, new quarantines, total quarantines) in notebook_utils.py
        > high = [10000, 100000, 25000000, 50000000]
    
