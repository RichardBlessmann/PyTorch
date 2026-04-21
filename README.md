Hallo Marie, falls du das liest ich habe hier ein paar Quellen, wo ich teile des Codes her habe
es ist zum Teil aus den vorherigen Testruns und tutorials entstanden zum teils von random quellen und googlen und zum teilen aus chatgpt

ich hab versucht üüüüberall Kommentare hinzumachen, damit du und ich es wieder nachvollziehen kiönnen was wie wo macht

Genannte QUellen:

	1. PyTorch official RL example (Policy Gradient)
		[https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html]
	2. OpenAI SpinningUp
		[https://spinningup.openai.com/en/latest/]
	3. Stable-Baselines3 docs
		[https://stable-baselines3.readthedocs.io/en/master/]


Das ist erstmal die erste Demo Version und die obstacles im Feld habe ich erstmal noch ausgelassen - sind aber schon programmiert, sodass es theoretisch möglich wäre sie problemlos einzufügen... 
sind in der Datei train_env.py vor den Trainingsepisoden als 
[#obst.spawn_curved_tower_course(start, goal, client=env.CLIENT, difficulty=1)] auskommentiert.


	
for install of gym-pybullet-drones on windows side:

1. verify that you have python in cmd/powershell

	python --version

be in wanted project base-directory:
2. Create Virtual Environment:

	python -m venv drones-env
	drones-env\Scripts\activate

3. Upgrade packaging tools

	python -m pip install --upgrade pip setuptools wheel

4. Clone the repository

	git clone https://github.com/utiasDSL/gym-pybullet-drones.git
	cd gym-pybullet-drones

5. Install dependencies

	pip install numpy scipy matplotlib pandas
	pip install pybullet gym

6. Install project

	pip install -e .
	
	
7. Test the install

	python example\fly.py
or	python example\learn.py
	
	
