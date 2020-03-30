# Status report6
#### Mark Yu
#### yum4@rpi.edu
#### 661845699
#### March 27 2020

## What I Have Done
* I have read the paper of RLcard game and found out that their winning rate of dqn agent on UNO is close to my modified version, so I want to try to see what will happen if I let my dqn agent plays against itself. The python code and results in this folder [https://github.com/ymkymkymkymx/Some-useless-codes/tree/master/statusupdate6](https://github.com/ymkymkymkymx/Some-useless-codes/tree/master/statusupdate6), to run it you need to first install RLcard by '''pip install rlcard'''.
* I have modified the  agent trainning part to let it train two dqn agent at the same time and let them play against each others and, as expected, they both got the win rate floating around 0.5 so I guess my model is working.
## What's next
* I will try to implement the save model option so that I can save the models after trainning. I will also try to see the behaviour of the models by playing agaisnt it.


## Anything blocking
* If you can finish my other courses' homework for me, that will be a great help! 
* I don't know how many layers and nodes are approximately needed for the game with 420 inputs and 60 outputs.


* A link to this page: [https://github.com/ymkymkymkymx/Some-useless-codes/blob/master/statusupdate6/statusupdate6_MAR_27.md](https://github.com/ymkymkymkymx/Some-useless-codes/blob/master/statusupdate6/statusupdate6_MAR_27.md)