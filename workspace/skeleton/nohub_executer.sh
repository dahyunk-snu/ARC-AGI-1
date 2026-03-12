# nohub으로 arc_trainer_Moe.py를 실행하고 동명의 log 파일을 생성합니다.
#!/bin/bash
nohup python3 -u arc_trainer_MoE.py > arc_trainer_MoE_1-2-5.log 2>&1 &
# ps -eo pid,stime,etime,cmd | grep '[p]ython3 -u arc_trainer_MoE.py
# pkill -f arc_trainer_MoE.py