# clean previous eval directory
rm -rf /workspace/evaluate/*

# emulate submission process
cp -r /home/student/workspace/arc /workspace/evaluate
cp -r /home/student/workspace/artifacts /workspace/evaluate
cp /home/student/workspace/setup.sh /workspace/evaluate

cd /workspace/evaluate
source /workspace/evaluate/setup.sh
cp /workspace/scripts/evaluate.py /workspace/evaluate
conda run -n $EVAL_ENV python evaluate.py
