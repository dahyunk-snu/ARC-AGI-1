# 1) 로그 저장 디렉토리 생성
mkdir -p logs

# 2) for-loop + nohup + & 로 10개 작업 동시 실행
for file in generated_code/*.jsonl; do
  base=$(basename "$file" .jsonl)               # ex: generated_code/foo.jsonl → foo
  nohup python generate_problems.py \
    --jsonl "$file" \
    --outdir generated_problems \
    > logs/${base}.log 2>&1 &                   # 표준출력·표준에러를 logs/foo.log 에 기록
done

# 3) (선택) 현재 띄운 백그라운드 잡 확인
jobs

# 4) 로그 내용 실시간 보기
#    - 특정 파일: tail -f logs/foo.log
#    - 전체 로그:  tail -f logs/*.log

# pkill -f generate_problems.py