curl -X POST "http://localhost:1996/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Hello, how are you?"}],
    "max_tokens": 2
  }'