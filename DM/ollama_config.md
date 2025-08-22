# Ollama Configuration & Optimization Guide

## 🚀 Performance Optimization for MacBook Air M2

### 1. Environment Variables
```bash
# Add to your ~/.zshrc or ~/.bash_profile
export OLLAMA_HOST=localhost
export OLLAMA_ORIGINS=*
export OLLAMA_GPU_LAYERS=50
export OLLAMA_NUMA=true
export OLLAMA_MAX_MEMORY=6G
```

### 2. Ollama Configuration File
Create `~/.ollama/ollama.json`:
```json
{
  "host": "localhost:11434",
  "models": "/Users/yavin/.ollama/models",
  "gpu_layers": 50,
  "numa": true,
  "max_memory": "6G"
}
```

### 3. Start Ollama with Optimizations
```bash
# Stop current Ollama
pkill ollama

# Start with performance flags
ollama serve --gpu-layers 50 --numa --host localhost:11434

# Or use environment variables
OLLAMA_GPU_LAYERS=50 OLLAMA_NUMA=true ollama serve
```

## ⏱️ Timeout Configuration

### Current Timeout Settings
- **Generation API**: 120 seconds
- **Embeddings API**: 120 seconds  
- **Connection**: 10 seconds (for health checks)

### Recommended Timeouts for M2
- **Simple queries**: 60 seconds
- **Complex RAG queries**: 180 seconds
- **Large context**: 300 seconds

## 🔧 Troubleshooting Timeout Issues

### 1. Check Ollama Status
```bash
# Health check
curl -s http://localhost:11434/api/tags

# Check if running
ps aux | grep ollama
```

### 2. Monitor Performance
```bash
# Check memory usage
top -pid $(pgrep ollama)

# Check GPU utilization (if applicable)
system_profiler SPDisplaysDataType
```

### 3. Model Optimization
```bash
# Use quantized models for better performance
ollama pull phi3:latest:q4_K_M
ollama pull nomic-embed-text:q4_K_M

# Remove heavy models if not needed
ollama rm gpt-oss:20b
```

## 📊 Expected Performance on M2

### Model Performance
- **phi3:latest**: ~15-30 seconds for complex queries
- **nomic-embed-text**: ~2-5 seconds for embeddings
- **Memory usage**: 2-4GB typical

### Optimization Results
- **GPU acceleration**: 2-3x faster generation
- **NUMA optimization**: Better memory management
- **Quantized models**: 20-30% faster inference

## 🚨 Emergency Timeout Fixes

### If timeouts persist:
1. **Restart Ollama with optimizations**
2. **Reduce context length** in RAG queries
3. **Use smaller models** for testing
4. **Check system resources** (CPU, memory, disk)

### Quick Test
```bash
# Test basic generation
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3:latest", "prompt": "Hello", "stream": false}' \
  --max-time 60
```
