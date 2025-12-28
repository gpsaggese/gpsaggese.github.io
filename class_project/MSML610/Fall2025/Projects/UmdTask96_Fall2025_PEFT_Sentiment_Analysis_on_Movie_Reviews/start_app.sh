#!/bin/bash

echo "🚀 Starting Fake News Detector Application..."
echo ""

# Start docker-compose in detached mode
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
echo ""

# Wait for backend to be healthy
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting for backend... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Backend failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Wait for Streamlit to be ready
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "✅ Streamlit is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting for Streamlit... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Streamlit failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "🎉 Application is ready!"
echo ""
echo "📱 Opening Streamlit app in your browser..."
echo "   URL: http://localhost:8501"
echo ""

# Open browser based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8501
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:8501 2>/dev/null || echo "Please open http://localhost:8501 in your browser"
else
    # Windows (Git Bash/WSL)
    start http://localhost:8501 2>/dev/null || echo "Please open http://localhost:8501 in your browser"
fi

echo ""
echo "📊 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
echo ""
