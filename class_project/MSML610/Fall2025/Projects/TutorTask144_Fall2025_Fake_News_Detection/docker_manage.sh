#!/bin/bash

chmod +x docker_*.sh

echo "Docker Manager:"
echo "1. Build"
echo "2. Run"
echo "3. Stop"
echo "4. Restart"
echo "5. Logs"
echo "6. Clean"
echo "7. Status"
echo "8. Full Setup (Build & Run)"
echo "9. Exit"
echo ""

read -p "Choose option (1-9): " choice

case $choice in
    1) ./docker_build.sh ;;
    2) ./docker_run.sh ;;
    3) ./docker_stop.sh ;;
    4) ./docker_restart.sh ;;
    5) ./docker_logs.sh ;;
    6) ./docker_clean.sh ;;
    7)
        echo "Container Status:"
        docker-compose -f docker-compose.mcp.yml ps
        ;;
    8)
        ./docker_build.sh
        ./docker_run.sh
        ;;
    9) exit 0 ;;
    *) echo "Invalid option" ;;
esac
