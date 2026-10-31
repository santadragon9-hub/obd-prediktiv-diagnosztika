#!/bin/bash
echo "=========================================="
echo "  Szakdolgozat Adatgyűjtő Indítása "
echo "=========================================="

echo "Bluetooth port csatlakoztatása..."

sudo rfcomm bind 0 13:E0:2F:8D:58:E1 2>/dev/null

sleep 2

echo "Adatgyűjtés indítása..."
/home/pi/szakdolg_env/bin/python3 /home/pi/szakdolgozat/seat_logger.py

echo ""
echo "=========================================="
echo "A program leállt. A kilépéshez nyomj egy ENTER-t..."
read
