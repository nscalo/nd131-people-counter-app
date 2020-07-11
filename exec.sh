max=300
for i in `seq 2 $max`
do
    sleep .2
    chromium-browser --headless --disable-gpu --screenshot="screenshot_$1.png" --no-sandbox "http://localhost:3000"
done
