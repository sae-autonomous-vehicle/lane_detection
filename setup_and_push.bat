@echo off
git init
git remote add origin https://github.com/sae-autonomous-vehicle/lane_detection.git
git add .
git commit -m "feat: Initial commit of lane detection project"
git push -u origin master

echo.
echo ---
echo Done! If you saw an error like "src refspec master does not match any", your default branch might be named 'main'.
echo If so, please run this command manually: git push -u origin main
pause
