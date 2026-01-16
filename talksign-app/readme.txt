HOW TO RUN TALKSIGN
===================

0. Prerequisites (Drivers):
   - Virtual Mic: Download & Install "VB-CABLE Driver" (https://vb-audio.com/Cable/)
   - Virtual Camera: Download & Install "OBS Studio" (https://obsproject.com/)
   * Restart your computer after installing these drivers.

1. First Time Setup (Install Dependencies):
   cd talksign-app
   pip install -r backend/requirements.txt
   npm install

2. Open Terminal #1 (Backend):
   cd talksign-app
   python server.py

3. Open Terminal #2 (Frontend):
   cd talksign-app
   npm start

Note: Keep both terminals open while using the app.