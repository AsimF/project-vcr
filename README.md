# project-vcr
mobile phone, vr application, android/iOS, hand recognition application

## Running environment
*you need graphic card to execute it.

## requirements
cudnn
cupy
opencv-python
requests
scipy
pyzmq

## How to learn
before you execute the python file, you need to download application "IP webcam" in google store.
In IP webcam, set the resolution as 480,320 and press start server.

in imagezmq-streaming/control_alg_server.py file, you need to modify the web address according to the server address shown in IP webcam app.

And finally execute the python file.
for server1, type python control_alg_server.py --server_id 1 
for server2, type python control_alg_server.py --server_id 2 

