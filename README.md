# project-vcr
mobile phone, vr application, android/iOS, hand recognition application

## Running environment
*you need graphic card to execute it.

## requirements
cudnn <br />
cupy<br />
opencv-python<br />
requests<br />
scipy<br />
pyzmq

## How to learn
before you execute the python file, you need to download application "IP webcam" in google store.<br />
In IP webcam, set the resolution as 480,320 and press start server.<br />
<br />
in imagezmq-streaming/control_alg_server.py file, you need to modify the web address according to the server address shown in IP webcam app.<br />
<br />
And finally execute the python file.<br />
for server1, type python control_alg_server.py --server_id 1 <br />
for server2, type python control_alg_server.py --server_id 2 <br />

