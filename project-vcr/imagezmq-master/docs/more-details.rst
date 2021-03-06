===========================================================
More details about the multiple RPi video streaming example
===========================================================

The beginning of this documentation showed a screenshot of a Mac computer
displaying simultaneous video streams from 8 Raspberry Pi cameras:

.. image:: images/screenshottest.png

This display was generated by running ``timing_send_jpg_buf.py`` on 8
Raspberry Pi computers with PiCamera modules and running
``timing_receive_jpg_buf.py`` on a single Mac. Here is a more detailed
description of the test programs and how they work.

The Raspberry Pi computers send images and the Mac computer displays them. The
**imagezmq** python classes transport the images between the computers. There
can be a single Raspberry Pi sending images, or there can be 8 or more. There
is always exactly one computer (a Mac in this example) receiving and displaying
the images. In ZMQ parlance, the Raspberry Pi computers are acting as
clients sending requests (REQ) and the Mac is acting as the server sending back
replies (REP). Each "request" from the Raspberry Pi is a 2 part message
consisting of an OpenCV image and the Raspberry Pi hostname. Each "reply" sent
by the Mac message is an "OK" that tells the Raspberry Pi that it can send
another image.

ZMQ is a powerful messaging library that allows many patterns for sending and
receiving messages. **imagezmq** provides access to  **REQ/REP** and **PUB/SUP** ZMQ 
messaging patterns. 

REQ/REP messaging pattern
=========================

When using **REQ/REP** (request/reply) pattern every time a Raspberry Pi sends an image, it waits for an "OK"
from the Mac before sending another image. It also means that there can be multiple
Raspberry Pi computers sending messages to the Mac at the same time, since
the ZMQ REQ/REP pattern allows many clients to send REQ messages to a single
REP server. With each image sent, the Raspberry Pi sends an identifier (the
Raspberry Pi hostname, in these test programs), so that the Mac can display the
images from each Raspberry Pi in a different window.

Let's look at the Python code in the Raspberry Pi sending program:

.. code-block:: python
  :number-lines:

    # run this program on each RPi to send a labelled image stream
    import socket
    import time
    from imutils.video import VideoStream
    import imagezmq

    sender = imagezmq.ImageSender(connect_to='tcp://jeff-macbook:5555')

    rpi_name = socket.gethostname() # send unique RPi hostname with each image
    picam = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)  # allow camera sensor to warm up
    while True:  # send images as stream until Ctrl-C
        image = picam.read()
        sender.send_image(rpi_name, image)


Lines 2 to 5 import the Python packages we will be using. Line 7 instantiates
an **ImageSender** class from **imagezmq**. Line 9 sets **rpi_name** to the
hostname of the Raspberry Pi. This will keep each Raspberry Pi's image stream in
a separate window on the Mac (provided that each Raspberry Pi has a unique
hostname). Line 10 starts a VideoSteam from the PiCamera. Line 11 allows
the PiCamera sensor to warm up (if we grab the first frame from the PiCamera without
this warmup time, it will fail with an error). Line 12 begins a forever loop of
2 lines: Line 13 reads a frame from the PiCamera into the **image** variable.
Line 14 uses imagezmq's **send_image** method to send the Raspberry Pi hostname
and the image to the Mac. These "read and send" lines repeat until Ctrl-C is
pressed to stop the program. This effectively sends a continuous stream of images
(up to 32 images per second) to the Mac, with each image labeled with the hostname
of the Raspberry Pi that is sending it. If there are multiple Raspberry Pi
computers sending images at the same time, the Mac receiving the images is able
to sort them to labelled windows because of the unique Raspberry Pi hostname
sent with each image.

Now, lets look at the Python code on the Mac (or other display computer):

.. code-block:: python
  :number-lines:

    # run this program on the Mac to display image streams from multiple RPis
    import cv2
    import imagezmq

    image_hub = imagezmq.ImageHub()
    while True:  # show streamed images until Ctrl-C
        rpi_name, image = image_hub.recv_image()
        cv2.imshow(rpi_name, image) # 1 window for each RPi
        cv2.waitKey(1)
        image_hub.send_reply(b'OK')

Lines 2 and 3 import the Python packages we will be using: cv2 (OpenCV) and
**imagezmq**.  Line 5 instantiates an **ImageHub** class from **imagezmq**.
Line 6 begins a forever loop: line 7 receives an **rpi_name** and an **image**
from imagezmq's **recv_image** method. Line 8 shows the image in a display
window with a window title of **rpi_name**. Line 9 waits for a millisecond,
then line 10 sends the required "reply" back to the Raspberry Pi per the ZMQ
REQ/REP pattern. Lines 9 and 10 repeatedly receive and display images as they
come in. The ``cv2.imshow()`` method displays each image received in a window
corresponding to the window name. If all the images come from a single
**rpi_name**, then all the image streams will appear in a single window. But if
the income stream has images from multiple **rpi_name**'s, then ``cv2.imshow()``
automatically sorts the images by **rpi_name** into unique windows. Thus, if
3 Raspberry Pi computers are sending images, the images will be displayed in
3 separate windows with each one labelled by its **rpi_name**. The ZMQ library
is fast enough to make these 3 streams of images appear as 3 continuous video
streams in separate windows. To create the picture at the top of this page, 8
Raspberry Pi computers were sending images to a single Mac. The picture is a
screenshot of the Mac's display with the 8 ``cv2.imshow()`` windows arranged
in 2 rows.

PUB/SUB messaging pattern
=========================

The shown example that uses REQ/REP pattern has one important feature that can be a huge disadvantage at certain scenarios: sending images in this pattern is a blocking operation. 

This means that if a recipient stops responding or simply disconnects the sender will stop at the *send_image()* method until recipient reconnects. 

If this is unacceptabe in your application, you can use **PUB/SUB** (publish/subscribe) pattern. Subscribers can connects and disconnect to publisher (sender) at any time.

When using PUB/SUB mode image sender creates a ZMQ PUB socket, but images are pushed
to the socket only if at least one subscriber is connected to this socket. If
there is no subscribers images are discarded immediatelly and execution continues.

Lets check a simple example (the code of sender is pretty similar to the previous 
example):

.. code:: python
  :number-lines:

    # run this program on each RPi to send a labelled image stream
    import socket
    import time
    from imutils.video import VideoStream
    import imagezmq

    # Accept connections on all interfaces, port 5555
    sender = imagezmq.ImageSender(connect_to='tcp://*:5555', REQ_REP=False)

    rpi_name = socket.gethostname() # send RPi hostname with each image
    picam = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)  # allow camera sensor to warm up
    while True:  # send images as stream until Ctrl-C
        image = picam.read()
        sender.send_image(rpi_name, image)
        # The execution will continue even if nobody is connected to us
    
Mind different pattern for the ``connect_to`` argument and a new ``REQ_REP=False`` 
argument in line 8.

Server code

.. code-block:: python
  :number-lines:

    # run this program to receive and display frames
    # stream frames received from the camera to the browser
    import cv2
    import imagezmq

    # When there is a request from the web browser, create a subscriber 
    image_hub = imagezmq.ImageHub(open_port='tcp://192.168.0.100:5555', REQ_REP=False)
    while True:  # show streamed images
        rpi_name, image = image_hub.recv_image()
        cv2.imshow(rpi_name, image) # 1 window for each RPi 
        cv2.waitKey(1)

The reciever part is very similar to **REQ/REP** example, however there are defferences:

* Line 7: We have to know IP address of the sender to connect to it. In REQ/REP case the direction of connection was opposite - the sender had to know address of the recipient. Also, we use *REQ_REP=False* parameter here.
* Line 12: There is no one as we don't have to send reply to sender :)


`Return to main documentation page README.rst <../README.rst>`_
