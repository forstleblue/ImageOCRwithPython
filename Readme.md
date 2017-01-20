Magic-box
=========

Example of connection written in python available at test.py


Api endpoints
=============
URL: magic.parionsendirect.fr

1) magic.parionsendirect.fr

2) magic.parionsendirect.fr/reconnect

3) magic.parionsendirect.fr/send_results

4) magic-test.parionsendirect.fr


magic.parionsendirect.fr
------------

1) Connect and send base64 encoded image

2) after connection sever sending json responses

3)  after connection

> {"status": "Connection open"}

4)  after message with base64-encoded image. Id of image can be used at /reconnect endpoint (to retrive data about this image)

> {"status": "Image processing", "id": "dbe1f1d6-11c0-42e1-818a-ca288473e9e5"}

5) after error

> {"status": "Error", "message": "Error message"}

6) after successful image processing. In response to one image can be one or more of these messages

> {"status": "Result", "reference": "127", "selection": "2", "id": "1/N/2", "match": "Hap. Beer Sheva-Olympiakos", "date": "09/09/2016"}

7) when image processing finished and all bets info sended to client

> {"status": "Finished"}

8) if date of first match not supported:
> {
    'status': 'Error',
    'message': 'Matches date not supported'
}

magic.parionsendirect.fr/reconnect # not ready
----------------------------------

1) after disconnect you can connect to this socket and send message with image id(such as "dbe1f1d6-11c0-42e1-818a-ca288473e9e5") and get data about image


magic.parionsendirect.fr/send_results
----------------------------------
Send results of parsed images to football fork


magic-test.parionsendirect.fr
----------------------------------
endpoint of magic box, which use test events csv files for using in the special test script


How to test magic-box locally
=============================
You can use the special test script with test images and prepared right results for them or use simple built-in the test script:

1) install tesseract and opencv (you forwarded these emails with instructions, so you should have these instructions)

2) pip install -r requirements.txt

3) create dir IMAGES_FOR_TEST inside magic-box folder and put all images(or folders with images) to these dir

4) run

> python main.py --port=8888

5) run

> python test.py

6) it will test all images from directory IMAGES_FOR_TEST


How to to run project on server
===============================

1) sudo pip install -r requirements.txt

2) sudo python main.py
