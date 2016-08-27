================
``pigeo`` readme
================

Introduction
------------

``pigeo`` is a document or Twitter user geolocation tool. Given a piece of text or a Twitter user, it can predict their locations based on pre-trained models.

A screencast of how ``pigeo`` works is available on Youtube: https://www.youtube.com/watch?v=QiV_ow7k2J4 

The design principles are as follows:

1. Lightweight and fast
2. Comes with text-based classification and network-based regression pre-trained models.
3. It is possible to train new text-based classification models.
4. It can be used in shell mode, web mode powered by Python Flask and as a library.
5. It supports informal text.
6. It's performance is evaluated over a standard Twitter geolocation dataset.

I try to keep the web-based app. Online here at http://bit.do/pigeo if it wasn't Online and you needed it
for testing the system you can contact me or easily bring up the server on your own machine using the instructions
below.


Quick Start
-----------

``pigeo``'s installation is straightforward:


1. download the zip file from github or run: ``git clone http://github.com/afshinrahimi/pigeo.git``

2. ``cd pigeo`` then ``chmod +x download_models.sh`` and then run ``./download_models.sh``.

   This downloads the pre-trained models and extracts them in models directory. alternatively 
   (e.g. if you are using Windows) you can download the models directory from https://www.dropbox.com/s/gw8z0r5nq5ccok0/models.tar?dl=0 and extract it with an archive program.

3. Requirements:
	
	3.0 ``sudo pip install -r requirements.txt`` and go to 4 install the libraries in requirements.txt
	one by one.
	
	Note: if you don't have root permission and can not run with sudo you can use pip with --user argument.

4. Set the Twitter keys and tokens in params.py. If you don't have your own Twitter credentials (keys and tokens) you can create one from http://apps.twitter.com. ``pigeo`` needs the Twitter credentials in order to geolocate Twitter users (e.g. @potus) otherwise it won't be able to download the user's tweets and won't be able to geolocate them. Text input though, will work without the Twitter credentials. 

5. ``pigeo`` is ready to use. Go to usage section.


Directory Structure
-------------------

The directory structure after installation and downloading the models should be:

```
.
├── download_models.sh
├── models
│   ├── lpworld
│   │   └── userhash_coordinate.pkl.gz
│   └── lrworld
│       ├── clf.pkl.gz
│       ├── coordinate_address.pkl.gz
│       ├── label_coordinate.pkl.gz
│       └── vectorizer.pkl.gz
├── params.py
├── pigeo.py
├── README.md
├── static
│   └── styles
│       ├── bootstrap-3.3.6-dist
│       │   ├── css
│       │   │   ├── bootstrap.css
│       │   │   ├── bootstrap.css.map
│       │   │   ├── bootstrap.min.css
│       │   │   ├── bootstrap.min.css.map
│       │   │   ├── bootstrap-theme.css
│       │   │   ├── bootstrap-theme.css.map
│       │   │   ├── bootstrap-theme.min.css
│       │   │   └── bootstrap-theme.min.css.map
│       │   ├── fonts
│       │   │   ├── glyphicons-halflings-regular.eot
│       │   │   ├── glyphicons-halflings-regular.svg
│       │   │   ├── glyphicons-halflings-regular.ttf
│       │   │   ├── glyphicons-halflings-regular.woff
│       │   │   └── glyphicons-halflings-regular.woff2
│       │   └── js
│       │       ├── bootstrap.js
│       │       ├── bootstrap.min.js
│       │       └── npm.js
│       └── main.css
├── templates
│   ├── index.html
│   └── index-simple.html
└── twitterapi.py
```

Usage
-----


```
usage: pigeo.py [-h] [--model MODEL] [--dump_dir DUMP_DIR] [--host HOST]
                [--port PORT] [--mode MODE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -d MODEL
                        text-based classification model directory to be used.
                        default(./models/lrworld)
  --dump_dir DUMP_DIR, -o DUMP_DIR
                        directory to which a newly trained model is saved.
                        default(./models/test_model)
  --host HOST           host name/IP address where Flask web server in web
                        mode will be running on. Set to 0.0.0.0 to make it
                        externally available. default (127.0.0.1)
  --port PORT, -p PORT  port number where Flask web server will bind to in web
                        mode. default (5000).
  --mode MODE, -m MODE  mode (web, shell) in which pigeo will be used. default
                        (shell).
```


Shell Mode
----------

This mode is well suited for initial testing of ``pigeo``.
Simply type ``python pigeo.py`` or ``python pigeo.py --mode shell`` to open the shell mode.
You can type a string or a single Twitter user to be geolocated.

``text to geolocate: yall``
and the result is:

```
{'city': u'Atlanta', 'state': u'Georgia', 'lat': 33.749000000000002, 'country': u'United States of America', 'lon': -84.387979999999999, 'label_distribution': {180: 0.063345477875493147}, 'top50': u'atlanta, newnan, atl, _madeinchyna, auc, spelman, emory, nisha_pooh_, mcdonough, lenox, redan, glambarsalon, morehouse, stockbridge, llh, riverdale, scoutmob, llf, marta, ladycaliibaybee, buckhead, georgia, ga, culc, decatur, cl_atlanta, coweta, peachtree, piedmont, colemankjohnson, cau, obsessions, followmeh, \uc9c0\uc6b0\uac1c, frfr, \uba38\ub9ac\uc18d\uc5d0, stonecrest, creekside, welcometoatlanta, lithonia, octane, duress, midtown, jortstorture, falcons, wpatl, a3c, criminalrecords, fairburn, frankski'}
```
The result is a json string which contains city, state, country and coordinates of the predicted location. It also contains the predicted class and its confidence.
Note that the LR-WORLD model has 930 classes/regions. The top 50 most important features of the predicted class are also returned.

Web Mode
--------

The web mode is powered by Flask which is a lightweight Python web framework.
To start the webservice simply run

``python pigeo.py --mode web --host 127.0.0.1 --port 5000``

If you want the web service to be available on the valid IP address run:

``python pigeo.py --mode web --host 0.0.0.0 --port 5000``

Use ``http://127.0.0.1:5000`` or ``http://valid-ip-address:5000`` in the browser to use the service.
The service is able to geolocate a piece of text or a single Twitter user (e.g. @potus).

Library Mode
------------

pigeo is well suited to be used in other python programs.
In the library mode it is possible to geolocate a single piece of text
and also a list of text documents. Simple use case:

```
import pigeo
# loads the world model (default)
pigeo.load_model()
# geolocate a sentence
pigeo.geo("gamble casino city")
# geolocate a Twitter user
pigeo.geo('@POTUS')
# geolocate a list of texts
pigeo.geo(['city centre', 'city center'])
```

Note that it is not efficient to call pigeo.geo multiple times
and the suggested way for geolocation of multiple documents is
passing them as a list to pigeo.geo. It is possible to return
the label distribution over all classes rather than only the predicted
class by calling ``pigeo.get('a text', True)``.


Training Mode
-------------

To train a new geolocation model one needs
a list of text and a list of corresponding
coordinates. pigeo then is able to train a
new model and save it as follows:

```
import pigeo
# train the model and save it in 'toy_model'
pigeo.train_model(['text1', 'text2'], 
[(lat1, lon1), (lat2, lon2)], num_classes=2, 
model_dir='toy_model')
# the new model can be loaded and be used
pigeo.load_model(model_dir='toy_modle')
```

LP Network-based Regression Mode
--------------------------------

pigeo has a trained network-based regression model that
can geolocate only Twitter user which requires both an
Internet connection and the tweepy library installed.

```
import pigeo
# load lpworld
pigeo.load_lpworld()
# geolocate a Twitter user (Internet neeeded).
pigeo.geo_lp('@potus')
```

Version
-------
Python version 2.7

scikit-learn version 0.17

pickle Revision: 72223

Numpy version 1.10.4

Note that the models might not be easily loadable by
other pickle/scikit-learn versions.

Contact
-------
Afshin Rahimi <afshinrahimi@gmail.com>
