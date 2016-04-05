================
``pigeo`` readme
================

Introduction
------------

``pigeo`` is a document or Twitter user geolocation tool. Given a piece of text or a Twitter user, it can predict their locations based on pre-trained models.

The design principles are as follows:

1. Lightweight and fast
2. Comes with text-based classification and network-based regression pre-trained models.
3. It is possible to train new text-based classification models.
4. It can be used in shell mode, web mode powered by Python Flask and as a library.
5. It supports informal text.
6. It's performance is evaluated over a standard Twitter geolocation dataset.

Installation
------------

``pigeo``'s installation is straightforward:
1. download the zip file from github or run:
    
    git clone http://github.com/afshinrahimi/pigeo.git

2. sdfsd

    cd pigeo
    
    chmod +x download_models.sh
    
    ./download_models.sh #this downloads the pretrained models and extracts them in models. alternatively you can do it manually.
