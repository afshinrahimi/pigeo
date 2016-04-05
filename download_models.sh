#!/bin/bash
#this command downloads the models.tar file and untars it in the current
#directory.
curl -L https://www.dropbox.com/s/gw8z0r5nq5ccok0/models.tar?dl=0 | tar x
#alternatively you can manually download the models.tar file from dropbox or
#this Google Drive address https://drive.google.com/file/d/0B9ZfPKPvp-JibDlLNTJnMnlQZ3c/view?usp=sharing
#copy it in the pigeo directory and extract it as it is.
#The final directory structure is:

#pigeo
#├── models
#│   ├── lpworld
#│   │   └── userhash_coordinate.pkl.gz
#│   └── lrworld
#│       ├── clf.pkl.gz
#│       ├── coordinate_address.pkl.gz
#│       ├── label_coordinate.pkl.gz
#│       └── vectorizer.pkl.gz
#├── params.py
#├── pigeo.py
#├── README.md
#├── static
#│   └── styles
#│       ├── bootstrap-3.3.6-dist
#│       │   ├── css
#│       │   │   ├── bootstrap.css
#│       │   │   ├── bootstrap.css.map
#│       │   │   ├── bootstrap.min.css
#│       │   │   ├── bootstrap.min.css.map
#│       │   │   ├── bootstrap-theme.css
#│       │   │   ├── bootstrap-theme.css.map
#│       │   │   ├── bootstrap-theme.min.css
#│       │   │   └── bootstrap-theme.min.css.map
#│       │   ├── fonts
#│       │   │   ├── glyphicons-halflings-regular.eot
#│       │   │   ├── glyphicons-halflings-regular.svg
#│       │   │   ├── glyphicons-halflings-regular.ttf
#│       │   │   ├── glyphicons-halflings-regular.woff
#│       │   │   └── glyphicons-halflings-regular.woff2
#│       │   └── js
#│       │       ├── bootstrap.js
#│       │       ├── bootstrap.min.js
#│       │       └── npm.js
#│       └── main.css
#├── templates
#│   ├── index.html
#│   └── index-simple.html
#└── twitterapi.py

