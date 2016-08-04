'''
Created on 27 Mar 2016

@author: af
'''
import argparse
from flask import Flask, jsonify, render_template, request
import gzip
import logging
import numpy
import os
import sys
import pickle
import hashlib
from scipy.sparse import csr_matrix, coo_matrix
import re
import pdb
import params


# Flask is a lightweight Python web framework based on Werkzeug and Jinja 2. 
# import global variables
app = Flask(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def get_coordinates(label):
	""" given a label returns the coordinates associated with that label/region.
	The coordinates are precomputed from training points. The coordinate of a label equals to the median
	latitude and longitude of all training points in that region.
	
	Args:
		label (int): label or region id associated with a class
	Returns:
		lat, lon associated with that region computing by taking the median lat,lon of training points in that region.
	"""
	
	lat, lon = params.label_coordinate[label]
	return lat, lon

def get_label(lat, lon):
	""" given a lat, lon (float) find the class label of the model closest to the coordinate.
	
	Args:
		lat (float): latitude of the location
		lon (float): longitude of the location
	Returns:
		label (int) the closest class id (region) to the given coordinates.
	"""
	from haversine import haversine
	#compute all the distances from the given point to the median point of every class
	label_distances = {label:haversine((lat, lon), coordinate) for label, coordinate in params.label_coordinate.iteritems()}
	#choose the label with minimum distance
	label = min(label_distances, key=label_distances.get)
	return label
		

def get_topk_features(label, topk=50):
	"""
	given a label (str) return the top k important features as a list
	"""
	topk_feature_indices = numpy.argsort(params.clf.coef_[label].toarray())[0,-topk:].tolist()[::-1]
	topk_features = [params.vectorizer.features[i] for i in topk_feature_indices]
	topk_features = [f for f in topk_features if 'user_' not in f]
	return topk_features

def get_location_info(label):
	"""
	given a label (str) returns a dictionary containing information about the corresponding location.
	"""
	lat, lon = get_coordinates(label)
	location = params.coordinate_address[(lat, lon)] 
	country = location['address'].get('country', '')
	state = location['address'].get('state', '')
	city = location['address'].get('city', '')	 
	return {"lat":lat, "lon":lon, "country":country, "state":state, "city":city}

def retrieve_location_from_coordinates(points):
	"""Given a list of coordinates, uses the geocoder and finds the corresponding
	locations and returns them in a dictionary. It requires internet connection
	to retrieve the addresses from openstreetmap.org. The geolocation service provider
	can be changed according to geopy documentation.
	
	Args:
		points (iterable): an iterable (e.g. a list) of tuples in the form of (lat, lon) where coordinates are of type float e.g. [(1.2343, -10.239834r),(5.634534, -12.47563)]
	Returns:
		co_loc (dict): a dictionary of the form point: location where location is itself a dictionary including the corresponding address of the point.
	
	"""
	from geopy.geocoders import Nominatim
	geocoder = Nominatim(timeout=10)
	coordinate_location = {}
	
	for coordinate in points:
		try:
			location = geocoder.reverse(coordinate)
		except:
			location = 'unknown'
		coordinate_location[coordinate] = location
	co_loc = {k:v.raw for k,v in coordinate_location.iteritems()}
	
	return co_loc

@app.route('/', methods=['GET', 'POST'])
def index():
	'''
	RESTful API
	index.html page provides a simple web UI that given a text, geolocates it and puts a marker on a Google map.
	'''
	return render_template('index.html')

@app.route('/geo', methods=['GET', 'POST'])
def geo_web():
	'''
	RESTful API
	given a piece of text, vectorize it, classify it into one of the regions using clf (a pre-trained classifier) and return a json which has info about the predicted location(s).
	'''
	text = request.args['text']
	if isinstance(text, list) or isinstance(text, tuple) or len(text) == 0:
		return
	result = None
	try:
		result = geo(text, return_lbl_dist=False, topk=3)
	except:
		return
	return jsonify(**result)
	
@app.route('/features', methods=['GET', 'POST'])
def geo_features():
	'''
	RESTful API
	given a piece of text, vectorize it, classify it into one of the regions using clf (a pre-trained classifier) and return a json which has info about the predicted location(s).
	'''
	lat = request.args['lat']
	lon = request.args['lon']
	features = []
	if lat and lon:
		label = get_label(float(lat), float(lon))
		features = get_topk_features(label, topk=100)
		location_info = get_location_info(label)
		#should we set the marker on the cluster or the clicked coordinates?
		#location_info['lat'] = lat
		#location_info['lon'] = lon
	features = ', '.join(features)
	result = {'topk': features}
	result.update(location_info)
	return jsonify(**result)


def geo(text, return_lbl_dist=False, topk=1):
	"""
	given a piece of text (str/unicode), vectorize it, classify it into one of the regions using 
	clf (a pre-trained classifier) and return a json which has info about the predicted location(s).
	If the input is a list of texts it calls geo_iterable.
	
	Efficiency Note: It is not efficient to call geo(text) several times. The best is to call it with a list of texts
	as input.
	
	Args:
		text (str/unicode): a string which should be geolocated. It can be a piece of text or one single Twitter screen name e.g. @user.
		return_lbl_dist: if True returns the probability distribution over all the classes.
		topk (int): default(1), if higher than 1, return the top K locations ordered by classifier's confidence.
	Returns:
		a dictionary containing the predicted geolocation information about text.
	"""
	if not text:
		return
	#if text is a list of strings
	if isinstance(text, list) or isinstance(text, tuple):
		return geo_iterable(text, return_lbl_dist)
	#check if text is a single Twitter screen
	if text[0] == '@' and len(text.split()) == 1:
		return geo_twitter(text, return_lbl_dist)
	#supports only 1 text sample
	test_samples = [text]
	X_test = params.vectorizer.transform(test_samples)
	#probability distribution over all labels
	label_distribution = params.clf.predict_proba(X_test)

	if return_lbl_dist:
		label_distribution = coo_matrix(label_distribution)
		label_distribution_dict = {}
		for lbl in range(0, label_distribution.shape[1]):
			label_distribution_dict[lbl] = label_distribution[0, lbl]
	elif topk>1 and topk<=label_distribution.shape[1]:
		topk_labels = numpy.argsort(label_distribution)[0][::-1][:topk].tolist()
		topk_probs = [label_distribution[0, i] for i in topk_labels]
		topk_label_dist = dict(zip(topk_labels, topk_probs))
		topk_locationinfo = {}
		for i, lbl in enumerate(topk_labels):
			location_info = get_location_info(lbl)
			topk_locationinfo['lat' + str(i)] = location_info['lat']
			topk_locationinfo['lon' + str(i)] = location_info['lon']
			topk_locationinfo['city' + str(i)] = location_info['city']
			topk_locationinfo['state' + str(i)] = location_info['state']
			topk_locationinfo['country' + str(i)] = location_info['country']
			
	pred = numpy.argmax(label_distribution)
	confidence = label_distribution[0, pred]

	top50_features = ', '.join(get_topk_features(pred))
	location_info = get_location_info(pred)
	if return_lbl_dist:
		result = {'top50':top50_features, 'label_distribution':label_distribution_dict}
	elif topk>1 and topk<=label_distribution.shape[1]:
		result = {'top50':top50_features, 'label_distribution':topk_label_dist}
		result.update(topk_locationinfo)
	else:
		result = {'top50':top50_features, 'label_distribution':{pred:confidence}}
	result.update(location_info)
	logging.debug(result)
	return result

def geo_iterable(texts, return_lbl_dist=False):
	"""
	given an iterable (e.g. a list) of texts (str/unicode), vectorize them, classify them into one of the regions using clf 
	(a pre-trained classifier) and returns results a list of dictionaries with the same order as texts corresponding 
	to each text item with info about the predicted location(s).
	
	Args:
		texts (list/tuple): a list of strings/unicodes which should be geolocated.
		return_lbl_dist: if True returns the probability distribution over all the classes.
	Returns:
		a dictionary containing the predicted geolocation information about text.
	"""
	results = []
	#supports only 1 text sample
	test_samples = texts
	num_samples = len(test_samples)
	X_test = params.vectorizer.transform(test_samples)
	#probability distribution over all labels
	label_distributions = params.clf.predict_proba(X_test)
	for i in range(num_samples):
		#probability distribution over all labels
		label_distribution = label_distributions[i]
		if return_lbl_dist:
			label_distribution = coo_matrix(label_distribution)
			label_distribution_dict = {}
			for j, lbl, prob in zip(label_distribution.row, label_distribution.col, label_distribution.data):
				label_distribution_dict[lbl] = prob
			label_distribution = label_distribution.toarray()
		
		pred = numpy.argmax(label_distribution)
		confidence = label_distribution[pred]
		top50_features = ', '.join(get_topk_features(pred))
		location_info = get_location_info(pred)
		if return_lbl_dist:
			result = {'top50':top50_features, 'label_distribution':label_distribution_dict}
		else:
			result = {'top50':top50_features, 'label_distribution':{pred:confidence}}
		result.update(location_info)
		results.append(result)
	return results

def geo_twitter(twitter_screen_name, return_lbl_dist=False):
	"""
	given a twitter id or screen_name, retrieves the top 100 tweets of the user, extracts the text, vectorizes it, classifies it into one of the regions using 
	clf (a pre-trained classifier) and returns a json which has info about the predicted location(s).
	Note that internet connection is required and authentication information should be set in twitterapi.py.
	
	Args:
		twitter_screen_name (str): Twitter user id or screen_name
	Returns:
		a dictionary including information about the predicted location of the user given the content of their tweets.
	"""
	from twitterapi import download_user_tweets
	timeline = []
	timeline = download_user_tweets(twitter_screen_name, count=100)
	if timeline:
		text = ' '.join([t.text for t in timeline])
	else:
		text = ' '
	return geo(text, return_lbl_dist)
	

def dump_model(clf, vectorizer, co_loc, label_coordinate, model_dir):
	"""
	Dumps the model into a directory. Each component of the model is pickled and gzipped.
	"""
	logging.info('dumping coordinate city mappings...')	
	with gzip.open(os.path.join(model_dir, 'coordinate_address.pkl.gz'), 'wb') as outf:
		pickle.dump(co_loc, outf)
	logging.info('dumping label_lat, label_lon...')	
	with gzip.open(os.path.join(model_dir, 'label_coordinate.pkl.gz'), 'wb') as outf:
		pickle.dump(label_coordinate, outf)
	
	logging.info('dumping vectorizer...')	
	with gzip.open(os.path.join(model_dir, 'vectorizer.pkl.gz'), 'wb') as outf:
		pickle.dump(vectorizer, outf)
	logging.info('dumping the trained classifier...')	
	with gzip.open(os.path.join(model_dir, 'clf.pkl.gz'), 'wb') as outf:
		pickle.dump(clf, outf)
	
def train_model(texts, points, num_classses, model_dir, text_encoding='utf-8'):
	""" Given an iterable of (text, lat, lon) items, cluster the points into #num_classes and use
	them as labels, then extract unigram features, train a classifier and save it in models/model_name
	for future use. 

	Args:
	texts -- an iterable (e.g. a list) of texts e.g. ['this is the first text', 'this is the second text'].
	points -- an iterable (e.g. a list) of tuples in the form of (lat, lon) where coordinates are of type float e.g. [(1.2343, -10.239834r),(5.634534, -12.47563)]
	num_classes -- the number of desired clusters/labels/classes of the model.
	model_name -- the name of the directory within models/ that the model will be saved.
	"""
	
	if os.path.exists(model_dir):
		logging.error("Model directory " + model_dir + " already exists, please try another address.")
		sys.exit(-1)
	else:
		os.mkdir(model_dir)
	
	from sklearn.cluster import KMeans
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.linear_model.stochastic_gradient import SGDClassifier
	
	kmeans = KMeans(n_clusters=num_classses, random_state=0)
	points_arr = numpy.array(points)
	kmeans.fit_transform(points_arr)
	cluster_centers = kmeans.cluster_centers_
	sample_clusters = kmeans.labels_
	label_coordinate = {}
	for i in range(cluster_centers.shape[0]):
		lat, lon = cluster_centers[i, 0], cluster_centers[i, 1]
		label_coordinate[i] = (lat, lon)
	
	logging.info('extracting features from text...')
	vectorizer = TfidfVectorizer(encoding=text_encoding, stop_words='english', ngram_range=(1,1), max_df=0.5, min_df=0, binary=True, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
	X_train = vectorizer.fit_transform(texts)
	Y_train = sample_clusters
	vectorizer.stop_words_ = None
	logging.info('the number of samples is %d and the number of features is %d' % (X_train.shape[0], X_train.shape[1]))
	
	logging.info('training the classifier...')
	logging.warn('Note that alpha (regularisation strength) should be tuned based on the performance on validation data.')
	clf = SGDClassifier(loss='log', penalty='elasticnet', alpha=5e-5, l1_ratio=0.9, fit_intercept=True, n_iter=5, n_jobs=2, random_state=0, learning_rate="optimal")
	clf.fit(X_train, Y_train)
	clf.coef_ = csr_matrix(clf.coef_)
	
	logging.info('retrieving address of the given points using geopy (requires internet access).')
	coordinate_address = retrieve_location_from_coordinates(points)

	logging.info('dumping the the vectorizer, clf (trained model), label_coordinates and coordinate_locations into pickle files in ' + model_dir)
	dump_model(clf, vectorizer, coordinate_address, label_coordinate, model_dir)
	


def load_lpmodel(lpmodel_file='./models/lpworld/userhash_coordinate.pkl.gz'):
	"""
	loads hashed_user: coordinate dictionary if it is not already loaded in params.hasheduser_coordiante.
	"""
	if not params.lp_model_loaded:
		with gzip.open(lpmodel_file, 'rb') as inf:
			params.userhash_coordinate = pickle.load(inf)
		params.lp_model_loaded = True

def geo_lp(twitter_user,return_address=False, lpmodel_file='./models/lpworld/userhash_coordinate.pkl.gz'):
	""" Given a twitter user, the timeline of the user is downloaded,
	the @-mentions are extracted and an ego graph for the user is built.
	The neighbors (@-mentions) of the user are matched with geolocated
	users of the WORLD dataset (Han et. al, 2012) and their locations
	are set if any matches are found.
	The user is then geolocated using the locations of its neighbours.
	The geolocation algorithm is based on real-valued label propagation (Rahimi et. al, 2015).
	
	Args:
		twitter_user (str): a Twitter screen/id.
		return_address : if True the predicted coordinates are mapped to an address using geopy. Default (False).
	Returns:
		a dictionary with location information
	"""
	load_lpmodel(lpmodel_file)
	from twitterapi import download_user_tweets
	timeline = download_user_tweets(twitter_user, count=200)
	text = ' '.join([t.text for t in timeline])
	#pattern for @mentions
	token_pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
	token_pattern = re.compile(token_pattern)
	mentions = [m.lower() for m in token_pattern.findall(text)]
	mention_hashes = [hashlib.md5(m).hexdigest() for m in mentions]
	neighbour_coordinates = []
	for mention_hash in mention_hashes:
		if mention_hash in params.userhash_coordinate:
			coordinate = params.userhash_coordinate[mention_hash]
			neighbour_coordinates.append(coordinate)
	
	#no match found, unable to geolocate
	if len(neighbour_coordinates) == 0:
		return
	
	latitudes = [coor[0] for coor in neighbour_coordinates]
	longitudes = [coor[1] for coor in neighbour_coordinates]
	median_lat = numpy.median(latitudes)
	median_lon = numpy.median(longitudes)
	co_loc = {}
	if return_address:
		co_loc = retrieve_location_from_coordinates(points=[(median_lat, median_lon)])
	result = {'lat':median_lat, 'lon':median_lon, 'address':co_loc.get((median_lat, median_lon), {})}
	logging.debug(result)
	return result
			
def geo_lp_iterable(twitter_users,return_address=False, lpmodel_file='./models/lpworld/userhash_coordinate.pkl.gz'):
	""" Given a twitter user, the timeline of the user is downloaded,
	the @-mentions are extracted and an ego graph for the user is built.
	The neighbors (@-mentions) of the user are matched with geolocated
	users of the WORLD dataset (Han et. al, 2012) and their locations
	are set if any matches are found.
	The user is then geolocated using the locations of its neighbours.
	The geolocation algorithm is based on real-valued label propagation (Rahimi et. al, 2015).
	
	Args:
		twitter_user (str): a Twitter screen/id.
		return_address : if True the predicted coordinates are mapped to an address using geopy. Default (False).
	Returns:
		a dictionary with location information
	"""
	load_lpmodel(lpmodel_file)
	from twitterapi import download_user_tweets_iterable
	timelines = download_user_tweets_iterable(twitter_users, count=200)
	results = {}
	for user, timeline in timelines.iteritems():
		text = ' '.join([t.text for t in timeline])
		#pattern for @mentions
		token_pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
		token_pattern = re.compile(token_pattern)
		mentions = [m.lower() for m in token_pattern.findall(text)]
		mention_hashes = [hashlib.md5(m).hexdigest() for m in mentions]
		neighbour_coordinates = []
		for mention_hash in mention_hashes:
			if mention_hash in params.userhash_coordinate:
				coordinate = params.userhash_coordinate[mention_hash]
				neighbour_coordinates.append(coordinate)
		
		#no match found, unable to geolocate
		if len(neighbour_coordinates) == 0:
			result = {'result':'no match found'}
			results[user] = result
			continue
		
		latitudes = [coor[0] for coor in neighbour_coordinates]
		longitudes = [coor[1] for coor in neighbour_coordinates]
		median_lat = numpy.median(latitudes)
		median_lon = numpy.median(longitudes)
	
		if return_address:
			co_loc = retrieve_location_from_coordinates(points=[(median_lat, median_lon)])
		
		result = {'lat':median_lat, 'lon':median_lon, 'address':co_loc.get((median_lat, median_lon), {})}
		results[user] = result
		logging.debug(result)
	
	return results	 
	


		
def load_model(model_dir='./models/lrworld'):
	""" Given a directory, loads the saved (pickled and gzipped) geolocation model into memory.
	"""
	if not os.path.exists(model_dir):
		logging.error('The input directory with --model/-m option does not exist: ' + model_dir)
		logging.error('If it is the first time you are using pigeo, please run download_models.sh or manually download the models from https://drive.google.com/file/d/0B9ZfPKPvp-JibDlLNTJnMnlQZ3c/view?usp=sharing or https://www.dropbox.com/s/gw8z0r5nq5ccok0/models.tar?dl=0')
		sys.exit(-1)

	logging.info('loading the saved model from pickle files in ' + model_dir)
	logging.info('it might take about 2 minutes...')
	logging.debug('loading coordinate city mappings...')
	params.coordinate_address = pickle.load(gzip.open(os.path.join(model_dir, 'coordinate_address.pkl.gz'), 'rb'))
	params.label_coordinate = pickle.load(gzip.open(os.path.join(model_dir, 'label_coordinate.pkl.gz'), 'rb'))
	logging.debug('loading feature extractor ...')
	params.vectorizer = pickle.load(gzip.open(os.path.join(model_dir, 'vectorizer.pkl.gz'), 'rb'))
	params.vectorizer.features = params.vectorizer.get_feature_names()
	logging.debug('loading the trained classifier ...')
	params.clf = pickle.load(gzip.open(os.path.join(model_dir, 'clf.pkl.gz'), 'rb'))
	params.model_loaded = True

def start_web(model_dir, debug=False, host='127.0.0.1', port=5000):
	if not params.model_loaded:
		load_model(model_dir)
	app.run(debug=debug, host=host, port=port)

def start_commandline(model_dir):
	if not params.model_loaded:
		load_model(model_dir)
	text = None
	while True:
		text = raw_input("text to geolocate: ")
		if text in ['exit', 'quit', 'q']:
			return
		result = geo(text)
		print result
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-d', default="./models/lrworld", help="text-based classification model directory to be used. default(./models/lrworld)")
	parser.add_argument('--dump_dir', '-o', default="./models/test_model", help="directory to which a newly trained model is saved. default(./models/test_model)")
	parser.add_argument('--host', default='127.0.0.1', help='host name/IP address where Flask web server in web mode will be running on. Set to 0.0.0.0 to make it externally available. default (127.0.0.1)')
	parser.add_argument('--port', '-p', type=int, default=5000, help='port number where Flask web server will bind to in web mode. default (5000).')
	parser.add_argument('--mode', '-m', default='shell', help='mode (web, shell) in which pigeo will be used. default (shell).')
	args = parser.parse_args()
	
	if args.mode == 'shell':
		start_commandline(args.model)
	elif args.mode == 'web':
		start_web(args.model, debug=True, host=args.host, port=args.port)
