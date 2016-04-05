########################################### Global Variables #################################
#sklearn pickled SGDClassifier where pre-trained clf.coef_ matrix is casted to a scipy.sparse.csr_matrix for efficiency and scalability
clf = None
#sklearn pickled TfidfVectorizer
vectorizer = None
#dictionary of labelid: (latitude, longitude) It is pre-computed as the median value of all training points in a region/cluster
label_coordinate = {}
#dictionary of (latitude,longitude):location (dictionary)
coordinate_address = {}
#check if model is loaded
model_loaded = False
#dictionary of hashed user name:(latitude, longitude) pre-trained by label propagation on TwitterWorld dataset
userhash_coordinate = {}
#check if lpworld model is loaded
lp_model_loaded = False
