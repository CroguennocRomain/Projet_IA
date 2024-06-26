Lien de notre dépot Git : "https://github.com/CroguennocRomain/Projet_IA.git"

Nous avons 3 script.py, 1 par fonctionnalité :

- "script_fonc1.py" --> script pour la première fonctionnalité pour prédire à quel cluster de taille totale un arbre appartient.

Le script prend en entrée une valeur pour chaque features : "haut_tot", "haut_tronc", "fk_stadedev", "fk_nomtech", "feuillage".
Et retourne un fichier JSON et crée un fichier JSON "script1_result.json" contenant le numéro du cluster auquel appartient l'arbre

Avant d'executer le fichier "script_fonc1.py", s'il n'y a pas de fichier "centroids.csv" ou "ordinal_encoder1.pkl" ou "scaler1.pkl"
Ou si vous souhaitez voir les scores et métriques du modèle d'apprentissage
Vous devez executer le fichier "fonctionnalité_1.py"

Son éxecution dans le terminal se fait sous cette forme :
Usage : python script_fonc1.py <haut_tot> <haut_tronc> <fk_stadedev> <fk_nomtech> <feuillage>
Exemple : python script_fonc1.py 15.1 2.1 "Adulte" "PINNIGnig" "Conifère"


- script_fonc2.py --> script pour la seconde fonctionnalité pour prédire à quel groupe d'âge un arbre appartient.

Le script prend en entrée une valeur pour chaque features et pour la méthode choisie : "haut_tot", "haut_tronc", "tronc_diam", "fk_stadedev", "fk_nomtech".
    - méthode 0 (age_SGD).
    - méthode 1 (age_neigh).
    - méthode 2 (age_SVM).
    - méthode 3 (age_tree).
Et retourne un fichier JSON et crée un fichier JSON "script2_result.json" contenant un tableau récapitulatif entre la classe (groupe d'âge) et la probabilité que l'arbre donné appartient à cette même classe

Avant d'exécuter le fichier "script_fonc2.py", s'il n'y a pas de fichier "ordinal_encoder2.pkl" ou "scaler2.pkl" ou "models/age_neigh.pkl" ou "models/age_SGD.pkl" ou "models/age_SVM.pkl" ou "models/age_tree.pkl"
Ou si vous souhaitez voir les scores et métriques du model d'apprentissage
Vous devez exécuter le fichier "fonctionnalité_2.py"

Son exécution dans le terminal se fait sous cette forme :
Usage : python script_fonc2.py <haut_tot> <haut_tronc> <tronc_diam> <fk_stadedev> <fk_nomtech> <numéro_method>
Exemple : python script_fonc2.py 15.1 2.1 2.5 "Adulte" "PINNIGnig" 3


- script_fonc3.py --> script pour la troisième fonctionnalité pour prédire si un arbre est susceptible d'être déraciné en cas de tempête.

Le script prend en entrée une valeur pour la méthode choisie et une valeur pour chaque features de la méthode :
    - méthode 0 (rf_model) : "haut_tronc", "latitude", "longitude", "fk_stadedev", "haut_tot", "clc_secteur".
    - méthode 1 (knn_model) : "latitude", "longitude", "clc_secteur", "fk_port".
    - méthode 2 (svm_model) : haut_tot", "fk_revetement".
Et retourne un fichier JSON et crée un fichier JSON "script3_result.json" contenant 1 si l'arbre est plus susceptible d'être déraciné par une tempête et 0 sinon.

Avant d'exécuter le fichier "script_fonc3.py", s'il n'y a pas de fichier "ordinal_encoder3.pkl" ou "scaler3.pkl" ou "models/knn_model.pkl" ou "models/rf_model.pkl" ou "models/svm_model.pkl"
Ou si vous souhaitez voir les scores et métriques du modèle d'apprentissage
Vous devez exécuter le fichier "fonctionnalité_3.py"

Son exécution dans le terminal se fait sous cette forme :
    - méthode 0 (rf_model) : Usage : python script_fonc3.py <haut_tronc> <latitude> <longitude> <fk_stadedev> <haut_tot> <clc_secteur> <numéro_method>.
    - méthode 1 (knn_model) : Usage : python script_fonc3.py <latitude> <longitude> <clc_secteur> <fk_port> <numéro_method>.
    - méthode 2 (svm_model) : Usage : python script_fonc3.py <haut_tot> <fk_revetement> <numéro_method>.
Exemple : python script_fonc3.py 49.84050020512298 3.2932636093638927 "Rue de Paris" "réduit relâché" 1
