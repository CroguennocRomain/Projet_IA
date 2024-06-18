import sys
import json


def predire_age(features, method):
    print(features)
    print(method)

    return 0


# On récupère les arguments noms de colonnes dans features
features = []
i_last_arg = len(sys.argv) - 1  # on ne compte pas argv[0] qui est le nom du script
for i in range(1, i_last_arg):
    features.append(sys.argv[i])


predire_age(features, sys.argv[i_last_arg])

