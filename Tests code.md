Tests à faire dans le code :

--> changer les hyperparamètres avec les résultats de validation, en mettant la seed :
- boucle sur le nombre d'epochs
- taille random des layers (10 fois, garder les meilleures tailles)
- nombre de layers (entre 1 et 15)

--> testing final sur 20000 nouvelles lignes
- comparer notre accuracy globale avec celle de l'article
- comparer notre accuracy signal ou background avec celle de l'article



Plots à faire pour la présentation :

- évolution accuracy et loss pendant le training pour MSE et Binary (comparer les 2)
- évolution accuracy du validate en fonction du nombre d'epochs pour Binary
- tableau des tailles et nombre de layers avec l'accuracy de validation (éventuellement)
- distribution des résultats (0 ou 1) du validate en fonction du nombre d'epochs
- signal VS background pour les variables discriminantes ou non après le testing
- signal VS background en fonction des epochs (nuage de points se resserre vers le 0 ou 1, éventuellement)
- tableau comparant notre accuracy avec celle de l'article



Autres :

- print accuracy et predicted_list dans l'étape de validation 
