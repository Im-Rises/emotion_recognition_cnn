#!/bin/bash

if [ $# == 2 ]; then	
	locationDataBase=$1
	nomFichier=$2
	nombreLignes=0

	if [ -f "$nomFichier" ]; then
	    rm $nomFichier
	fi

	emotions=$(ls $locationDataBase)

	echo ",emotion,image" >> $nomFichier

	for emo in $emotions
	do 
		images=$(ls $locationDataBase$emo)
		for image in $images
		do 
			echo "$nombreLignes,$emo,$locationDataBase$emo/$image" >> $nomFichier
			let "nombreLignes+=1"
		done
	done
else 
	echo -e "\033[01m\033[47m\033[31mMauvaise utilisation\033[00m"
	echo "voici un exemple d'usage : "
	echo "./creer_csv path_vers_le_dossier_database_avant_emotions nom_du_fichier_csv.csv"
fi
