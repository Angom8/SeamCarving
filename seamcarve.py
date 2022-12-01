#THEBAULT Antoine 2021 - Projet CHPS0705, URCA

import PIL
import math
from PIL import Image, ImageFilter
from threading import Thread
from multiprocessing import Process
import numpy as np
import os
import cv2
from pathlib import Path
import json
import time

"On calcule le coefficient de la matrice du filtre (on ne peut pas faire une simple somme à cause des possibles négatifs notamment sur Sobel), ainsi que le décalage qui devra être appliqué"
def calculer_coefficient(filtre):
	nbPositifs = 0
	nbNegatifs = 0
	taille_i = len(filtre)
	taille_j = len(filtre[0])
	
	for i in range(taille_i):
        	for j in range(taille_j):
        		if filtre[i][j] >= 0:
        			nbPositifs += filtre[i][j]
        		else:
        			nbNegatifs -= filtre[i][j]
        			
	coeff = nbNegatifs + nbPositifs
	decalage = (nbNegatifs*255)
	
	return (coeff, decalage)

"A partir d'un filtre et d'une image donnée, on calcule la convolution d'un pixel en fonction de ses voisins (le nombre dépend du filtre utilisé, 8 voisins pour Sobel par exemple)"
def calculer_convolution(img, x, y, filtre, largeur, hauteur):

	convolution = 0
	intensity = lambda pixel : math.sqrt((pixel[0]**2)+(pixel[1]**2)+(pixel[2]**2))
	centre = len(filtre)//2
	
	for i in range(-centre,centre):
		for j in range(-centre,centre):
			if filtre[i + centre][j + centre] != 0: 
				#On vérifie ici que l'on ne sort pas de l'image (erreur sinon)
				if x + i < largeur-centre  and x + i  >= 0 and y + j < hauteur-centre and y + j >= 0:
					xx = x+i
					yy = y+j
				else:
					xx = x-i
					yy = y-j
				if not (xx + i < largeur-centre  and xx+i >= 0 and yy + j < hauteur-centre and yy+j >= 0):
					xx = x
					yy = y
            
				convolution += (intensity(img.getpixel((xx,yy))) * (filtre[i+centre][j+centre]))

	return convolution

"Division du traitement pour le calcul de la convolution sur un pixel, ainsi que le placement de celui-ci sur la nouvelle image"
def thread_task(n, size, img,imgContour,filtre,hauteur,coeff,decalage):
	#sections de l'image découpées
	for x in range(n*size,n*size+size):
		for y in range(0,hauteur):
			conv = calculer_convolution(img,x,y,filtre,largeur,hauteur)
			val = round((conv+decalage)/coeff)
			imgContour.putpixel((x,y),(val,val,val))

"Application du filtre de sobel à l'aide des fonctions précédentes." 
def contours_sobel(img, largeur, hauteur,user_threads):
	imgContour = img.copy()
	#filtre horizontal pour avoir le maximum de détails (étant sur un traitement vertical)
	filtreContour = [[-1,-2,-1],[0,0,0],[1,2,1]]
	coeff, decalage = calculer_coefficient(filtreContour)

	threads = []	
	size = largeur//user_threads
	print(size)
	for n in range(0, user_threads):
		t = Process(target=thread_task, args=(n,size,img,imgContour,filtreContour,hauteur,coeff,decalage))
		threads.append(t)
		t.start()
		
	if((largeur - size*user_threads) != 0):
		for x in range(user_threads*size,largeur):
			for y in range(0,hauteur):
				conv = calculer_convolution(img,x,y,filtreContour,largeur,hauteur)
				val = round((conv+decalage)/coeff)
				imgContour.putpixel((x,y),(val,val,val))
				
	for t in threads:
		t.join()			
	return imgContour
	
"Application du filtre de Canny (avec flou) à l'aide des fonctions précédentes."
def contours_canny(img, largeur, hauteur,user_threads):
	imgFloue = img.copy()
	imgContour = img.copy()
	filtreFlou = [[2, 4, 5, 4, 2],
		      [4, 9, 12, 9, 4],
		      [5,12, 15,12, 5],
		      [4, 9, 12, 9, 4],
		      [2, 4, 5, 4, 2]]
		      
	filtreContour = [[-1,-2,-1],[0,0,0],[1,2,1]]
	
	#calcul du flou
	coeff, decalage = calculer_coefficient(filtreFlou)
	threads = []	
	size = largeur//user_threads
	for n in range(0, user_threads):
		t = Process(target=thread_task, args=(n, size,img,imgFloue,filtreFlou,hauteur,coeff,decalage))
		threads.append(t)
		t.start()

	if((largeur - size*user_threads) != 0):
		for x in range(user_threads*size,largeur):
			for y in range(0,hauteur):
				conv = calculer_convolution(img,x,y,filtreFlou,largeur,hauteur)
				val = round((conv+decalage)/coeff)
				imgContour.putpixel((x,y),(val,val,val))
	for t in threads:
		t.join()			
	
	#calcul du gradient
	coeff, decalage = calculer_coefficient(filtreContour)
	threads = []	
	for n in range(0, user_threads):
		t = Process(target=thread_task, args=(n, size,imgFloue,imgContour,filtreContour,hauteur,coeff,decalage))
		threads.append(t)
		t.start()

	
	if((largeur - size*user_threads) != 0):
		for x in range(user_threads*size,largeur):
			for y in range(0,hauteur):
				conv = calculer_convolution(img,x,y,filtreContour,largeur,hauteur)
				val = round((conv+decalage)/coeff)
				imgContour.putpixel((x,y),(val,val,val))
	
	for t in threads:
		t.join()
	del threads		
				
	return imgContour
    	
"A partir de l'image contenant les gradients, on calcule le plus court chemin pour la colonne x. Si cette colonne penche pour la précédente, on détermine que le parcours au delà de ce point y est identique au précédent pcm."
"pcm est un tableau indiquant pour chaque colonne x à une hauteur y le prochain pixel, sa hauteur et la valeur du gradient."
def plus_court_chemin(imageContour, pcm, x, largeur):
	#pour éviter de sortir de l'image 
	sous = -1
	add = 1 
	if(x==0):
		sous = 0
	if(x==largeur-1):
		add = 0

	#On parcourt de haut en bas
	for dy in range(0,len(imageContour[0])-2):
		#val minimale différent gradient
		a = imageContour[x + sous][dy+1][0]
		b = imageContour[x][dy+1][0]
		c = imageContour[x+add][dy+1][0]
		
		if(a <= b and a <= c):
			next_pixel = (x + sous,dy+1,a)
		elif (b <= a and b <= c):
			next_pixel = (x,dy+1,b)
		else:
			next_pixel = (x+add,dy+1,c)
		
		#Si on suit le même chemin que le pcm-1, on le copie (rend le multithread quasi-impossible malheureusement)
		if(x != 0):
			if(pcm[x+sous][dy][0] and next_pixel[0] == pcm[x+sous][dy][0]):
				for dy in range (dy, len(imageContour[0])-1):
					pcm[x][dy][0] =  pcm[x+sous][dy][0]
					pcm[x][dy][1] =  pcm[x+sous][dy][1]
					pcm[x][dy][2] =  pcm[x+sous][dy][2]	
			else:
				pcm[x][dy][0] = next_pixel[0]
				pcm[x][dy][1] = next_pixel[1]
				pcm[x][dy][2] = next_pixel[2]
		else:
			pcm[x][dy][0] = next_pixel[0]
			pcm[x][dy][1] = next_pixel[1]
			pcm[x][dy][2] = next_pixel[2]
	return pcm

"Version thread"
def plus_court_chemin_task(imageContour, pcm, n,size, largeur):
	for x in range(n*size,n*size+size):
		#pour éviter de sortir de l'image 
		sous = -1
		add = 1 
		if(x==0):
			sous = 0
		if(x==largeur-1):
			add = 0

		#On parcourt de haut en bas
		for dy in range(0,len(imageContour[0])-2):
			#val minimale différent gradient
			a = imageContour[x + sous][dy+1][0]
			b = imageContour[x][dy+1][0]
			c = imageContour[x+add][dy+1][0]
			
			if(a <= b and a <= c):
				next_pixel = (x + sous,dy+1,a)
			elif (b <= a and b <= c):
				next_pixel = (x,dy+1,b)
			else:
				next_pixel = (x+add,dy+1,c)
			
			#Si on suit le même chemin que le pcm-1, on le copie (rend le multithread quasi-impossible malheureusement)
			if(x != 0):
				if(pcm[x+sous][dy][0] and next_pixel[0] == pcm[x+sous][dy][0]):
					for dy in range (dy, len(imageContour[0])-1):
						pcm[x][dy][0] =  pcm[x+sous][dy][0]
						pcm[x][dy][1] =  pcm[x+sous][dy][1]
						pcm[x][dy][2] =  pcm[x+sous][dy][2]	
				else:
					pcm[x][dy][0] = next_pixel[0]
					pcm[x][dy][1] = next_pixel[1]
					pcm[x][dy][2] = next_pixel[2]
			else:
				pcm[x][dy][0] = next_pixel[0]
				pcm[x][dy][1] = next_pixel[1]
				pcm[x][dy][2] = next_pixel[2]
	return pcm

"On crée un nouveau tableau à une dimension qui pour chaque plus court chemin d'une colonne x associe la somme de son gradient"
def tri_pcm(pcm):	
	tri_pcm = np.zeros(len(pcm)-1)
	for x in range(0,len(pcm)-1):
		diff = 0
		for y in range(1,len(pcm[0])-1):
			diff += abs(pcm[x][y][2]-pcm[x][y-1][2])
		
		tri_pcm[x] = diff
	return tri_pcm
	
"Task pour le redimensionnement de l'image. On décale les pixels à supprimer hors de la zone visible (crop à la fin du traitement)"
def resize_task(n, size,imgContour, i, selected,nouvelleImage, pcm):
	for y in range(n*size,n*size+size):
		x = int(pcm[selected][y][0])
		#sur l'image
		for x in range(x,len(nouvelleImage)-1):
			nouvelleImage[x][y] = nouvelleImage[x+1][y]
			imgContour[x][y]= imgContour[x+1][y]
		#sur le gradient
		for x in range(x,len(imgContour)-i-1):	
			imgContour[x][y]= imgContour[x+1][y]

"On parcourt au fur et à mesure avec affichage progressif"
def seam_carving(img,nb,succ, user_threads, user_gradientmode,user_autoscroll):
	largeur, hauteur = img.size	
	#Initialisation
	
	#On transforme notre image contour en un tableau numpy (on oublie pas de swap les axes pour rendre le code plus lisible)
	imgContour = np.array(user_gradientmode(img, largeur, hauteur,user_threads))
	imgContour = np.swapaxes(imgContour, 1, 0)
	nouvelleImage = np.array(np.swapaxes(img.copy(), 1, 0))
	pcm_min = []
	pcm = []
	
	#boucle principale
	for i in range(0,nb):
	
		print("Tour " + str(i+1))
		
		#calcul du plus petit chemin
		del pcm
		pcm = np.zeros((largeur-i,hauteur-1,3))
		
		size = largeur//(len(nouvelleImage[0])-i-1)
		threads = []
		for n in range(0, user_threads):
			t = Process(target=plus_court_chemin_task, args=(imgContour, pcm, n,size, len(nouvelleImage[0])-i))
			threads.append(t)
			t.start()
		
		#si nombre de thread n'est pas pile
		if(((len(nouvelleImage[0])-i-1) - size*user_threads) != 0):
			for z in range (size*user_threads,len(nouvelleImage[0])-i-1):
				plus_court_chemin(imgContour, pcm, z, len(nouvelleImage[0])-i)
		
		for t in threads:
			t.join()
			
		del pcm_min		
		#choix du + petit chemin
		pcm_min = tri_pcm(pcm)
		selected = int(np.argmin(pcm_min))
	
		#décalage des pixels
		
		size = largeur//(len(nouvelleImage[0])-1)
		for n in range(0, user_threads):
			t = Process(target=resize_task, args=(n, size,imgContour,i, selected,nouvelleImage, pcm))
			threads.append(t)
			t.start()
			
		#si nombre de thread n'est pas pile
		if(((len(nouvelleImage[0])-1) - size*user_threads) != 0):
			for y in range(size*user_threads,(len(nouvelleImage[0])-1)):
				x = int(pcm[selected][y][0])
				#sur l'image
				for x in range(x,len(nouvelleImage)-1):
					nouvelleImage[x][y] = nouvelleImage[x+1][y]
					imgContour[x][y]= imgContour[x+1][y]
				#sur le gradient
				for x in range(x,len(imgContour)-i-1):	
					imgContour[x][y]= imgContour[x+1][y]
		
		for t in threads:
			t.join()						
		
		#Affichage ou non
		if(succ):			
			cv2.imshow('image',cv2.cvtColor(np.swapaxes(nouvelleImage, 1, 0)[:,0:largeur-i], cv2.COLOR_BGR2RGB))
			if(user_autoscroll == 'Y' or user_autoscroll == 'y'):
				cv2.waitKey(0)
			else:
				cv2.waitKey(300)
	
	#remise en place de l'image
	nouvelleImage = np.swapaxes(nouvelleImage, 1, 0)
	nouvelleImage = Image.fromarray(nouvelleImage)	

	#On retire tous les pixels déplacés hors de la zone
	nouvelleImage = nouvelleImage.crop((0, 0, largeur-nb, hauteur))
	nouvelleImage.show()
	
	return nouvelleImage

def montrer_gradient(img, fonction_filtre,user_threads):
	imgContour = np.array(fonction_filtre(img, largeur, hauteur,user_threads))
	imgContour = np.swapaxes(imgContour, 1, 0)
	cv2.imshow('Gradient de l\'image',cv2.cvtColor(np.swapaxes(imgContour, 1, 0)[:,0:largeur], cv2.COLOR_BGR2RGB))
	cv2.waitKey(0)

##============================================  Initialisation

# Nom du fichier
user_filename = input("Nom du fichier image :\n")

fichier = os.path.basename(str(user_filename))
nom_image = os.path.splitext(fichier)[0]

ext = os.path.splitext(fichier)[1]
img = Image.open(nom_image + ext);

# Affiche les informations sur l'image
print(img.format, img.size, img.mode)
print("Fichier :", img.filename)
print("L =", img.width, "x H =", img.height)
largeur, hauteur = img.size
print("Nombre de pixels =", (img.width*img.height) , "pixels")

##============================================ Lecture ou non à partir d'un fichier de config s'il existe
if(Path("settings.json").exists()):
	user_usesettings = input("Un fichier de configuration existe. Voulez-vous l'utiliser ? (Y/N) :\n")
	
	#reprise du fichier existant
	if(user_usesettings   == "y" or user_usesettings  == "Y"):
		with open('settings.json') as settings:
  			data = json.load(settings)
  			user_gradientmodesave = data[0]
  			user_showgradient = data[1]
  			user_mode = data[2]
  			user_iterations = data[3]
  			user_threads = data[4]
  			user_enableoutput = data[5]
  			user_autoscroll = data[6]
  			
  			if(user_gradientmodesave == "contours_canny"):
  				user_gradientmode = contours_canny
  			else:
  				user_gradientmode = contours_sobel
  				
  			#mode non reconnu
  			if(not (user_mode  == "s" or user_mode  == "S" or user_mode  == "u" or user_mode  == "U")):
  				exit(0)
  				
  	#Ecriture des paramètres			
	else:
		#Mode du gradient : Sobel + rapide, Canny + lent, Sobel par défaut
		user_gradientmode = input("Quel filtre voulez-vous utiliser ? (Sobel par défaut, c/C pour Canny) :\n")
		if(user_gradientmode  == "c" or user_gradientmode  == "C"):
			user_gradientmode = contours_canny
			user_gradientmodesave = "contours_canny"
		else:
			user_gradientmode = contours_sobel
			user_gradientmodesave = "contours_sobel"

		#Afficher gradient ou non
		user_showgradient = input("Voulez-vous afficher le gradient ? Y/N (Non par défaut) \n")
		if(not(user_showgradient == "y" or user_showgradient  == "Y")):
			user_showgradient = "N"
			

		# Mode successif ou unifié
		user_mode = input("Mode successif ou unifié ? (s/u pour successif ou unifié, autres pour quitter) :\n")

		if(not (user_mode is None or user_mode == '' or user_mode  == "s" or user_mode  == "S" or user_mode  == "u" or user_mode  == "U")):
			print("Caractère de sortie, fermeture de l'application")
			exit(0)
		elif (user_mode is None or user_mode == ''):
			user_mode = "s"

		# Nombre d'itérations
		user_iterations = input("Nombre itérations :\n")
		if(user_iterations is None or user_iterations == ''):
			user_iterations = 10
		else :
			user_iterations= int(user_iterations)

		#Nombre de threads max (6 = par défaut)
		user_threads = input("Nombre de threads à allouer (0 ou 1 pour désactiver) :\n")
		if(user_threads is None or user_threads == '' or user_threads == 0):
			user_threads = 1
		else :
			user_threads = int(user_threads)

		#Activer output
		user_enableoutput = input("Voulez-vous activer l'écriture de fichier en sortie ? (Y/N) :\n")
		if(user_enableoutput is None or user_enableoutput ==''):
			user_enableoutput = "N"

		#Ajouter défilement AUTO ou PAR INPUT
		user_autoscroll = input("Mode successif : Voulez-vous activer le défilement manuel ? (Y/N) :\n")
		if(user_autoscroll is None or user_autoscroll == ''):
			user_autoscroll  = "N"

##============================================ PROGRAMME POST-CONFIG

#Liste des paramètres :
## user_gradientmodesave = data[0] = filtre de canny ou filtre de sobel pour le gradient 
## user_showgradient = data[1] = affichage avant l'image de son gradient  
## user_mode = data[2] = mode unifié ou successif, avant le lancement  
## user_iterations = data[3] = nombre pixels à retirer 
## user_threads = data[4] = nombre de threads à utiliser
## user_enableoutput = data[5] = active ou non la sortie dans un fichier  
## user_autoscroll = data[6] = pour l'affichage unifié, active ou non l'autoscroll 
  			

#Affichage du gradient
if(user_showgradient == "y" or user_showgradient  == "Y"):
	montrer_gradient(img, user_gradientmode,user_threads)
	
#Seam carving successif
if(user_mode == "s" or user_mode == "S"):
	succ=True
#Seam carving unifié
elif (user_mode == "u" or user_mode == "U"):
	succ=False
else :
	exit(0)
	
resultImg = seam_carving(img, user_iterations,succ, user_threads, user_gradientmode,user_autoscroll)

#sauvegarde de l'image si sauvegarde activée 
if(user_enableoutput == "Y" or user_enableoutput == "y"):
	resultImg.save(time.strftime("%Y-%m-%d-%H:%M_") + img.filename)

# Fermeture des fichiers
resultImg.close()
img.close()
	
#sauvegarde des paramètres	
save_settings = open("settings.json", "w")
save_settings.write(json.dumps([user_gradientmodesave, user_showgradient, user_mode, user_iterations, user_threads,user_enableoutput,user_autoscroll], indent=4))
