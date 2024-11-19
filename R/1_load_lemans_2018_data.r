# Passons aux IF

rm(list = ls())

source("R/it_functions.r")

library(data.table) # pour charger rapidement les fichiers .csv
library(ggplot2)

# On charge directement la db que l'on avait sauvegardé
base_data_path = "C:/Users/flori/Documents/Home/Research/SPECTR/ISPECTR/data/lemans_2018"
db_F_full <- as.data.frame(fread(file.path(base_data_path, "db.F.anonymized.csv"), header = T, sep = ','))
dim(db_F_full) # 17717 lignes, 6 colonnes

# write.csv(db_F_full[1:14,-1], "C:/Users/admin/Documents/Capillarys/reports/DELTA/Synthèse/Annexe 2.csv", quote=F, row.names=F)

decipherWholeIF <- function(data_lines) {
  df=data.frame(x=rep(seq(1:300),7),
                y=numeric(300*7),
                m=character(300*7),
                stringsAsFactors = FALSE)
  for(i in 1:nrow(data_lines)) {
    new_curve = decipherIF(data_lines[i,])$curve
    df[((i-1)*300+1):((i-1)*300+length(new_curve)),"y"]=new_curve
    df[((i-1)*300+1):((i-1)*300+length(new_curve)),"m"]=data_lines[i,]$method
  }
  df=df[nchar(df$m)>0,]
  return(df)
}

# On va exporter sans les coordonnées des pics, pour voir si le modèle de détection des pics sur SPE peut marcher dessus....

# Ok, on a une idée de comment ça marche maintenant
# Ce qu'on va faire
# C'est qu'on va analyser les commentaires pour essayer de comprendre les anomalies

# On va commencer par garder uniquement les AID pour lesquels on a 7 courbes
# Et un commentaire (plus de 3 caractères)

correct_AID = db_F_full$analysisanonid[db_F_full$method=="ELP" & nchar(db_F_full$comment_long)>3]

db_filt = db_F_full[db_F_full$method == "ELP" & db_F_full$analysisanonid %in% correct_AID, ]

comments=as.character(db_filt$comment_long)
comments=iconv(comments,from="UTF-8",to="ASCII//TRANSLIT")
comments=tolower(comments)

# Ok, on a saisi le principe global
# Maintenant on va essayer d'attribuer à chaque courbe : soit G, soit M, soit A
# et soit kappa, soit lambda
# on va voir s'il y a des fois ou on a les deux qui reviennent

iggl = grepl("igg (monoclonale )?a chaine(s)? legere(s)? lambda", comments) | grepl("igg lambda", comments)
igal = grepl("iga (monoclonale )?a chaine(s)? legere(s)? lambda", comments) | grepl("iga lambda", comments)
igml = grepl("igm (monoclonale )?a chaine(s)? legere(s)? lambda", comments) | grepl("igm lambda", comments)
iggk = grepl("igg (monoclonale )?a chaine(s)? legere(s)? kappa", comments) | grepl("igg kappa", comments)
igak = grepl("iga (monoclonale )?a chaine(s)? legere(s)? kappa", comments) | grepl("iga kappa", comments)
igmk = grepl("igm (monoclonale )?a chaine(s)? legere(s)? kappa", comments) | grepl("igm kappa", comments)

ig_normal = grepl("immunotypage normal", comments)

sum(iggl) # 411
sum(igal) # 85
sum(igml) # 99
sum(iggk) # 773
sum(igak) # 113
sum(igmk) # 368

sum_ig = (iggl*1+igal*1+igml*1+iggk*1+igak*1+igmk*1)

table(sum_ig, ig_normal) # 7 unknown, 75 with two spikes

comments[sum_ig==0 & !ig_normal]
############
# TODO
# le numéro 2 est à corriger manuellement
# le numéro 3 aussi
# le numéro 5 aussi
############

############
# TODO
sum(grepl("disparition", comments)) # 2 avec "disparition" !! attention, à retirer !!
comments[grepl("disparition", comments)] # deux : présence d'une anomalie + disparition d'une autre !
############

# qu'en est-il des courbes avec deux pics ?

comments[sum_ig>1]
# tout a l'air bien (p ou moins)

# qu'en est-il des profil oligo ?
sum(grepl("oligoclonal", comments)) # 6
comments[grepl("oligoclonal", comments)] # seulement 6, on va considérer ça comme des pics...

# on va faire les dernières corrections manuelles et enregistrer tout ça
# tout d'abord les ig non vues
iggk[which(sum_ig==0 & !ig_normal)[2]] = TRUE
iggk[which(sum_ig==0 & !ig_normal)[3]] = TRUE
iggl[which(sum_ig==0 & !ig_normal)[5]] = TRUE
# puis les disparitions
iggk[which(grepl("disparition", comments))[1]] = FALSE
igml[which(grepl("disparition", comments))[2]] = FALSE

# Ok
# Maintenant on va pouvoir filtrer et convertir tout ça en une nouvelle matrice !

sum_ig = (iggl*1+igal*1+igml*1+iggk*1+igak*1+igmk*1)
sum_ok = sum_ig+ig_normal*1

table(sum_ok) # on va en garder 1803, et on en jette 4
comments[sum_ok==0]
# [1] "persistance de l'anomalie identifiee le : 28/06/2011\\n"
# [2] "persistance de l'anomalie identifiee le : 16/01/2015\\n"
# [3] "immunotypage realise apres administration de gammaglobulines"
# [4] "presence de chaines legeres libres kappa.\\nfaire ifp pour confirmer si il n'y a pas d'igd ou d'ige."

# on va pouvoir sélectionner les ruks correspodnant et faire une match table
match_table_for_F = data.frame(AID = db_filt$analysisanonid[sum_ok>0],
                               iggk = iggk[sum_ok>0],
                               iggl = iggl[sum_ok>0],
                               igak = igak[sum_ok>0],
                               igal = igal[sum_ok>0],
                               igmk = igmk[sum_ok>0],
                               igml = igml[sum_ok>0]) # ps : on ne note pas "normal" car c'est par défaut !

dim(match_table_for_F) # 1803 x 7 IF

# on va parcourir cette liste et rajouter les valeurs des courbes ELP, IgG, IgA, IgM, K et L


library(pbar)

array_order = c("ELP","IgG","IgA","IgM","K","L")
data_array = array(NA, dim=c(nrow(match_table_for_F),length(c("ELP","IgG","IgA","IgM","K","L")),300))
my_pb = pb$new(nrow(match_table_for_F))
for(i in 1:nrow(match_table_for_F)) {
  chosen_ID = match_table_for_F$AID[i]
  tmp=decipherWholeIF(db_F_full[db_F_full$analysisanonid==chosen_ID,])
  # check all curves are here and same length
  prev_l = 298
  OK = T
  for(arr in array_order) {
    if(sum(tmp$m==arr) != prev_l) {
      OK = F
      break
    }
  }
  if (!OK) {
    print(paste0("Not ok for i: ",i))
    break
  }
  # fill mat
  for(arr in 1:length(array_order)) {
    unpadded_curve <- tmp[tmp$m==array_order[arr],]$y
    padded_curve <- c(0,unpadded_curve,0)
    data_array[i,arr,] = padded_curve
  }
  my_pb$update()
}

sum(is.na(array_order)) # 0 ! everything went well ! :)

final_df <- match_table_for_F
final_df$iggk <- final_df$iggk*1
final_df$iggl <- final_df$iggl*1
final_df$igak <- final_df$igak*1
final_df$igal <- final_df$igal*1
final_df$igmk <- final_df$igmk*1
final_df$igml <- final_df$igml*1

for(i in 1:length(array_order)) {
  tmp_df <- as.data.frame(data_array[,i,])
  colnames(tmp_df) <- paste0(array_order[i], 'x', 1:300)
  final_df <- cbind(final_df, tmp_df)
}

dim(final_df)
# 1803 x 1831 (7+304*6)

# export for python
write.csv(x = final_df,
          file = paste0("C:/Users/admin/Documents/Capillarys/data/2021/ifs/lemans_if_matUNNORMED.csv"),
          row.names = F)




