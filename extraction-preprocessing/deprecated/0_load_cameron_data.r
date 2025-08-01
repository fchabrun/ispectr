rm(list=ls())

source("R/it_functions.r")

library(openxlsx)
library(stringr)
library(tidyverse)
library(gridExtra)
library(progress)

base_data_path = "C:/Users/flori/Documents/Home/Research/SPECTR/ISPECTR/data/cameron_may_2024/is_data_xlsx"
output_data_path = "C:/Users/flori/Documents/Home/Research/SPECTR/ISPECTR/data/proc"

found_db_files <- list.files(base_data_path)
print(paste0("Found ", length(found_db_files), " files in base directory"))

alldb_data <- NULL
for (db_file in found_db_files) {
  print(paste0("Parsing file ", db_file))

  db_data = read.xlsx(file.path(base_data_path, db_file))

  print(paste0("File contains ", nrow(db_data), " raw entries"))

  # limit to F program (IS) and J (SPEP)
  db_data <- db_data[db_data$programma %in% c("F", "J"), ]
  print(paste0("File contains ", nrow(db_data), " raw F/J entries"))

  strsplits <- str_split(db_data$id, "-")

  db_data$id_unique <- sapply(strsplits, function(x) {x[1]})
  db_data$id_track <- sapply(strsplits, function(x) {ifelse(is.na(x[2]),"",x[2])})
  db_data[ ,"id_track"] <- ifelse(db_data$programma == "J", "Ref", db_data[ ,"id_track"])
  db_data$id_ctrl <- substr(db_data$id_unique, 1, 9) == "CONTROLID"
  db_data <- db_data[!db_data$id_ctrl, ]

  #check how many, etc.
  db_data <- db_data %>%
    group_by(id_unique) %>%
    mutate(n_ref = sum(id_track == "Ref")) %>%
    mutate(n_elp = sum(id_track == "ELP")) %>%
    mutate(n_g = sum(id_track == "IgG")) %>%
    mutate(n_a = sum(id_track == "IgA")) %>%
    mutate(n_m = sum(id_track == "IgM")) %>%
    mutate(n_k = sum(id_track == "K")) %>%
    mutate(n_l = sum(id_track == "L"))

  db_data$checked_valid = (db_data$n_ref == 1) & (db_data$n_elp == 1) & (db_data$n_g == 1) & (db_data$n_a == 1) & (db_data$n_m == 1) & (db_data$n_k == 1) & (db_data$n_l == 1)

  # filter the db
  db_data <- db_data[db_data$checked_valid, ]

  print(paste0("File contains ", nrow(db_data), " valid entries for ", length(unique(db_data$id_unique)), " unique PB samples"))

  # just for ensuring consistency between datasets
  db_data[ ,paste0("nome_", 1:10)] <- lapply(db_data[ ,paste0("nome_", 1:10)], as.character)
  db_data[ ,paste0("nome_picco_", 1:10)] <- lapply(db_data[ ,paste0("nome_", 1:10)], as.character)
  db_data[ ,"rack_barcode"] <- lapply(db_data[ ,"rack_barcode"], as.character)

  db_data <- as.data.frame(db_data)

  # add to global file
  if (is.null(alldb_data)) {
    alldb_data <- db_data
  } else {
    # check no duplicates
    if (any(unique(db_data$id_unique) %in% unique(alldb_data$id_unique))) {
      warning("Duplicated entries with previous dataset(s)! Removing")
      unique_ids <- unique(db_data$id_unique)
      keep_unique_ids <- unique_ids[!(unique_ids %in% unique(alldb_data$id_unique))]
      db_data <- db_data[db_data$id_unique %in% keep_unique_ids, ]
      print(paste0("Filtered file contains ", nrow(db_data), " valid entries for ", length(unique(db_data$id_unique)), " unique PB samples"))
    }
    alldb_data <- rbind(alldb_data, db_data)
  }
}

print(paste0("Final full db contains ", nrow(alldb_data), " valid entries for ", length(unique(alldb_data$id_unique)), " unique PB samples"))

# re-sort
alldb_data <- alldb_data[order(alldb_data$id), ]

is_data = list()
for (id_track in c("Ref", "ELP", "IgG", "IgA", "IgM", "K", "L")) {
  # take ref
  track_df <- alldb_data[alldb_data$id_track == id_track, ]
  track_df_extract <- convert_to_spep_matrix(db_full = track_df)
  # add previous metadata, without extracted columns
  keepcols <- c("id_unique", "id", "id_track", "data_analisi", "pt", "commento1", "commento2", "longmemo")
  track_df <- cbind(track_df[ ,keepcols], track_df_extract)
  # change column names according to track
  is_data[[id_track]] <- track_df
}

global_is_data <- data.frame(id_unique = is_data[["Ref"]][ ,c("id_unique")])
for (id_track in c("Ref", "ELP", "IgG", "IgA", "IgM", "K", "L")) {
  if (any(global_is_data$id_unique != is_data[[id_track]]$id_unique)) {
    stop("Order was lost during process!")
  }
  is_data_transfer <- is_data[[id_track]][ ,-1]
  colnames(is_data_transfer) <- paste(id_track, colnames(is_data_transfer), sep="_")
  global_is_data <- cbind(global_is_data, is_data_transfer)
}

# save
# library(rhdf5)
# h5_fp <- file.path(root_path, 'spep_full.h5')
# h5createFile(h5_fp)
# h5write(spe_df, h5_fp, "dataset")
# also as csv for easier reading
write.csv(global_is_data, file.path(output_data_path, 'sa_is_full.csv'), row.names=F)

# TODO plus qu'à exporter vers python
# TODO indice: on voit que le pic disparaît de la piste pos, mais on voit aussi que la précipitation crée une bosse vers l'albu
# => on pourrait trouver un moyen de "masquer" le pic pour forcer le modèle à regarder pas seulement le pic mais aussi l'albu de cette manière! (plus robuste?)





























