old_path = "C:/Users/flori/Documents/Home/Research/SPECTR/ISPECTR/data/lemans_2018/already_proc/lemans_if_curva.csv"
output_data_path = "C:/Users/flori/Documents/Home/Research/SPECTR/ISPECTR/data/lemans_2018/temp_proc"

rm(list=ls())

IS_TRACKS = c("ELP", "IgG", "IgA", "IgM", "K", "L")

source("R/it_functions.r")

library(openxlsx)
library(stringr)
library(tidyverse)
library(gridExtra)
library(progress)

db_data = read.csv(old_path, sep=",")

print(paste0("File contains ", nrow(db_data), " raw entries"))

is_data = list()
for (id_track in IS_TRACKS) {
  # take ref
  track_df <- db_data[ ,c("AID", id_track)]
  colnames(track_df) = c("id", "curva")
  track_df$numfraz = 1
  for (j in 1:10) {
    track_df[ ,paste0("nome_", j)] = "?"
  }
  track_df$coord_picco = ""
  track_df$id_unique = track_df$id
  track_df$id = paste0(track_df$id_unique, "-", id_track)
  track_df$id_track = id_track
  track_df$data_analisi = NA
  track_df$pt = NA
  track_df$commento1 = NA
  track_df$"commento2" = NA
  track_df$longmemo = NA
  track_df_extract <- convert_to_spep_matrix(db_full = track_df)
  # add previous metadata, without extracted columns
  keepcols <- c("id_unique", "id", "id_track", "data_analisi", "pt", "commento1", "commento2", "longmemo")
  track_df <- cbind(track_df[ ,keepcols], track_df_extract)
  # change column names according to track
  is_data[[id_track]] <- track_df
}

global_is_data <- data.frame(id_unique = is_data[["ELP"]][ ,c("id_unique")])
for (id_track in IS_TRACKS) {
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
write.csv(global_is_data, file.path(output_data_path, 'lemans_is_full.csv'), row.names=F)















































































