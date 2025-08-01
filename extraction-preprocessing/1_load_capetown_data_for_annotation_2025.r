rm(list=ls())

source("it_functions.r")

library(openxlsx)
library(stringr)
library(tidyverse)
library(gridExtra)
library(progress)

base_data_path = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/cameron_may_2024/is_data_xlsx"
output_data_path = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/proc"

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

# NEW IN THIS SCRIPT (VS "0_" version)
# CONVERT TO JSON ANNOTATION FILES FOR MANUAL REVIEW
dim(global_is_data)  # 719 x 2325

head(colnames(global_is_data), 30)
tail(colnames(global_is_data), 30)
print(paste(colnames(global_is_data), collapse=", "))
# Ref_f1, Ref_f2 ... Ref_f8

# reload annotations
annot_path = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/cameron_may_2024/processed_IFE_reports.xlsx"
annots = read.xlsx(annot_path)

post_process_ab_names <- function(s) {
  s <- gsub("[][']+", "", s)
  s <- gsub("-+", " ", s)
  s <- gsub("igg", "IgG", s)
  s <- gsub("iga", "IgA", s)
  s <- gsub("igm", "IgM", s)
  s <- gsub("lambda", "Lambda", s)
  s <- gsub("kappa", "Kappa", s)
  s
}

# create input json files
input_json_directory <- "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/2025/capetown/preannotation/input_jsons"

for (data_position in 1:nrow(global_is_data)) {
  aaid <- global_is_data$id_unique[data_position]
  if (aaid %in% annots$episode) {
    ife_comment <- annots[annots$episode == aaid, ]$ife_comment
    ife_category <- annots[annots$episode == aaid, 4]
    ife_fabs <- annots[annots$episode == aaid, ]$`Heavy-Light.Chain.Pairs`
    ife_fhcs <- annots[annots$episode == aaid, ]$Unpaired.Heavy.Chains
    ife_flcs <- annots[annots$episode == aaid, ]$Unpaired.Light.Chains
    
    ife_fabs <- ifelse(is.na(ife_fabs), "", ife_fabs)
    ife_fhcs <- ifelse(is.na(ife_fhcs), "", ife_fhcs)
    ife_flcs <- ifelse(is.na(ife_flcs), "", ife_flcs)
    
    short_comments <- ife_category
    long_comments <- ""
    if (nchar(ife_fabs) > 2) {
      long_comments <- paste(long_comments, paste("Full antibodies:", post_process_ab_names(ife_fabs)), sep="\n")
    }
    if (nchar(ife_fhcs) > 2) {
      long_comments <- paste(long_comments, paste("Free HCs:", post_process_ab_names(ife_fhcs)), sep="\n")
    }
    if (nchar(ife_flcs) > 2) {
      long_comments <- paste(long_comments, paste("Free LCs:", post_process_ab_names(ife_flcs)), sep="\n")
    }
    long_comments <- paste(long_comments, ife_comment, sep="\n")
    long_comments <- substr(long_comments, 2, nchar(long_comments))
  }
  
  # metadata
  tp <- global_is_data[data_position, "Ref_pt"]
  # mandatory traces
  elp_trace <- as.numeric(global_is_data[data_position, paste0(c("ELP_x"), 1:304)] )
  g_trace <- as.numeric(global_is_data[data_position, paste0(c("IgG_x"), 1:304)] )
  a_trace <- as.numeric(global_is_data[data_position, paste0(c("IgA_x"), 1:304)] )
  m_trace <- as.numeric(global_is_data[data_position, paste0(c("IgM_x"), 1:304)] )
  k_trace <- as.numeric(global_is_data[data_position, paste0(c("K_x"), 1:304)] )
  l_trace <- as.numeric(global_is_data[data_position, paste0(c("L_x"), 1:304)] )
  # ref traces
  ref_trace <- global_is_data[data_position, paste0(c("Ref_x"), 1:304)]
  exactly_1_ref_trace_is_present <- any(elp_trace > 0)
  if (!exactly_1_ref_trace_is_present) {
    elp_trace <- NULL
  }
  # spep (same as ref here (?))
  exactly_1_spe_trace_is_present <- exactly_1_ref_trace_is_present
  spe_trace <- elp_trace
  # spep fractions
  spe_fraction_coords <- as.numeric(global_is_data[data_position, paste0(c("Ref_f"), 1:10)])
  spe_fraction_coords <- spe_fraction_coords[!is.na(spe_fraction_coords)]
  if (length(spe_fraction_coords) > 0) {
    spe_fractions = list(coords=spe_fraction_coords, names=rep("?", length(spe_fraction_coords)-1))
  } else {
    spe_fractions <- NULL
  }
  # peaks
  spe_peak_data = as.numeric(global_is_data[data_position, paste0(c("Ref_p"), 1:10)])
  spe_peak_data <- spe_peak_data[!is.na(spe_peak_data)]
  if (length(spe_peak_data) == 0) {
    spe_peak_data <- NULL
  }

  sample_json_data <- list(paid = "",
                           aaid = aaid,
                           age = -1,
                           sex = "N/A",
                           total_protein = tp,
                           short_comments = short_comments,
                           long_comments = long_comments,
                           traces=list(ELP = list(exists=T, data=elp_trace),
                                       IgG = list(exists=T, data=g_trace),
                                       IgA = list(exists=T, data=a_trace),
                                       IgM = list(exists=T, data=m_trace),
                                       K = list(exists=T, data=k_trace),
                                       L = list(exists=T, data=l_trace),
                                       Ref = list(exists=exactly_1_ref_trace_is_present, data=ref_trace),
                                       SPE = list(exists=exactly_1_spe_trace_is_present, data=spe_trace, fractions=spe_fractions, peaks=spe_peak_data)))
  
  # save at desired location
  write(rjson::toJSON(sample_json_data, indent=2), file.path(input_json_directory, paste0(aaid, ".json")))
}

























